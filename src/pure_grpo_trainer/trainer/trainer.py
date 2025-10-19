import math
import os
from collections.abc import Sized

import torch
from torch.utils.data import DataLoader, Sampler

from transformers import Trainer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, TrainerState
from datasets import Dataset
from typing import Union, Optional, Any
from pure_grpo_trainer.trainer.config import GRPOConfig
from pure_grpo_trainer.trainer.utils import get_last_checkpoint, maybe_apply_chat_template, selective_log_softmax


def empty_data_collator(x):
    return x


TRAINER_STATE_NAME = "trainer_state.json"


class GRPOSampler(Sampler):
    """

    """

    def __init__(
        self,
        data_source: Sized,
        group_size: int,
        distinct_batch_size: int = 1,
        generation_total_steps: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.group_size = group_size
        self.distinct_batch_size = distinct_batch_size
        self.generation_total_steps = generation_total_steps
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (distinct_batch_size = 3)
        indexes = [indexes[i : i + self.distinct_batch_size] for i in range(0, len(indexes), self.distinct_batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.distinct_batch_size]

        for chunk in indexes:
            for _ in range(self.generation_total_steps):
                for index in chunk:
                    for _ in range(self.group_size):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.group_size * self.generation_total_steps


class GRPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs,
            train_dataset: Dataset,
            processing_class,
            optimizer, scheduler,
            eval_dataset: Optional[Dataset] = None,
            ref_model: Optional[Union[str, PreTrainedModel]] = None,
            model_init_kwargs:Optional[dict] = None,
            args: Optional[GRPOConfig] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_in_generation = 0
        self._buffered_generation = None

        # prepare GRPOConfig
        if args is None:
            args = GRPOConfig()

        # prepare model
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # prepare ref_model
        if args.use_ref_model is not None:
            if args.kl_loss_coef is None:  # args.kl_loss_coef * kl_between_self_and_ref
                raise ValueError("kl_loss_coef must be provided if use_ref_model is True.")
            if self.ref_model is None and args.self_as_ref_model is False:
                raise ValueError("ref_model must be provided if self_as_ref_model is False.")
            if self.ref_model is None:
                if args.sync_ref_model_steps is None:
                    raise ValueError("sync_ref_model_steps must be provided if self_as_ref_model is True.")
                # create an isolate copy from the self model as ref_model
                # this ref_model's parameters will be updated every args.sync_ref_model_steps steps
                self.ref_model = AutoModelForCausalLM.from_pretrained(model.config.name_or_path, **model_init_kwargs)

        # prepare processing_class
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(model.config.name_or_path)

        # prepare reward_weights
        if args.reward_weights is None:
            args.reward_weights = torch.ones(len(reward_funcs),dtype=torch.float32)
        else:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError("The number of reward weights must be equal to the number of reward functions.")
            args.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )

    def train(
            self,
            resume_from_checkpoint: Optional[bool] = None,
            **kwargs
    ):
        print("Starting training...")
        # prepare dataloader
        train_dataloader = self.get_train_dataloader(self.train_dataset)
        if self.args.use_eval is True:
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)

        # resume from checkpoint
        # resume_from_checkpoint is just a flag to show whether to use checkpoint.
        # the checkpoint_save_dir of last checkpoint is set in kwargs.
        checkpoint_path = ""
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            checkpoint_save_dir = kwargs.get("checkpoint_save_dir", None)
            checkpoint_path = get_last_checkpoint(checkpoint_save_dir)
            if checkpoint_path is None:
                raise ValueError("No checkpoint found in the checkpoint_save_dir.")

            # load checkpoint if not fsdp or deepspeed
            if not self.is_fsdp_enabled:
                self._load_from_checkpoint(checkpoint_path)
        # get tp_size
        # todo tp_size is model parallel

        # init state
        self.state = TrainerState()
        self.state.train_batch_size = self._train_batch_size

        # wrap fsdp/fsdp2
        most_external_model = self.model_wrapped
        self.model = self._wrap_model(most_external_model)

        # prepare optimizer and lr_scheduler

        # start training
        print("***** Running training *****")
        train_total_loss = torch.tensor(0.0, device=self.args.device)
        train_step = -1
        update_step = -1
        self.state.epoch = 0
        epochs_trained = 0
        steps_in_current_epoch = 0
        dataloader_len = len(train_dataloader)
        total_steps_in_one_epoch = dataloader_len
        update_num_per_epoch = dataloader_len // self.args.gradient_accumulation_steps + int(dataloader_len % self.args.gradient_accumulation_steps > 0)

        if resume_from_checkpoint is not None and resume_from_checkpoint and os.path.isfile(
            os.path.join(checkpoint_path, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(checkpoint_path, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // update_num_per_epoch)
            steps_trained_in_current_epoch = (self.state.global_step % update_num_per_epoch) * self.args.gradient_accumulation_steps

        # clear weight grad
        self.model.zero_grad()

        # start train loop
        for epoch in range(epochs_trained, math.ceil(self.args.num_train_epochs)):
            epoch_dataloader = train_dataloader
            epoch_iterator = iter(epoch_dataloader)
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            for _ in range(update_num_per_epoch):
                update_step += 1
                # todo 这里直接是可以复用底层 get_batch_samples，不过后面可以看看，是否在这个文件直接重新定义一个 get_batch_samples
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, self.args.gradient_accumulation_steps, self.args.device)
                for i, batch_inputs in enumerate(batch_samples):
                    train_step += 1
                    will_update_in_this_step = (train_step + 1) % self.args.gradient_accumulation_steps == 0 or (train_step + 1) == total_steps_in_one_epoch
                    # todo: accelerator _set_sync_gradients ?

                    # 设置训练进度

                    # 开始一步训练：train_step
                    train_step_loss = self._do_train_step(batch_inputs)
                    train_total_loss = train_total_loss + train_step_loss
                    if will_update_in_this_step:
                        print("Updating model...")
                        self.optimizer.step()
                        self.scheduler.step()
                        self.state.global_step += 1

    def _do_train_step(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]):
        print("Starting training...")
        self.model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        # first generation
        # here generation_outputs include self.args.generation_cover_steps * self.args.per_device_batch_size examples
        generation_outputs = self.generate(inputs)
        # second get train_batch_size samples
        # here one_batch_generation_outputs include self.args.per_device_batch_size examples
        one_batch_generation_outputs = self.get_one_batch_from_generation_output(generation_outputs)

        # third compute loss
        self.compute_loss(one_batch_generation_outputs)
        return 0

    def generate(self, inputs: list[dict[str, Union[torch.Tensor, Any]]], **kwargs):
        print("Starting generation...")
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            repeat_train_steps_per_generation = self.args.num_iterations * self.args.generation_cover_steps
            if self.step_in_generation % repeat_train_steps_per_generation == 0 or self._buffered_generation is None:
                # todo _generate
                generation_outputs = self._generate(inputs, **kwargs)
                # todo: _shuffle_generation_outputs
                shuffled_generation_outputs = self._shuffle_generation_outputs(generation_outputs)
                # todo: _split_generation_outputs
                self._buffered_generation = self._split_generation_outputs(shuffled_generation_outputs)
            self.step_in_generation += 1
            # clear step_in_generation, and do _generate in next step
            # todo: 是否采用全局变量累加，不清零更好一点
            if self.step_in_generation == repeat_train_steps_per_generation:
                self.step_in_generation = 0
            return self._buffered_generation
        else:
            return self._generate(inputs, **kwargs)

    def _generate(self, inputs: list[dict[str, Union[torch.Tensor, Any]]], **kwargs):
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(x, self.processing_class)["prompt"] for x in inputs]
        prompt_inputs = self.processing_class(text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_masks = prompt_inputs["attention_mask"]

        # todo: support vLLM
        with torch.no_grad():
            prompt_completion_ids = self.model.generate(prompt_ids, attention_mask=prompt_masks)

            origin_prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :origin_prompt_length]
            completion_ids = prompt_completion_ids[:, origin_prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.args.device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=self.args.device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_length = completion_mask.sum(dim=1)

            completion_ids_list = [[id.item() for id, mask in zip(row, mask_row) if mask] for row, mask_row in zip(completion_ids,completion_mask)]

            # if prompt is [today is a nice day, let's] and completion is [go to the beach]
            # prompt_masks = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] to make sure length = 10
            # completion_mask = [1, 1, 1, 1, 0] to make sure length = 5
            # attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0] to make sure length = 15
            attention_mask = torch.cat([prompt_masks, completion_mask], dim=1)

            # compute old_model and ref_model token logps
            with torch.no_grad():
                if self.args.generation_repeat_num > 1 or self.args.generation_cover_steps > self.args.gradient_accumulation_steps:
                    old_per_token_logps = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask, self.args.logits_to_keep, self.args.per_device_batch_size
                )["logps"]
                else:
                    old_per_token_logps = None
                if self.args.use_ref_model is True:
                    ref_per_token_logps = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, self.args.logits_to_keep, self.args.per_device_batch_size
                    )["logps"]
                else:
                    ref_per_token_logps = None

                rewards_per_rewardfun = self._compute_rewards()

                rewards = (rewards_per_rewardfun * self.args.reward_weights.to(self.args.device).unsqueeze(0)).nansum(dim=1)

                deduplicated_group_rewards_mean = rewards.view(-1, self.args.generation_cover_steps).mean(dim=1)
                deduplicated_group_rewards_std = rewards.view(-1, self.args.generation_cover_steps).std(dim=1)

                duplicated_group_rewards_mean = deduplicated_group_rewards_mean.repeat_interleave(self.args.generation_cover_steps, dim=0)
                duplicated_group_rewards_std = deduplicated_group_rewards_std.repeat_interleave(self.args.generation_cover_steps, dim=0)

                advantages = rewards - duplicated_group_rewards_mean
                if self.args.scale_rewards:
                    advantages = advantages / (deduplicated_group_rewards_std + 1e-8)

                # todo: process slice

                return {
                    "prompt_ids": prompt_ids,
                    "prompt_mask": prompt_masks,
                    "completion_ids": completion_ids,
                    "completion_mask": completion_mask,
                    "advantages": advantages,
                    "old_per_token_logps": old_per_token_logps,
                    "ref_per_token_logps": ref_per_token_logps,
                }

    def _get_per_token_logps_and_entropies(
            self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False
    ) -> dict[str, Optional[torch.Tensor]]:
        all_log_probs = []
        for i in range(0, input_ids.size(0), batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            batch_attention_mask = attention_mask[i : i + batch_size]
            batch_logits = model(batch_input_ids, attention_mask=batch_attention_mask, logits_to_keep=logits_to_keep + 1).logits

            batch_logits = batch_logits[:, :-1, :]
            batch_logits = batch_logits / self.args.temperature

            log_probs = selective_log_softmax(batch_logits, batch_input_ids[:, -logits_to_keep:])

            all_log_probs.append(log_probs)

        logps = torch.cat(all_log_probs, dim=0)
        # todo: entropies
        return {"logps": logps, "entropies": None}

    def get_one_batch_from_generation_output(self, generation_outputs):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generation_output = generation_outputs[self.step_in_generation % self.args.generation_cover_steps]
            return generation_output
        else:
            return generation_outputs

    def compute_loss(self, inputs: dict[str, Union[torch.Tensor, Any]], **kwargs):
        print("Starting compute_loss...")
        prompt_completion_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=0)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=0)
        logits_to_keep = inputs["completion_ids"].size(1)

        # todo: entropy threshold mask
        per_token_logps = self._get_per_token_logps_and_entropies(
            self.model, prompt_completion_ids, attention_mask, logits_to_keep, self.args.per_device_batch_size
        )["logps"]

        if self.args.use_ref_model is True:
            diff_with_ref = inputs["ref_per_token_logps"] - per_token_logps
            per_token_kl_with_ref = (torch.exp(diff_with_ref) - diff_with_ref - 1)

        # compute ratio based on: exp(log(π_new(a|s)) - log(π_old(a|s))) = π_new(a|s)/π_old(a|s)
        ratio = torch.exp(per_token_logps - inputs["old_per_token_logps"])
        clamped_ratio = torch.clamp(ratio, 1 - self.args.cliprange_low_value, 1 + self.args.cliprange_high_value)

        per_token_loss = torch.min(
            ratio * inputs["advantages"].unsqueeze(1),
            clamped_ratio * inputs["advantages"].unsqueeze(1),
        )

        if self.args.use_ref_model is True:
            per_token_loss = per_token_loss + per_token_kl_with_ref * self.args.kl_coef

        loss = ((per_token_loss * inputs["completion_mask"]).sum(-1) / inputs["completion_mask"].sum(-1).clamp(min=1.0)).mean()
        return loss

    def get_train_dataloader(
            self,
            dataset: Dataset = None,
    ):
        print("Starting training dataloader...")

        dataloader_params = {
            # self.args.train_batch_size is per_device_batch_size * max(1, self.n_gpu)
            "batch_size": self.args.train_batch_size * self.args.generation_cover_steps,
            "collate_fn": empty_data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "sampler": self._get_train_sampler(),
        }
        dataloader = DataLoader(dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)

    def _get_train_sampler(self, dataset: Dataset = None):
        if dataset is None:
            dataset = self.train_dataset

        return GRPOSampler(
            data_source=dataset,
            group_size=self.args.group_size,
            distinct_batch_size=self.args.generation_batch_size // self.args.group_size,
            generation_total_steps=self.args.generation_repeat_num * self.args.generation_cover_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )


