import os

import torch
from torch.utils.data import DataLoader

from transformers import Trainer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, TrainerState
from datasets import Dataset
from typing import Union, Optional
from pure_grpo_trainer.trainer.config import GRPOConfig
from pure_grpo_trainer.trainer.utils import get_last_checkpoint


def empty_data_collator(x):
    return x


TRAINER_STATE_NAME = "trainer_state.json"

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
        self.state.epoch = 0

        if resume_from_checkpoint is not None and resume_from_checkpoint and os.path.isfile(
            os.path.join(checkpoint_path, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(checkpoint_path, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)

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
        }
        dataloader = DataLoader(dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)


