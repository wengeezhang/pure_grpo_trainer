import torch

from transformers import Trainer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import Union, Optional
from pure_grpo_trainer.trainer.config import GRPOConfig


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
            **kwargs
    ):
        print("Starting training...")
        # prepare dataloader
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()

    def get_train_dataloader(
            self,
            dataset: Dataset = None,
    ):
        print("Starting training dataloader...")