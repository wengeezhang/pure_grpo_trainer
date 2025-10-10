from typing import Union, Optional
import torch

from transformers import Trainer, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

from pure_grpo_trainer.trainer.config import GRPOConfig


class GRPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs,
            train_dataset, eval_dataset,
            processing_class,
            optimizer, scheduler,
            args: Optional[GRPOConfig] = None,
    ):
        self.model = model
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
            model = AutoModelForCausalLM.from_pretrained(model)

        # prepare processing_class
        if self.processing_class is None:
            self.processing_class = AutoTokenizer.from_pretrained(model.config.name_or_path)

        # prepare reward_weights
        if args.reward_weights is None:
            args.reward_weights = torch.ones(len(reward_funcs),dtype=torch.float32)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )

    def train(self):
        print("Starting training...")

