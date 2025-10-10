from typing import Union

from transformers import Trainer, PreTrainedModel


class GRPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs,
            train_dataset, eval_dataset,
            processing_class,
            optimizer, scheduler,
            args
    ):
        self.model = model
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.scheduler = scheduler

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
        )

    def train(self):
        print("Starting training...")

