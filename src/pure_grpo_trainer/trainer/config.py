from dataclasses import field
from typing import Optional, List

from transformers import TrainingArguments


class GRPOConfig(TrainingArguments):
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": (
                "The weights of the reward functions. "
            )
        }
    )
