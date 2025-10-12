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
    use_ref_model: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the reference model. "
            )
        }
    )
    self_as_ref_model: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the self model as the reference model. "
                "must be provided if use_ref_model is True, but without position arg 'ref_model'."
            )
        }
    )
    sync_ref_model_steps: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The number of steps to update/sync the reference model. "
                "must be provided if self_as_ref_model is True. "
            )
        }
    )
    kl_loss_coef: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "The coefficient of the KL loss. "
                "must be provided if use_ref_model is True. "
                "or it will not be used. "
            )
        }
    )
    use_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the eval during training. "
            )
        }
    )
    eval_strategy: Optional[dict[str, int]] = field(
        default=None,
        metadata={
            "help": (
                "The eval strategy. "
                "must be provided if use_eval is True. "
                "e.g. {'steps': 512} or {'epochs': 1}"
            )
        }
    )
