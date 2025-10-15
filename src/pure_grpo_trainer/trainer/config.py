from dataclasses import field
from typing import Optional, List

from transformers import TrainingArguments


class GRPOConfig(TrainingArguments):
    group_size: int = field(
        default=8,
        metadata={
            "help": (
                "The group size for the GRPO. "
            )
        }
    )
    generation_repeat_num: int = field(
        default=2,
        metadata={
            "help": (
                "The number of iterations for the generation. "
                "that means one generation will be used `generation_repeat_num` times."
            )
        }
    )
    generation_cover_steps: int = field(
        default=None,
        metadata={
            "help": (
                "The number of steps one generation samples can cover. "
                "e.g. if per_device_eval_batch_size=10, which means one step need 10 samples. "
                "if we want cover 10 steps in one generation, "
                "we need 100 (per_device_eval_batch_size * generation_cover_steps) samples. "
                "so this arg is used to calculate the `generation_batch_size` which is not set by user. "
            )
        }
    )
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The batch size for one generation. "
                "it can not be set by user, only calculated by `per_device_eval_batch_size * generation_cover_steps`. "
            )
        }
    )
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
    save_checkpoint_strategy: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "The checkpoint strategy. "
                "e.g. ['new_best_eval', 'epochs']"
            )
        }
    )
