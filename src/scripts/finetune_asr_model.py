"""Finetune a speech recognition model.

Usage:
    Running on a single GPU:

    python src/scripts/finetune_asr_model.py \
        [key=value] [key=value] ...

    Running on multiple GPUs:

    accelerate launch [--use-deepspeed] src/scripts/finetune_asr_model.py \
        [key=value] [key=value] ...
"""

import datetime as dt
import logging
import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from coral.finetune import finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("coral_finetuning")


load_dotenv()


@hydra.main(config_path="../../config", config_name="asr_finetuning", version_base=None)
def main(config: DictConfig) -> None:
    """Finetune an ASR model.

    Args:
        config:
            The Hydra configuration object.
    """
    # In case we are running in a multi-GPU setting, we need to force certain
    # hyperparameters
    is_main_process = os.getenv("RANK", "0") == "0"
    if os.getenv("WORLD_SIZE") is not None:
        logger.info(
            f"Setting PyTorch distributed timeout to {config.distributed_timeout_mins} "
            "minutes"
        )
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=dt.timedelta(minutes=config.distributed_timeout_mins),
        )
        if "layerdrop" in config.model and config.model.layerdrop != 0.0:
            if is_main_process:
                logger.info(
                    "Forcing `layerdrop` to be 0.0 as this is required in a multi-GPU "
                    "training"
                )
            config.model.layerdrop = 0.0

    elif torch.cuda.device_count() > 1:
        if is_main_process:
            logger.info(
                "You seem to be running on multiple GPUs, but not running the script "
                "with `accelerate`. This will result in slower training. To use "
                "`accelerate`, run the script with `accelerate launch "
                "[--use-deepspeed] src/scripts/finetune_asr_model.py [key=value] "
                "[key=value] ...`"
            )
        if "gradient_checkpointing" in config and config.gradient_checkpointing is True:
            if is_main_process:
                logger.info(
                    "Disabling gradient checkpointing as this is required in a multi-"
                    "GPU training without `accelerate`"
                )
            config.gradient_checkpointing = False

    finetune(config=config)


if __name__ == "__main__":
    main()
