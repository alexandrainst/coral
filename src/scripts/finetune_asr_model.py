"""Finetune a speech recognition model.

Usage:
    python src/scripts/finetune_asr_model.py [key=value] [key=value] ...
"""

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
    if os.getenv("WORLD_SIZE") is not None or torch.cuda.device_count() > 1:
        if "gradient_checkpointing" in config and config.gradient_checkpointing is True:
            if is_main_process:
                logger.info(
                    "Disabling gradient checkpointing as this is required in a multi-"
                    "GPU training"
                )
            config.gradient_checkpointing = False

    finetune(config=config)


if __name__ == "__main__":
    main()
