"""Finetune a speech recognition model.

Usage:
    python src/scripts/finetune_asr_model.py [key=value] [key=value] ...
"""

import logging
import os

import hydra
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
        if "gradient_checkpointing" in config and config.gradient_checkpointing is True:
            if is_main_process:
                logger.info(
                    "Disabling gradient checkpointing as this is required in a multi-"
                    "GPU training"
                )
            config.gradient_checkpointing = False
        if "layerdrop" in config.model and config.model.layerdrop != 0.0:
            if is_main_process:
                logger.info(
                    "Forcing `layerdrop` to be 0.0 as this is required in a multi-GPU "
                    "training"
                )
            config.model.layerdrop = 0.0
        if config.padding != "max_length":
            if is_main_process:
                logger.info(
                    "Forcing `padding` to be 'max_length' as this is required in a "
                    "multi-GPU training"
                )
            config.padding = "max_length"

    finetune(config=config)


if __name__ == "__main__":
    main()
