"""Finetune a speech model.

Usage:
    python finetune_model.py <key>=<value> <key>=<value> ...
"""

import hydra
from omegaconf import DictConfig
import os
import logging

from coral.finetune import finetune


logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Finetune an ASR model.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """
    # In case we are running in a multi-GPU setting, we need to force certain
    # hyperparameters
    is_main_process = os.getenv("RANK", "0") == "0"
    if os.getenv("WORLD_SIZE") is not None:
        if "layerdrop" in cfg.model and cfg.model.layerdrop != 0.0:
            if is_main_process:
                logger.info(
                    "Forcing `layerdrop` to be 0.0 as this is required in a multi-GPU "
                    "training"
                )
            cfg.model.layerdrop = 0.0
        if cfg.padding != "max_length":
            if is_main_process:
                logger.info(
                    "Forcing `padding` to be 'max_length' as this is required in a "
                    "multi-GPU training"
                )
            cfg.padding = "max_length"

    finetune(cfg)


if __name__ == "__main__":
    main()
