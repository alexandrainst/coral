"""Finetune a speech model.

Usage:
    python finetune_model.py <key>=<value> <key>=<value> ...
"""

import hydra
from omegaconf import DictConfig
import os
import logging

from coral_models.finetune import finetune


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
    if os.getenv("WORLD_SIZE") is not None:
        if "layerdrop" in cfg.model and cfg.model.layerdrop != 0.0:
            logger.info(
                "Forcing layerdrop to 0.0 as this is required in a multi-GPU training"
            )
            cfg.model.layerdrop = 0.0
        if cfg.padding != "max_length":
            logger.info(
                "Forcing padding to 'max_length' as this is required in a multi-GPU "
                "training"
            )
            cfg.padding = "max_length"

    finetune(cfg)


if __name__ == "__main__":
    main()
