"""Finetune a speech model.

Usage:
    python finetune_model.py <key>=<value> <key>=<value> ...
"""

import hydra
from omegaconf import DictConfig
import os

from coral_models.finetune import finetune


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
        if "layerdrop" in cfg.model:
            cfg.model.layerdrop = 0.0
        cfg.padding = "max_length"

    import logging

    logger = logging.getLogger(__name__)
    logger.info(cfg)
    finetune(cfg)


if __name__ == "__main__":
    main()
