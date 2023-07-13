"""Finetune a speech model.

Usage:
    python finetune.py <key>=<value> <key>=<value> ...
"""

import hydra
from omegaconf import DictConfig

from coral_models import finetune


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Finetune an ASR model.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """
    finetune(cfg)


if __name__ == "__main__":
    main()
