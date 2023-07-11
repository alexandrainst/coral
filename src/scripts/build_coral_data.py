"""Script that preprocesses the raw CoRal data.

Usage:
    python build_coral_data.py
"""
import warnings

import hydra
from omegaconf import DictConfig

from coral_models import prepare_raw_data


@hydra.main(config_path="config/dataset", config_name="coral", version_base=None)
def main(cfg: DictConfig) -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    prepare_raw_data(cfg)


if __name__ == "__main__":
    main()
