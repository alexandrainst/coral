"""Functions related to the data loading and processing"""

from datasets import DatasetDict
from omegaconf import DictConfig


def load_data(cfg: DictConfig) -> DatasetDict:
    """Load an audio dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        DatasetDict:
            The audio dataset.
    """
    raise NotImplementedError
