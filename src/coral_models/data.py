"""Functions related to the data loading and processing"""

from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig


def load_data(cfg: DictConfig) -> DatasetDict:
    """Load an audio dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        DatasetDict:
            The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    # Load the dataset
    subset: str | None = None if cfg.dataset.subset == "" else cfg.dataset.subset
    dataset = load_dataset(path=cfg.dataset.id, name=subset)

    # Check if the dataset class is supported
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Unsupported dataset class: {type(dataset)}")

    # Only include the train, validation and test splits of the dataset
    return DatasetDict(
        dict(
            train=dataset[cfg.dataset.train_name],
            val=dataset[cfg.dataset.val_name],
            test=dataset[cfg.dataset.test_name],
        )
    )
