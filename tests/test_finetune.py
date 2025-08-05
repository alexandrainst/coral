"""Unit tests for the `finetune` module."""

from omegaconf import DictConfig

from coral.finetune import finetune


def test_finetune(finetuning_config: DictConfig) -> None:
    """Test the `finetune` function."""
    finetune(config=finetuning_config)
