"""Unit tests for the `finetune` module."""

from coral.finetune import finetune


def test_finetune(finetuning_config):
    """Test the `finetune` function."""
    finetune(config=finetuning_config)
