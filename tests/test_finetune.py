"""Unit tests for the `finetune` module."""

from coral.finetune import finetune


def test_finetune(config):
    """Test the `finetune` function."""
    finetune(config=config)
