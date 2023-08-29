"""Unit tests for the `finetune` module."""

from coral_models.finetune import finetune


def test_finetune(cfg):
    finetune(cfg)
