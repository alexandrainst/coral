"""Unit tests for the `finetune` module."""

from coral.finetune import finetune


def test_finetune(cfg):
    finetune(cfg)
