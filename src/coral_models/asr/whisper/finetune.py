"""Functions related to the finetuning of Whisper models on ASR datasets."""

from typing import NoReturn

from omegaconf import DictConfig


def finetune(cfg: DictConfig) -> NoReturn:
    """Finetune a Whisper model on a given ASR dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """
    raise NotImplementedError
