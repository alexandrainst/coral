"""Abstract model setups for the different types of models."""

from pathlib import Path
from typing import Callable, Protocol, Type

from omegaconf import DictConfig
from transformers import (
    BatchEncoding,
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)
from transformers.data.data_collator import DataCollatorMixin

from .wav2vec2 import Wav2Vec2ModelSetup


class Processor(Protocol):
    feature_extractor: FeatureExtractionMixin
    tokenizer: PreTrainedTokenizer

    def __init__(
        self, feature_extractor: FeatureExtractionMixin, tokenizer: PreTrainedTokenizer
    ) -> None:
        ...

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> "Processor":
        ...

    def save_pretrained(
        self, save_directory, push_to_hub: bool = False, **kwargs
    ) -> None:
        ...

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        ...

    def decode(self, *args, **kwargs) -> str:
        ...


class ModelSetup(Protocol):
    def __init__(self, cfg: DictConfig) -> None:
        ...

    def load_processor(self) -> Processor:
        ...

    def load_model(self) -> PreTrainedModel:
        ...

    def load_data_collator(self) -> DataCollatorMixin:
        ...

    def load_trainer_class(self) -> Type[Trainer]:
        ...

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        ...


def load_model_setup(cfg: DictConfig) -> ModelSetup:
    """Get the model setup for the given configuration.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        ModelSetup:
            The model setup.
    """
    model_type: str = cfg.model.type
    if model_type == "wav2vec2":
        return Wav2Vec2ModelSetup(cfg)
    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")
