"""Abstract model setups for the different types of models."""

from dataclasses import dataclass
from typing import Callable, Protocol, Type

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from transformers import (
    BatchEncoding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm import (
    Wav2Vec2DecoderWithLMOutput,
)

from .wav2vec2 import Wav2Vec2ModelSetup


class Processor(Protocol):
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        ...

    def decode(
        self,
        logits: NDArray[np.float_],
        beam_width: int | None = None,
        beam_prune_logp: float | None = None,
        token_min_logp: float | None = None,
        hotwords: list[str] | None = None,
        hotword_weight: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        unk_score_offset: float | None = None,
        lm_score_boundary: bool | None = None,
        output_word_offsets: bool = False,
        n_best: int = 1,
    ) -> Wav2Vec2DecoderWithLMOutput | str:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "Processor":
        ...

    def save_pretrained(self, save_directory: str) -> None:
        ...


@dataclass
class PreTrainedModelData:
    model: PreTrainedModel
    processor: Processor
    data_collator: DataCollatorMixin
    compute_metrics: Callable[[EvalPrediction], dict]


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

    def load_saved(self) -> PreTrainedModelData:
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
