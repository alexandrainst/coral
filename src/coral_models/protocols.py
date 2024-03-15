"""Protocols used throughout the project."""

from dataclasses import dataclass
from typing import Callable, Protocol, Type

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from transformers import (
    BatchEncoding,
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm import (
    Wav2Vec2DecoderWithLMOutput,
)


class Processor(Protocol):
    def __call__(self, *args, **kwargs) -> BatchEncoding: ...

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
    ) -> Wav2Vec2DecoderWithLMOutput | str: ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "Processor": ...

    def save_pretrained(self, save_directory: str) -> None: ...


@dataclass
class PreTrainedModelData:
    model: PreTrainedModel
    processor: Processor
    data_collator: DataCollatorMixin
    compute_metrics: Callable[[EvalPrediction], dict]


class ModelSetup(Protocol):
    def __init__(self, cfg: DictConfig) -> None: ...

    def load_processor(self) -> Processor: ...

    def load_model(self) -> PreTrainedModel: ...

    def load_data_collator(self) -> DataCollatorMixin: ...

    def load_trainer_class(self) -> Type[Trainer]: ...

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]: ...

    def load_training_arguments(self) -> TrainingArguments: ...

    def load_saved(self) -> PreTrainedModelData: ...
