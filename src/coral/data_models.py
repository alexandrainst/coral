"""Data models used throughout the project."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Type, TypeAlias

from omegaconf import DictConfig
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    WhisperProcessor,
)
from transformers.data.data_collator import DataCollatorMixin

Processor: TypeAlias = Wav2Vec2Processor | Wav2Vec2ProcessorWithLM | WhisperProcessor


@dataclass
class PreTrainedModelData:
    """Data class for the pre-trained model and related objects.

    Attributes:
        model:
            The pre-trained model.
        processor:
            The processor used to process the data.
        data_collator:
            The data collator used to collate the data.
        compute_metrics:
            The function used to compute the metrics.
    """

    model: PreTrainedModel
    processor: Processor
    data_collator: DataCollatorMixin
    compute_metrics: Callable[[EvalPrediction], dict]


class ModelSetup(ABC):
    """Base class for a model setup."""

    @abstractmethod
    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The configuration object.
        """

    @abstractmethod
    def load_processor(self) -> Processor:
        """Return the processor for the model."""

    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        """Return the pre-trained model."""

    @abstractmethod
    def load_data_collator(self) -> DataCollatorMixin:
        """Return the data collator."""

    @abstractmethod
    def load_trainer_class(self) -> Type[Trainer]:
        """Return the trainer class."""

    @abstractmethod
    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        """Return the function used to compute the metrics."""

    @abstractmethod
    def load_training_arguments(self) -> TrainingArguments:
        """Return the training arguments."""

    @abstractmethod
    def load_saved(self) -> PreTrainedModelData:
        """Return the saved model data."""
