"""Data collators for the models."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_audiomentations as ta
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

from .data_models import Processor

logger = logging.getLogger(__package__)


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor:
            The processor used for proccessing the data.
        sample_rate:
            The sample rate that the audio is in.
        max_seconds_per_example:
            The maximum number of seconds per example.
        padding:
            Select a strategy to pad the returned sequences (according to the model's
            padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only a
                single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length or to
                the maximum acceptable input length for the model if that argument is
                not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of different
                lengths).
    """

    processor: Processor
    sample_rate: int
    max_seconds_per_example: float
    padding: bool | str
    return_tensors: str = "pt"
    training: bool = False
    augmenter = ta.Compose(
        [
            ta.Gain(p=1.0),
            ta.AddBackgroundNoise(background_paths=Path("background-noises"), p=0.7),
            ta.AddColoredNoise(p=0.2),
            ta.OneOf(
                [
                    ta.BandPassFilter(p=1.0),
                    ta.BandStopFilter(p=1.0),
                    ta.HighPassFilter(p=1.0),
                    ta.LowPassFilter(p=1.0),
                ],
                p=0.2,
            ),
        ],
        p=1.0,
    )

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate the features.

        Args:
            features:
                A list of feature dicts.

        Returns:
            BatchFeature:
                A dictionary of the collated features.
        """
        if "input_values" in features[0]:
            audio_features = [dict(input_values=f["input_values"]) for f in features]
        elif "audio" in features[0]:
            audio_features = [dict(input_values=f["audio"]["array"]) for f in features]
        else:
            raise ValueError(
                "Features must contain either 'input_values' or 'audio' key."
            )

        # Get the batch
        batch: BatchFeature = self.processor.pad(  # type: ignore[union-attr]
            audio_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=int(self.sample_rate * self.max_seconds_per_example),
        )

        # Augment the audio
        if self.training:
            input_column = "input_values"
            inputs: torch.Tensor = batch[input_column]
            is_2d = inputs.dim() == 2
            if is_2d:
                inputs = inputs.unsqueeze(1)  # Add channel dimension
            augmented_inputs = self.augmenter(inputs, sample_rate=self.sample_rate)
            if is_2d:
                augmented_inputs = augmented_inputs.squeeze(1)  # Remove channel dim
            batch[input_column] = augmented_inputs

        label_features = [dict(input_ids=feature["labels"]) for feature in features]
        labels_batch: BatchEncoding = self.processor.pad(  # type: ignore[union-attr]
            labels=label_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=min(self.processor.tokenizer.model_max_length, 512),  # type: ignore[union-attr]
        )

        # Replace padding with -100 to ignore loss correctly
        non_one_entries: torch.Tensor = labels_batch.attention_mask.ne(1)
        labels: torch.Tensor = labels_batch.input_ids.masked_fill(non_one_entries, -100)

        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor:
            The processor used for proccessing the data.
        sample_rate:
            The sample rate that the audio is in.
        max_seconds_per_example:
            The maximum number of seconds per example.
        padding:
            Select a strategy to pad the returned sequences (according to the model's
            padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only a
                single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length or to
                the maximum acceptable input length for the model if that argument is
                not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of different
                lengths).
    """

    processor: Processor
    sample_rate: int
    max_seconds_per_example: float
    padding: bool | str
    return_tensors: str = "pt"
    training: bool = False
    augmenter = ta.Compose(
        [
            ta.Gain(p=1.0),
            ta.AddBackgroundNoise(background_paths=Path("background-noises"), p=0.7),
            ta.AddColoredNoise(p=0.2),
            ta.OneOf(
                [
                    ta.BandPassFilter(p=1.0),
                    ta.BandStopFilter(p=1.0),
                    ta.HighPassFilter(p=1.0),
                    ta.LowPassFilter(p=1.0),
                ],
                p=0.2,
            ),
        ],
        p=1.0,
    )

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate the features.

        Args:
            features:
                A list of feature dicts.

        Returns:
            BatchFeature:
                A dictionary of the collated features.
        """
        if "input_features" in features[0]:
            audio_features = [
                dict(input_features=f["input_features"]) for f in features
            ]
        elif "audio" in features[0]:
            audio_features = [dict(audio=f["audio"]["array"]) for f in features]
        else:
            raise ValueError(
                "Features must contain either 'input_features' or 'audio' key."
            )

        # Get the batch
        batch = self.processor.feature_extractor.pad(  # type: ignore[union-attr]
            audio_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=int(self.sample_rate * self.max_seconds_per_example),
        )

        # Normalise and augment the audio
        input_column = "input_features"
        inputs: torch.Tensor = batch[input_column]
        is_2d = inputs.dim() == 2
        if is_2d:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
        if self.training:
            inputs = self.augmenter(inputs, sample_rate=self.sample_rate)
        if is_2d:
            inputs = inputs.squeeze(1)  # Remove channel dim again
        batch[input_column] = inputs

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(  # type: ignore[union-attr]
            label_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=min(self.processor.tokenizer.model_max_length, 512),  # type: ignore[union-attr]
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step, cut BOS token here as
        # it's appended later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():  # type: ignore[union-attr]
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
