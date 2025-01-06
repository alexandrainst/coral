"""Data collators for the models."""

from dataclasses import dataclass

import torch
from transformers import BatchEncoding, BatchFeature
from transformers.data.data_collator import DataCollatorMixin

from .data_models import Processor


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor:
            The processor used for proccessing the data.
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
            Defaults to True.
    """

    processor: Processor
    max_seconds_per_example: float
    padding: bool | str
    return_tensors: str = "pt"

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
        batch: BatchFeature = self.processor.pad(
            audio_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=16_000 * self.max_seconds_per_example,
        )

        label_features = [dict(input_ids=feature["labels"]) for feature in features]
        labels_batch: BatchEncoding = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=min(self.processor.tokenizer.model_max_length, 512),
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
            Defaults to True.
    """

    processor: Processor
    max_seconds_per_example: float
    padding: bool | str = True
    return_tensors: str = "pt"

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
        batch = self.processor.feature_extractor.pad(
            audio_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=16_000 * self.max_seconds_per_example,
        )

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=min(self.processor.tokenizer.model_max_length, 512),
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step, cut BOS token here as
        # it's appended later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
