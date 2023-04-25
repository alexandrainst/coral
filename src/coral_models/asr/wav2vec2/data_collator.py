"""Data collator used in training Wav2Vec 2.0 models."""

from dataclasses import dataclass

import torch
from transformers import BatchEncoding, BatchFeature, Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor (Wav2Vec2Processor)
            The processor used for proccessing the data.
        padding (bool, str or PaddingStrategy, optional):
            Select a strategy to pad the returned sequences (according to the
            model's padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only
                a single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length
                or to the maximum acceptable input length for the model if that
                argument is not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of
                different lengths).
            Defaults to True.
    """

    processor: Wav2Vec2Processor
    padding: bool | str = True

    def __call__(self, features: list[dict]) -> BatchFeature:
        """Collate the features.

        Args:
            features (list of dict):
                A list of feature dicts.

        Returns:
            BatchFeature:
                A dictionary of the collated features.
        """
        # Split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {
                "input_values": self.processor(
                    feature["audio"]["array"],
                    sampling_rate=feature["audio"]["sampling_rate"],
                ).input_values[0]
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Process audio
        batch: BatchFeature = self.processor.pad(  # type: ignore
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Process labels
        with self.processor.as_target_processor():
            labels_batch: BatchEncoding = self.processor.pad(  # type: ignore
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        non_one_entries: torch.Tensor = labels_batch.attention_mask.ne(1)
        labels: torch.Tensor = labels_batch.input_ids.masked_fill(non_one_entries, -100)

        # Update the batch labels
        batch["labels"] = labels

        # Return the updated batch
        return batch
