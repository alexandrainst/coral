"""Module related to validating a dataset using an ASR model."""

import logging
from collections.abc import Iterable
from typing import TypeVar

import torch
from datasets import Audio, Dataset, DatasetDict
from transformers import AutomaticSpeechRecognitionPipeline, pipeline

from .compute_metrics import compute_metrics_of_dataset_using_pipeline
from .data import process_dataset

logger = logging.getLogger(__package__)


NonIterableData = TypeVar("NonIterableData", bound=Dataset | DatasetDict)


def add_validations(
    dataset: NonIterableData,
    text_column: str,
    audio_column: str,
    model_id: str,
    clean_text: bool,
    lower_case: bool,
    sampling_rate: int,
    characters_to_keep: Iterable[str],
    batch_size: int,
    max_cer: float,
) -> NonIterableData:
    """Add the ASR validation columns to the dataset.

    Args:
        dataset:
            The dataset to add the validation columns to.
        text_column:
            The name of the column containing the transcriptions.
        audio_column:
            The name of the column containing the audio samples.
        model_id:
            The ID of the ASR model to use for validation.
        clean_text:
            Whether to clean the text before transcribing.
        lower_case:
            Whether to lower case the transcriptions.
        sampling_rate:
            The sampling rate to use for the audio samples.
        characters_to_keep:
            The characters to keep in the transcriptions.
        batch_size:
            The batch size to use for transcribing the audio.
        max_cer:
            The maximum CER value to keep the samples.

    Returns:
        The dataset with the validation columns added.
    """
    input_is_single_split = isinstance(dataset, Dataset)
    if input_is_single_split:
        dataset = DatasetDict(dict(train=dataset))

    dataset = dataset.cast_column(
        column=audio_column, feature=Audio(sampling_rate=sampling_rate)
    )

    processed_dataset = process_dataset(
        dataset=dataset,
        clean_text=clean_text,
        characters_to_keep=characters_to_keep,
        text_column=text_column,
        audio_column=audio_column,
        convert_numerals=False,
        remove_input_dataset_columns=True,
        lower_case=lower_case,
    )

    logger.info(f"Loading the {model_id!r} ASR model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    transcriber = pipeline(
        task="automatic-speech-recognition", model=model_id, device=device
    )
    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)

    for split_name, split in processed_dataset.items():
        logger.info(f"Validating the {split_name} split of the dataset...")
        predictions, labels, score_dict = compute_metrics_of_dataset_using_pipeline(
            dataset=split,
            transcriber=transcriber,
            metric_names=["cer"],
            characters_to_keep=characters_to_keep,
            text_column=text_column,
            audio_column=audio_column,
            batch_size=batch_size,
        )

        # Create a new split with the predictions, labels, and scores
        dataset[split_name] = (
            dataset[split_name]
            .add_column(name="asr_prediction", column=predictions)
            .add_column(name="asr_label", column=labels)
            .add_column(name="asr_validation_model", column=[model_id] * len(split))
        )
        for metric_name, scores in score_dict.items():
            dataset[split_name] = dataset[split_name].add_column(
                name=f"asr_{metric_name.lower()}", column=scores
            )

    # Filter the dataset based on the metrics from the validation model
    num_samples_before = sum(len(split) for split in dataset.values())
    dataset = dataset.filter(
        lambda sample: sample["asr_cer"] < max_cer,
        desc=f"Removing samples with CER >= {max_cer}",
    )
    num_samples_removed = num_samples_before - sum(
        len(split) for split in dataset.values()
    )
    logger.info(
        f"Removed {num_samples_removed:,} samples based on the validation model."
    )

    if input_is_single_split:
        dataset = dataset["train"]

    return dataset
