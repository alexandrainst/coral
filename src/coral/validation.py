"""Module related to validating a dataset using an ASR model."""

import logging
from collections.abc import Iterable

import torch
from datasets import Audio, Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers.pipelines import pipeline
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.pipelines.base import KeyDataset

from coral.data import DEFAULT_CONVERSION_DICT, process_example
from coral.utils import transformers_output_ignored

from .data import filter_dataset, process_dataset
from .metrics import cer, wer

logger = logging.getLogger(__package__)


def add_validations(
    dataset: Dataset | DatasetDict,
    text_column: str,
    audio_column: str,
    model_id: str,
    lower_case: bool,
    sampling_rate: int,
    characters_to_keep: Iterable[str],
    batch_size: int,
    max_cer: float,
) -> Dataset | DatasetDict:
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

    # We have to filter the dataset for audioclips that are too small or too big to ASR
    # validate
    dataset = filter_dataset(
        dataset=dataset,
        audio_column=audio_column,
        text_column=text_column,
        min_seconds_per_example=0.25,
        max_seconds_per_example=60 * 60,
        is_main_process=False,
    )

    processed_dataset = process_dataset(
        dataset=dataset,
        lower_case=lower_case,
        characters_to_keep=characters_to_keep,
        text_column=text_column,
        audio_column=audio_column,
        convert_numerals=False,
        remove_input_dataset_columns=False,
        normalise_audio=True,
        augment_audio=False,
    )
    assert isinstance(dataset, DatasetDict)

    logger.info(f"Loading the {model_id!r} ASR model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    transcriber = pipeline(
        task="automatic-speech-recognition", model=model_id, device=device
    )
    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)

    for split_name, split in processed_dataset.items():
        logger.info(f"Validating the {split_name} split of the dataset...")

        predictions: list[str] = list()
        with (
            tqdm(  # pyrefly: ignore[bad-context-manager]
                total=len(split), desc="Transcribing"
            ) as pbar,
            transformers_output_ignored(),
        ):
            for out in transcriber(
                KeyDataset(  # pyrefly: ignore[bad-argument-type,not-callable]
                    dataset=split, key=audio_column
                ),
                batch_size=batch_size,
                generate_kwargs=dict(language="danish", task="transcribe"),
            ):
                prediction = process_example(
                    example=dict(text=out["text"]),
                    characters_to_keep="".join(characters_to_keep),
                    conversion_dict=DEFAULT_CONVERSION_DICT,
                    text_column="text",
                    audio_column=None,
                    lower_case=True,
                    convert_numerals=True,
                    processor=None,
                    normalise_audio=True,
                    augment_audio=False,
                )["text"]
                predictions.append(prediction)
                pbar.update()

        # Compute the error rates
        score_dict = dict(
            cer=cer(predictions=predictions, labels=split[text_column]),
            wer=wer(predictions=predictions, labels=split[text_column]),
        )

        # Create a new split with the predictions, labels, and scores
        dataset[split_name] = (
            dataset[split_name]
            .add_column(name="asr_prediction", column=predictions)
            .add_column(name="asr_label", column=split[text_column])
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
