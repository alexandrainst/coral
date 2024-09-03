"""Functions related to the data loading and processing."""

import logging
import multiprocessing as mp
import os
import re
from collections.abc import Callable, Iterable, Sized
from functools import partial
from pathlib import Path
from typing import Any
from unicodedata import normalize

from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    NamedSplit,
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig

from .types import Data
from .utils import convert_iterable_dataset_to_dataset, filter_with_info, map_with_info

logger = logging.getLogger(__package__)


def load_data_for_finetuning(
    config: DictConfig, processor: Callable | None = None
) -> IterableDatasetDict:
    """Load an audio dataset for finetuning.

    Args:
        config:
            The Hydra configuration object.
        processor (optional):
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Defaults to `None`.

    Returns:
        The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    all_datasets: list[IterableDataset] | list[Dataset] = list()
    for dataset_name, dataset_config in config.datasets.items():
        if is_main_process:
            logger.info(f"Loading dataset {dataset_name!r}")

        # Load from disk if the dataset ID is a path and it is stored as an arrow dataset
        if Path(dataset_config.id).exists():
            train_path = Path(dataset_config.id) / dataset_config.train_name
            data_files = list(map(str, train_path.glob("data-*.arrow")))
            if len(data_files) == 0:
                ds = load_dataset(
                    path=dataset_config.id,
                    name=dataset_config.subset,
                    split=dataset_config.train_name,
                    streaming=config.streaming,
                    cache_dir=config.cache_dir,
                )
            else:
                try:
                    ds = load_dataset(
                        "arrow",
                        data_files=data_files,
                        split=dataset_config.train_name,
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )
                except ValueError:
                    ds = load_dataset(
                        "arrow",
                        data_files=data_files,
                        split="train",
                        streaming=config.streaming,
                        cache_dir=config.cache_dir,
                    )

        # Load dataset from the Hugging Face Hub. The HUGGINGFACE_HUB_TOKEN is only
        # used during CI - normally it is expected that the user is logged in to the
        # Hugging Face Hub using the `huggingface-cli login` command.
        else:
            ds = load_dataset(
                path=dataset_config.id,
                name=dataset_config.subset,
                split=dataset_config.train_name,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                streaming=config.streaming,
                trust_remote_code=True,
                cache_dir=config.cache_dir,
            )

        assert isinstance(
            ds, Dataset | IterableDataset
        ), f"Unsupported dataset type: {type(ds)}"

        if dataset_config.text_column != "text":
            ds = ds.rename_column(dataset_config.text_column, "text")
        if dataset_config.audio_column != "audio":
            ds = ds.rename_column(dataset_config.audio_column, "audio")

        ds = ds.remove_columns(
            column_names=[
                column
                for column in ds.column_names or list()
                if column not in ["audio", "text"]
            ]
        ).shuffle(seed=config.seed)

        if config.filter_dataset:
            ds = filter_dataset(
                dataset=ds,
                audio_column="audio",
                min_seconds_per_example=config.min_seconds_per_example,
                max_seconds_per_example=config.max_seconds_per_example,
                is_main_process=is_main_process,
            )

        ds = process_dataset(
            dataset=ds,
            clean_text=config.model.clean_text,
            lower_case=config.model.lower_case,
            characters_to_keep=config.characters_to_keep,
            text_column="text",
            audio_column="audio",
            cast_to_sampling_rate=config.model.sampling_rate,
            processor=processor,
        )

        all_datasets.append(ds)

    assert len(all_datasets) > 0, "No datasets were loaded"

    if len(all_datasets) > 1:
        if is_main_process:
            logger.info("Interleaving datasets")
            if config.dataset_probabilities is None and len(all_datasets) > 1:
                logger.warning(
                    "No dataset probabilities were specified for the training split. "
                    "This means that each dataset will be sampled with equal "
                    "probability, which means that the smaller datasets will be "
                    "sampled more often than the larger datasets. This is probably "
                    "not what you want."
                )

        probabilities = config.dataset_probabilities
        if probabilities is None:
            probabilities = [1 / len(all_datasets)] * len(all_datasets)
            probabilities[-1] = 1 - sum(probabilities[:-1])
        elif sum(probabilities) != 1:
            raise ValueError(
                f"Dataset probabilities must sum to 1, but sum to {sum(probabilities)}"
            )

        train = interleave_datasets(
            datasets=[ds for ds in all_datasets],
            probabilities=probabilities,
            seed=config.seed,
            split=NamedSplit("train"),
            stopping_strategy="all_exhausted",
        )
    else:
        train = all_datasets[0]

    data_dict = dict(train=train)
    dataset = IterableDatasetDict(data_dict)

    if is_main_process:
        logger.info("Loading CoRal validation dataset...")

    val = load_dataset(
        path=config.evaluation_dataset.id,
        split=config.evaluation_dataset.val_name,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        streaming=config.streaming,
        trust_remote_code=True,
    )
    if config.evaluation_dataset.text_column != "text":
        val = val.rename_column(config.evaluation_dataset.text_column, "text")
    if config.evaluation_dataset.audio_column != "audio":
        val = val.rename_column(config.evaluation_dataset.audio_column, "audio")

    val = process_dataset(
        dataset=val,
        clean_text=config.model.clean_text,
        lower_case=config.model.lower_case,
        characters_to_keep=config.characters_to_keep,
        text_column="text",
        audio_column="audio",
        cast_to_sampling_rate=config.model.sampling_rate,
        processor=processor,
    )
    dataset["val"] = val

    return dataset


def load_dataset_for_evaluation(config: DictConfig) -> Dataset:
    """Load the evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DatasetDict containing the validation and test datasets.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    if is_main_process:
        logger.info(
            f"Loading the {config.eval_split_name} split of the CoRal dataset..."
        )

    dataset = load_dataset(
        path="alexandrainst/coral",
        name=config.dataset_subset,
        split=config.eval_split_name,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        trust_remote_code=True,
        cache_dir=config.cache_dir,
        streaming=True,
    )
    assert isinstance(dataset, IterableDataset)
    dataset = filter_dataset(
        dataset=dataset,
        audio_column="audio",
        min_seconds_per_example=config.min_seconds_per_example,
        max_seconds_per_example=config.max_seconds_per_example,
        is_main_process=is_main_process,
    )
    dataset = process_dataset(
        dataset=dataset,
        clean_text=config.clean_text,
        lower_case=config.lower_case,
        characters_to_keep=config.characters_to_keep,
        text_column="text",
        audio_column="audio",
        cast_to_sampling_rate=config.sampling_rate,
    )
    dataset = convert_iterable_dataset_to_dataset(
        iterable_dataset=dataset, split_name=config.eval_split_name
    )
    return dataset


def filter_dataset(
    dataset: Data,
    audio_column: str,
    min_seconds_per_example: int,
    max_seconds_per_example: int,
    is_main_process: bool,
) -> Data:
    """Filter the dataset.

    Note that this removes samples from the dataset.

    Args:
        dataset:
            The dataset to filter.
        audio_column:
            The name of the column containing the audio.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can have.
        is_main_process:
            Whether the current process is the main process.

    Returns:
        The filtered dataset.
    """
    num_samples_before = len(dataset) if isinstance(dataset, Sized) else 0

    filter_fn = partial(
        filter_example,
        audio_column=audio_column,
        min_seconds_per_example=min_seconds_per_example,
        max_seconds_per_example=max_seconds_per_example,
    )
    filtered = filter_with_info(
        dataset=dataset,
        function=filter_fn,
        num_proc=mp.cpu_count(),
        desc="Filtering dataset",
    )

    if isinstance(dataset, Sized) and is_main_process:
        num_samples_removed = num_samples_before - len(dataset)
        logger.info(f"Removed {num_samples_removed:,} samples from the dataset")

    return filtered


def filter_example(
    sample: dict[str, Any],
    audio_column: str,
    min_seconds_per_example: int,
    max_seconds_per_example: int,
) -> bool:
    """Filter samples based on the validation status.

    Args:
        sample:
            The sample to filter.
        audio_column:
            The name of the column containing the audio.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can

    Returns:
        Whether the sample should be kept.
    """
    audio = sample[audio_column]
    if audio["array"].shape[0] <= audio["sampling_rate"] * min_seconds_per_example:
        return False
    if audio["array"].shape[0] >= audio["sampling_rate"] * max_seconds_per_example:
        return False
    return "validated" not in sample or sample["validated"] != "rejected"


def process_dataset(
    dataset: Data,
    clean_text: bool,
    lower_case: bool,
    characters_to_keep: Iterable[str] | None,
    text_column: str,
    audio_column: str | None,
    cast_to_sampling_rate: int | None = None,
    processor: Callable | None = None,
) -> Data:
    """Process the dataset.

    Note that this does not remove any samples from the dataset.

    Args:
        dataset:
            The dataset to be cleaned.
        clean_text:
            Whether to clean the text.
        lower_case:
            Whether to make the text lower case. Only relevant if `clean_text` is True.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept. Only relevant if `clean_text` is True.
        text_column:
            The name of the column containing the text.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        cast_to_sampling_rate:
            The sampling rate to cast the audio to. If `None`, then the audio is not
            cast. Defaults to `None`.
        processor:
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Defaults to `None`.

    Returns:
        The cleaned dataset.
    """
    if audio_column is not None:
        dataset = dataset.cast_column(
            column=audio_column, feature=Audio(sampling_rate=cast_to_sampling_rate)
        )

    # Dictionary that contains characters to be converted (from the key to the value).
    # Some values contain spaces to ensure that they're separated from other
    # characters, and superfluous spaces are removed later. Note also that these are
    # converted in the order they appear in the dictionary.
    conversion_dict = {
        "aa": "å",
        "ğ": "g",
        "ñ": "n",
        "ń": "n",
        "è": "e",
        "kg": " kilo ",
        "μg": " mikrogram ",
        "-": " minus ",
        "+": " plus ",
        "μ": " mikro ",
        "§": " paragraf ",
        "%": " procent ",
        "‰": " promille ",
        "ú": "u",
        "ş": "s",
        "ê": "e",
        "ã": "a",
        "ë": "e",
        "ć": "c",
        "ä": "æ",
        "í": "i",
        "š": "s",
        "î": "i",
        "ě": "e",
        "ð": "d",
        "á": "a",
        "ó": "o",
        "þ": "th",
        "ı": "i",
        "ö": "ø",
        "ç": "c",
        "ș": "s",
        "\u0301": " ",  # Empty whitespace symbol
        "\u200b": " ",  # Empty whitespace symbol
    }

    if isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
        column_names = dataset.column_names
    elif isinstance(dataset, DatasetDict) or isinstance(dataset, IterableDatasetDict):
        column_names = dataset["train"].column_names

    func = partial(
        process_example,
        characters_to_keep=characters_to_keep,
        conversion_dict=conversion_dict,
        text_column=text_column,
        audio_column=audio_column,
        clean_text=clean_text,
        lower_case=lower_case,
        processor=processor,
    )
    mapped = map_with_info(
        dataset=dataset,
        function=func,
        num_proc=mp.cpu_count(),
        desc="Processing dataset",
        remove_columns=column_names,
    )
    return mapped


def process_example(
    example: dict,
    characters_to_keep: Iterable[str] | None,
    conversion_dict: dict[str, str],
    text_column: str,
    audio_column: str | None,
    clean_text: bool,
    lower_case: bool,
    processor: Callable | None,
) -> dict:
    """Helper function which cleans a single example.

    Args:
        example:
            The example to be cleaned.
        characters_to_keep:
            All the characters that should be kept in the transcriptions. Can be None if
            all characters should be kept.
        conversion_dict:
            A dictionary of characters to be converted.
        text_column:
            The name of the column containing the text.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        clean_text:
            Whether to clean the text.
        lower_case:
            Whether to make the text lower case.
        processor:
            The processor to use for processing the audio and transcriptions. If `None`,
            then the processor is not used. Requires `audio_column` to be specified.

    Returns:
        The cleaned example.
    """
    breakpoint()
    doc = example[text_column]

    if lower_case:
        doc = doc.lower()

    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    if clean_text:
        doc = normalize("NFKC", doc)

        for key, value in conversion_dict.items():
            doc = doc.replace(key, value)

        # Remove all non-standard characters
        if characters_to_keep is not None:
            characters_to_keep = "".join(char for char in characters_to_keep)
            if lower_case:
                characters_to_keep = characters_to_keep.lower()
            else:
                characters_to_keep = (
                    characters_to_keep.upper() + characters_to_keep.lower()
                )
            non_standard_characters_regex = re.compile(
                f"[^{re.escape(characters_to_keep + ' |')}]"
            )
            doc = re.sub(non_standard_characters_regex, " ", doc.strip())

        # Replace superfluous spaces
        doc = re.sub(r" +", " ", doc)

        # Strip each newline
        doc = "\n".join([line.strip() for line in doc.split("\n")]).strip("\n")

    # Re-assign the cleaned transcription
    example[text_column] = doc

    if processor is None:
        return example

    # Prepare audio
    audio = example[audio_column]
    sampling_rate = audio["sampling_rate"]
    processed = processor(audio["array"], sampling_rate=sampling_rate)
    if "input_values" in processed:
        example["input_values"] = processed.input_values[0]
        example["num_seconds"] = len(example["input_values"]) / sampling_rate
    if "input_features" in processed:
        example["input_features"] = processed.input_features[0]
        example["num_seconds"] = len(example["input_features"]) / sampling_rate

    # Prepare transcriptions
    example["labels"] = processor(text=example[text_column], truncation=True).input_ids
    example["input_length"] = len(example["labels"])

    return example
