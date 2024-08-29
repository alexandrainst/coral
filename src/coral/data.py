"""Functions related to the data loading and processing."""

import logging
import multiprocessing as mp
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, TypeVar
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

logger = logging.getLogger(__package__)


Data = TypeVar(
    "Data", bound=Dataset | IterableDataset | DatasetDict | IterableDatasetDict
)


def load_data_for_finetuning(config: DictConfig) -> IterableDatasetDict:
    """Load an audio dataset for finetuning.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    all_datasets: list[IterableDataset] = list()
    for dataset_name, dataset_config in config.datasets.items():
        if is_main_process:
            logger.info(f"Loading dataset {dataset_name!r}")

        # Load from disk if the dataset ID is a path
        if Path(dataset_config.id).exists():
            train_path = Path(dataset_config.id) / dataset_config.train_name
            data_files = list(map(str, train_path.glob("data-*.arrow")))
            if len(data_files) == 0:
                raise FileNotFoundError(
                    f"No train data files found for the dataset {dataset_name!r}. "
                    f"Please check that the provided dataset directory {train_path} "
                    "contains arrow files of the form 'data-*.arrow'."
                )
            try:
                ds = load_dataset(
                    "arrow",
                    data_files=data_files,
                    split=dataset_config.train_name,
                    streaming=True,
                )
            except ValueError:
                ds = load_dataset(
                    "arrow", data_files=data_files, split="train", streaming=True
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
                streaming=True,
                trust_remote_code=True,
            )

        assert isinstance(ds, IterableDataset), f"Unsupported dataset type: {type(ds)}"

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

        ds = process_dataset(
            dataset=ds,
            characters_to_keep=config.characters_to_keep,
            text_column="text",
            audio_column="audio",
            lower_case=config.model.lower_case,
            cast_to_sampling_rate=config.model.sampling_rate,
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
        logger.info("Loading CoRal validation dataset")

    val = load_dataset(
        path=config.evaluation_dataset.id,
        split=config.evaluation_dataset.val_name,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        streaming=True,
        trust_remote_code=True,
    )
    if config.evaluation_dataset.text_column != "text":
        val = val.rename_column(config.evaluation_dataset.text_column, "text")
    if config.evaluation_dataset.audio_column != "audio":
        val = val.rename_column(config.evaluation_dataset.audio_column, "audio")

    val = process_dataset(
        dataset=val,
        characters_to_keep=config.characters_to_keep,
        text_column="text",
        audio_column="audio",
        lower_case=config.model.lower_case,
        cast_to_sampling_rate=config.model.sampling_rate,
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
    logger.info(f"Loading dataset {config.dataset_id}...")
    dataset = load_dataset(
        path="alexandrainst/coral",
        name=config.dataset_subset,
        split=config.eval_split_name,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        trust_remote_code=True,
        cache_dir=config.cache_dir,
    )

    assert isinstance(dataset, Dataset)
    if config.filter_dataset:
        dataset = filter_dataset(
            dataset=dataset,
            audio_column="audio",
            min_seconds_per_example=config.min_seconds_per_example,
            max_seconds_per_example=config.max_seconds_per_example,
            remove_maybe_validated=True,
        )
    if config.process_dataset:
        dataset = process_dataset(
            dataset=dataset,
            characters_to_keep=config.characters_to_keep,
            text_column="text",
            audio_column="audio",
            lower_case=True,
            cast_to_sampling_rate=config.sampling_rate,
        )
    return dataset


def filter_dataset(
    dataset: Data,
    audio_column: str,
    min_seconds_per_example: int,
    max_seconds_per_example: int,
    train_name: str | None = None,
    remove_maybe_validated: bool | None = None,
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
        train_name:
            The name of the training split. This is only relevant if `dataset` is a
            DatasetDict or IterableDatasetDict. If `None`, then we assume this is not
            needed. Defaults to `None`.
        remove_maybe_validated:
            Whether to remove samples that are validated as "maybe". This is only
            relevant if `dataset` is a Dataset or IterableDataset. If `None`, then
            we assume this is not needed. Defaults to `None`.

    Returns:
        The filtered dataset.

    Raises:
        ValueError:
            If `remove_maybe_validated` is not `None` and `dataset` is not a
            Dataset or IterableDataset.
    """
    assert (
        not isinstance(dataset, DatasetDict | IterableDatasetDict)
        or train_name is not None
    ), (
        "The `train_name` argument needs to be specified if the dataset is a "
        "DatasetDict."
    )

    assert (
        not isinstance(dataset, Dataset | IterableDataset)
        or remove_maybe_validated is not None
    ), (
        "The `remove_maybe_validated` argument needs to be specified if the dataset "
        "is a Dataset."
    )

    logger.info("Filtering the dataset...")

    if isinstance(dataset, Dataset):
        assert remove_maybe_validated is not None
        num_samples_before = len(dataset)
        filter_fn = partial(
            filter_examples,
            remove_maybe_validated=remove_maybe_validated,
            audio_column=audio_column,
            min_seconds_per_example=min_seconds_per_example,
            max_seconds_per_example=max_seconds_per_example,
        )
        dataset = dataset.filter(
            filter_fn, batched=True, num_proc=mp.cpu_count(), desc="Filtering dataset"
        )
        num_samples_removed = num_samples_before - len(dataset)
        logger.info(f"Removed {num_samples_removed:,} samples from the dataset")

    elif isinstance(dataset, IterableDataset):
        assert remove_maybe_validated is not None
        filter_fn = partial(
            filter_examples,
            remove_maybe_validated=remove_maybe_validated,
            audio_column=audio_column,
            min_seconds_per_example=min_seconds_per_example,
            max_seconds_per_example=max_seconds_per_example,
        )
        dataset = dataset.filter(filter_fn, batched=True)

    elif isinstance(dataset, DatasetDict):
        for split_name, split in dataset.items():
            num_samples_before = len(split)
            filter_fn = partial(
                filter_examples,
                remove_maybe_validated=not split_name == train_name,
                audio_column=audio_column,
                min_seconds_per_example=min_seconds_per_example,
                max_seconds_per_example=max_seconds_per_example,
            )
            dataset[split_name] = split.filter(
                filter_fn,
                batched=True,
                num_proc=mp.cpu_count(),
                desc=f"Filtering {split_name} split",
            )
            num_samples_removed = num_samples_before - len(dataset[split_name])
            logger.info(
                f"Removed {num_samples_removed:,} samples from the {split_name} split."
            )

    elif isinstance(dataset, IterableDatasetDict):
        for split_name, split in dataset.items():
            filter_fn = partial(
                filter_examples,
                remove_maybe_validated=not split_name == train_name,
                audio_column=audio_column,
                min_seconds_per_example=min_seconds_per_example,
                max_seconds_per_example=max_seconds_per_example,
            )
            dataset[split_name] = split.filter(filter_fn, batched=True)

    return dataset


def filter_examples(
    samples: dict[str, Any],
    remove_maybe_validated: bool,
    audio_column: str,
    min_seconds_per_example: int,
    max_seconds_per_example: int,
) -> list[bool]:
    """Filter samples based on the validation status.

    Args:
        samples:
            The samples to filter.
        remove_maybe_validated:
            Whether to remove samples that are validated as "maybe".
        audio_column:
            The name of the column containing the audio.
        min_seconds_per_example:
            The minimum number of seconds that an example can have.
        max_seconds_per_example:
            The maximum number of seconds that an example can

    Returns:
        A list of booleans indicating whether the samples should be kept.
    """
    idxs_too_short = [
        audio_dct["array"].shape[0]
        < audio_dct["sampling_rate"] * min_seconds_per_example
        for audio_dct in samples[audio_column]
    ]
    idxs_too_long = [
        audio_dct["array"].shape[0]
        > audio_dct["sampling_rate"] * max_seconds_per_example
        for audio_dct in samples[audio_column]
    ]
    if "validated" in samples:
        idxs_rejected = [
            validated in {"rejected", "maybe"}
            if remove_maybe_validated
            else validated == "rejected"
            for validated in samples["validated"]
        ]
    else:
        idxs_rejected = [False] * len(samples[audio_column])
    return [
        not (too_short or too_long or rejected)
        for too_short, too_long, rejected in zip(
            idxs_too_short, idxs_too_long, idxs_rejected
        )
    ]


def process_dataset(
    dataset: Data,
    characters_to_keep: str,
    text_column: str,
    audio_column: str | None,
    lower_case: bool,
    cast_to_sampling_rate: int | None = None,
) -> Data:
    """Process the dataset.

    Note that this does not remove any samples from the dataset.

    Args:
        dataset:
            The dataset to be cleaned.
        characters_to_keep:
            A string containing all the characters that should be kept in the
            transcriptions.
        text_column:
            The name of the column containing the text.
        audio_column:
            The name of the column containing the audio. Can be `None` if the dataset
            does not have an audio column.
        lower_case:
            Whether to make the text lower case.
        cast_to_sampling_rate:
            The sampling rate to cast the audio to. If `None`, then the audio is not
            cast. Defaults to `None`.

    Returns:
        The cleaned dataset.
    """
    logger.info("Processing the dataset...")

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

    func = partial(
        process_example,
        characters_to_keep=characters_to_keep,
        conversion_dict=conversion_dict,
        text_column=text_column,
        lower_case=lower_case,
    )
    if isinstance(dataset, Dataset | DatasetDict):
        mapped = dataset.map(
            function=func, num_proc=mp.cpu_count(), desc="Processing dataset"
        )
    else:
        mapped = dataset.map(function=func)

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    if isinstance(dataset, Dataset | IterableDataset) and isinstance(
        mapped, Dataset | IterableDataset
    ):
        mapped._info = dataset._info
    elif isinstance(dataset, DatasetDict | IterableDatasetDict) and isinstance(
        mapped, DatasetDict | IterableDatasetDict
    ):
        for key, value in dataset.items():
            if key in mapped:
                mapped[key]._info = value._info

    return mapped


def process_example(
    example: dict,
    characters_to_keep: str,
    conversion_dict: dict[str, str],
    text_column: str,
    lower_case: bool,
) -> dict:
    """Helper function which cleans a single example.

    Args:
        example:
            The example to be cleaned.
        characters_to_keep:
            A string containing all the characters that should be kept in the
            transcriptions.
        conversion_dict:
            A dictionary of characters to be converted.
        text_column:
            The name of the column containing the text.
        lower_case:
            Whether to make the text lower case.

    Returns:
        The cleaned example.
    """
    doc = example[text_column]

    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    doc = normalize("NFKC", doc)

    for key, value in conversion_dict.items():
        doc = doc.replace(key, value)

    if lower_case:
        doc = doc.lower()
        characters_to_keep = characters_to_keep.lower()
    else:
        characters_to_keep = characters_to_keep.lower() + characters_to_keep.upper()

    # Remove all non-standard characters, and make the document lower case
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(characters_to_keep + ' |')}]"
    )
    doc = re.sub(non_standard_characters_regex, " ", doc.lower().strip())

    # Replace superfluous spaces
    doc = re.sub(r" +", " ", doc)

    # Strip each newline
    doc = "\n".join([line.strip() for line in doc.split("\n")])

    # Re-assign the cleaned transcription
    example[text_column] = doc

    return example
