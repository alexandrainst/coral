"""Functions related to the data loading and processing."""

import logging
import os
import re
from functools import partial
from pathlib import Path
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


def load_data(cfg: DictConfig) -> DatasetDict | IterableDatasetDict:
    """Load an audio dataset for training.

    Args:
        cfg:
            The Hydra configuration object.

    Returns:
        The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    all_datasets: list[DatasetDict | IterableDatasetDict] = list()
    for dataset_name, dataset_cfg in cfg.datasets.items():
        if is_main_process:
            logger.info(f"Loading dataset {dataset_name!r}")

        # Load from disk if the dataset ID is a path
        if Path(dataset_cfg.id).exists():
            train_path = Path(dataset_cfg.id) / dataset_cfg.train_name
            data_files = list(map(str, train_path.glob("data-*.arrow")))
            if len(data_files) == 0:
                raise FileNotFoundError(
                    f"No train data files found for the dataset {dataset_name!r}. "
                    f"Please check that the provided dataset directory {train_path} "
                    "contains arrow files of the form 'data-*.arrow'."
                )
            dataset = load_dataset("arrow", data_files=data_files, streaming=True)

        # Load dataset from the Hugging Face Hub. The HUGGINGFACE_HUB_TOKEN is only
        # used during CI - normally it is expected that the user is logged in to the
        # Hugging Face Hub using the `huggingface-cli login` command.
        else:
            dataset = load_dataset(
                path=dataset_cfg.id,
                name=dataset_cfg.subset,
                split=dataset_cfg.train_name,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
                streaming=True,
            )

        assert isinstance(dataset, Dataset) or isinstance(
            dataset, IterableDataset
        ), f"Unsupported dataset type: {type(dataset)}"

        if dataset_cfg.text_column != "text":
            dataset = dataset.rename_column(dataset_cfg.text_column, "text")

        if dataset_cfg.audio_column != "audio":
            dataset = dataset.rename_column(dataset_cfg.audio_column, "audio")

        dataset = dataset.cast_column(
            column="audio", feature=Audio(sampling_rate=cfg.model.sampling_rate)
        )
        dataset = dataset.remove_columns(
            [
                column
                for column in dataset.column_names
                if column not in ["audio", "text"]
            ]
        )
        dataset = dataset.shuffle(seed=cfg.seed)

        if cfg.model.clean_dataset:
            dataset = clean_dataset(cfg, dataset=dataset)

        all_datasets.append(dataset)

    assert len(all_datasets) > 0, "No datasets were loaded"

    if len(all_datasets) > 1:
        if is_main_process:
            logger.info("Interleaving datasets")
            if cfg.dataset_probabilities is None and len(all_datasets) > 1:
                logger.warning(
                    "No dataset probabilities were specified for the training split. "
                    "This means that each dataset will be sampled with equal "
                    "probability, which means that the smaller datasets will be "
                    "sampled more often than the larger datasets. This is probably "
                    "not what you want."
                )

        probabilities = cfg.dataset_probabilities
        if probabilities is None:
            probabilities = [1 / len(all_datasets)] * len(all_datasets)
            probabilities[-1] = 1 - sum(probabilities[:-1])
        elif sum(probabilities) != 1:
            raise ValueError(
                f"Dataset probabilities must sum to 1, but sum to {sum(probabilities)}"
            )

        train = interleave_datasets(
            datasets=[dataset["train"] for dataset in all_datasets],
            probabilities=probabilities,
            seed=cfg.seed,
            split=NamedSplit("train"),
            stopping_strategy="all_exhausted",
        )

        data_dict = dict(train=train)
        if isinstance(train, Dataset):
            dataset = DatasetDict(data_dict)
        else:
            dataset = IterableDatasetDict(data_dict)

    else:
        dataset = all_datasets[0]

    # Load CoRal validation and test sets
    split_names = dict(
        val=cfg.evaluation_dataset.val_name, test=cfg.evaluation_dataset.test_name
    )
    for new_split_name, old_split_name in split_names.items():
        breakpoint()
        split = load_dataset(
            path=cfg.evaluation_dataset.id,
            split=old_split_name,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        )
        if cfg.evaluation_dataset.text_column != "text":
            split = split.rename_column(cfg.evaluation_dataset.text_column, "text")

        if cfg.evaluation_dataset.audio_column != "audio":
            split = split.rename_column(cfg.evaluation_dataset.audio_column, "audio")

        split = split.cast_column(
            column="audio", feature=Audio(sampling_rate=cfg.model.sampling_rate)
        )
        split = split.remove_columns(
            [column for column in split.column_names if column not in ["audio", "text"]]
        )
        if cfg.model.clean_dataset:
            split = clean_dataset(cfg=cfg, dataset=split)
        dataset[new_split_name] = split

    return dataset


def clean_dataset(
    cfg: DictConfig, dataset: Dataset | IterableDataset
) -> Dataset | IterableDataset:
    """Clean the transcriptions in a dataset.

    Args:
        cfg:
            The Hydra configuration object
        dataset:
            The dataset to be cleaned.

    Returns:
        The cleaned dataset.
    """
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

    # This contains all the punctuation characters that will be removed from the
    # transcriptions, as they do not have an influence on the pronunciation of the
    # words.
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(cfg.characters_to_keep + ' |')}]"
    )

    mapped = dataset.map(
        partial(
            clean_example,
            non_standard_characters_regex=non_standard_characters_regex,
            conversion_dict=conversion_dict,
        )
    )

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    mapped._info = dataset._info

    return mapped


def clean_example(
    example: dict,
    non_standard_characters_regex: re.Pattern[str],
    conversion_dict: dict[str, str],
) -> dict:
    """Helper function which cleans a single example.

    Args:
        example:
            The example to be cleaned.
        non_standard_characters_regex:
            A compiled regex expression that matches all non-standard characters.
        conversion_dict:
            A dictionary of characters to be converted.

    Returns:
        The cleaned example.
    """
    doc = example["text"]

    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    doc = normalize("NFKC", doc)

    for key, value in conversion_dict.items():
        doc = doc.replace(key, value)

    # Replace superfluous spaces
    doc = re.sub(r" +", " ", doc)

    # Remove all non-standard characters, and make the document lower case
    doc = re.sub(non_standard_characters_regex, "", doc.lower().strip())

    # Re-assign the cleaned transcription
    example["text"] = doc

    return example
