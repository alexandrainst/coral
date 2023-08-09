"""Functions related to the data loading and processing"""

import logging
import os
import re
from unicodedata import normalize

from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    IterableDatasetDict,
    interleave_datasets,
    load_dataset,
)
from omegaconf import DictConfig

logger = logging.getLogger(__package__)


def load_data(cfg: DictConfig) -> DatasetDict | IterableDatasetDict:
    """Load an audio dataset.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    all_datasets: list[DatasetDict | IterableDatasetDict] = list()
    for dataset_name, dataset_cfg in cfg.datasets.items():
        logger.info(f"Loading dataset {dataset_name!r}")

        # Load dataset from the Hugging Face Hub. The HUGGINGFACE_HUB_TOKEN is only used
        # during CI - normally it is expected that the user is logged in to the Hugging
        # Face Hub using the `huggingface-cli login` command.
        dataset = load_dataset(
            path=dataset_cfg.id,
            name=dataset_cfg.subset,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
            streaming=True,
            keep_in_memory=False,
        )

        assert isinstance(dataset, DatasetDict) or isinstance(
            dataset, IterableDatasetDict
        ), f"Unsupported dataset type: {type(dataset)}"

        train = dataset[dataset_cfg.train_name]
        if dataset_cfg.val_name is not None:
            val = dataset[dataset_cfg.val_name]
        else:
            val = None
        if dataset_cfg.test_name is not None:
            test = dataset[dataset_cfg.test_name]
        else:
            test = None

        splits_dict = dict(train=train)
        if val is not None:
            splits_dict["val"] = val
        if test is not None:
            splits_dict["test"] = test

        if isinstance(dataset, DatasetDict):
            dataset = DatasetDict(splits_dict)
        elif isinstance(dataset, IterableDatasetDict):
            dataset = IterableDatasetDict(splits_dict)
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

        if dataset_cfg.text_column != "text":
            dataset = dataset.rename_column(dataset_cfg.text_column, "text")

        dataset = dataset.cast_column(
            column="audio", feature=Audio(sampling_rate=cfg.model.sampling_rate)
        )
        dataset = dataset.remove_columns(
            [
                column
                for column in dataset["train"].column_names
                if column not in ["audio", "text"]
            ]
        )
        dataset = dataset.shuffle(seed=cfg.seed)

        if cfg.model.clean_dataset:
            dataset = clean_dataset(cfg, dataset=dataset)

        all_datasets.append(dataset)

    logger.info("Interleaving datasets")

    probabilities: dict[str, list[float]] = dict()
    for split_name, split_probs in cfg.dataset_probabilities.items():
        if split_probs is None:
            split_probs = [1 / len(all_datasets)] * len(all_datasets)
            split_probs[-1] = 1 - sum(split_probs[:-1])
        elif sum(split_probs) != 1:
            raise ValueError(
                f"Dataset probabilities must sum to 1, but sum to {sum(split_probs)} "
                f"for split {split_name!r}"
            )
        probabilities[split_name] = split_probs

    train = interleave_datasets(
        datasets=[dataset["train"] for dataset in all_datasets],
        probabilities=probabilities["train"],
        seed=cfg.seed,
        stopping_strategy="all_exhausted",
    )

    # Interleave the validation sets, where we tweak the sampling probabilities in case
    # any of the datasets do not have a validation split
    has_vals = ["val" in dataset for dataset in all_datasets]
    val_probabilities = [
        prob for has_val, prob in zip(has_vals, probabilities["val"]) if has_val
    ]
    val_probabilities = [prob / sum(val_probabilities) for prob in val_probabilities]
    val_probabilities[-1] = 1 - sum(val_probabilities[:-1])
    val = interleave_datasets(
        datasets=[
            dataset["val"]
            for has_val, dataset in zip(has_vals, all_datasets)
            if has_val
        ],
        probabilities=val_probabilities,
        seed=cfg.seed,
        stopping_strategy="first_exhausted",
    )

    # Interleave the test sets, where we tweak the sampling probabilities in case any
    # of the datasets do not have a test split
    has_tests = ["test" in dataset for dataset in all_datasets]
    test_probabilities = [
        prob for has_test, prob in zip(has_tests, probabilities["test"]) if has_test
    ]
    test_probabilities = [prob / sum(test_probabilities) for prob in test_probabilities]
    test_probabilities[-1] = 1 - sum(test_probabilities[:-1])
    test = interleave_datasets(
        datasets=[
            dataset["test"]
            for has_test, dataset in zip(has_tests, all_datasets)
            if has_test
        ],
        probabilities=test_probabilities,
        seed=cfg.seed,
        stopping_strategy="first_exhausted",
    )

    if isinstance(train, Dataset):
        return DatasetDict(dict(train=train, val=val, test=test))
    else:
        return IterableDatasetDict(dict(train=train, val=val, test=test))


def clean_dataset(
    cfg: DictConfig, dataset: DatasetDict | IterableDatasetDict
) -> DatasetDict | IterableDatasetDict:
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
        "ü": "ue",
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
        f"[^{re.escape(cfg.model.characters_to_keep)}]"
    )

    def clean_examples(example: dict) -> dict:
        example["text"] = clean_transcription(
            doc=example["text"],
            non_standard_characters_regex=non_standard_characters_regex,
            conversion_dict=conversion_dict,
        )
        return example

    mapped = dataset.map(clean_examples)

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    for split in dataset.keys():
        mapped[split]._info = dataset[split]._info

    return mapped


def clean_transcription(
    doc: str,
    non_standard_characters_regex: re.Pattern[str],
    conversion_dict: dict[str, str],
) -> str:
    """Cleans the transcription of a document.

    Args:
        doc (str):
            A document to be cleaned.
        non_standard_characters_regex (compiled regex expression):
            A compiled regex expression that matches all non-standard characters.
        conversion_dict (dict[str, str]):
            A dictionary of characters to be converted.

    Returns:
        str:
            The cleaned document.
    """
    # Normalise the transcription, which uniformises the characters. For instance, the
    # "long dash" (－) is converted to the normal dash (-).
    doc = normalize("NFKC", doc)

    for key, value in conversion_dict.items():
        doc = doc.replace(key, value)

    # Replace spaces with a pipe, to emphasise the word boundaries
    doc = re.sub(r" +", "|", doc)

    return re.sub(non_standard_characters_regex, "", doc.lower().strip().strip("|"))
