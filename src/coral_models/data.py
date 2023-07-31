"""Functions related to the data loading and processing"""

import os
import re
from unicodedata import normalize

from datasets import DatasetDict, IterableDatasetDict, load_dataset
from omegaconf import DictConfig


def load_data(cfg: DictConfig) -> DatasetDict | IterableDatasetDict:
    """Load an audio dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        DatasetDict or IterableDatasetDict:
            The audio dataset.

    Raises:
        ValueError:
            If the dataset is not supported.
    """
    # Load dataset from the Hugging Face Hub. The HUGGINGFACE_HUB_TOKEN is only used
    # during CI - normally it is expected that the user is logged in to the Hugging
    # Face Hub using the `huggingface-cli login` command.
    dataset = load_dataset(
        path=cfg.dataset.id,
        name=cfg.dataset.subset,
        token=os.getenv("HUGGINGFACE_HUB_TOKEN", True),
        streaming=True,
    )

    # Only include the train, validation and test splits of the dataset, and rename
    # these splits to the default split names.
    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            dict(
                train=dataset[cfg.dataset.train_name],
                val=dataset[cfg.dataset.val_name],
                test=dataset[cfg.dataset.test_name],
            )
        )
    elif isinstance(dataset, IterableDatasetDict):
        return IterableDatasetDict(
            dict(
                train=dataset[cfg.dataset.train_name],
                val=dataset[cfg.dataset.val_name],
                test=dataset[cfg.dataset.test_name],
            )
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def clean_dataset(
    cfg: DictConfig,
    dataset: DatasetDict | IterableDatasetDict,
) -> DatasetDict | IterableDatasetDict:
    """Clean the transcriptions in a dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
        dataset (DatasetDict or IterableDatasetDict):
            The dataset to be cleaned.

    Returns:
        DatasetDict or IterableDatasetDict:
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
        f"[^{re.escape(cfg.characters_to_keep)}]"
    )

    def clean_examples(example: dict) -> dict:
        example[cfg.dataset.text_column] = clean_transcription(
            doc=example[cfg.dataset.text_column],
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
