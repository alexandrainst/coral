"""Functions related to the cleaning of the data for Wav2Vec 2.0 models."""

import re
from unicodedata import normalize

from datasets import DatasetDict
from omegaconf import DictConfig


def clean_dataset(cfg: DictConfig, dataset: DatasetDict) -> DatasetDict:
    """Clean the transcriptions in a dataset.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
        dataset (DatasetDict):
            The dataset to be cleaned.

    Returns:
        DatasetDict:
            The cleaned dataset.
    """

    def clean_examples(example: dict) -> dict:
        example[cfg.dataset.text_column] = clean_transcription(
            example[cfg.dataset.text_column]
        )
        return example

    return dataset.map(clean_examples)


def clean_transcription(doc: str) -> str:
    """Cleans the transcription of a document.

    Args:
        doc (str):
            A document to be cleaned.

    Returns:
        str:
            The cleaned document.
    """
    # NFKC normalize the transcriptions
    doc = normalize("NFKC", doc)

    # Remove punctuation
    regex = re.compile(r"[\[\]\{\}\(\)\,\?\.\!\;\:\"\“\'\’\”\�\•\n\r\⁄\’\~]")
    doc = re.sub(regex, "", doc)

    # Remove non-vocabulary characters
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
    }
    for key, value in conversion_dict.items():
        doc = doc.replace(key, value)

    # Remove empty whitespace
    doc = re.sub("\u0301", " ", doc)
    doc = re.sub("\u200b", " ", doc)

    # Replace spaces with a pipe, to emphasise the word boundaries
    doc = re.sub(r" +", "|", doc)

    # Make the transcription lowercase and strip whitespace
    doc = doc.lower().strip().strip("|")

    return doc
