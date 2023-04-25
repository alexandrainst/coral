"""Functions related to the cleaning of the data for Wav2Vec 2.0 models."""

import re
from unicodedata import normalize

from datasets import DatasetDict


def clean_dataset(dataset: DatasetDict) -> DatasetDict:
    """Clean the transcriptions in a dataset.

    Args:
        dataset (DatasetDict):
            The dataset to be cleaned.

    Returns:
        DatasetDict:
            The cleaned dataset.
    """

    def clean_examples(example: dict) -> dict:
        example["sentence"] = _clean_transcription(example["sentence"])
        return example

    return dataset.map(clean_examples)


def _clean_transcription(doc: str) -> str:
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
    regex = re.compile(r"[\[\]\{\}\(\)\,\?\.\!\-\—\–\;\:\"\“\'\’\%\”\�\•\n\r\⁄\’]")
    doc = re.sub(regex, "", doc)

    # Remove non-vocabulary characters
    conversion_dict = {
        "aa": "å",
        "ğ": "g",
        "ñ": "n",
        "ń": "n",
        "è": "e",
        "μ": "mikro",
        "§": " paragraf ",
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
        "(?<![0-9])(18|19|20)([0-9]{2})(?![0-9])": "\1 \2",
        "1000": " tusind ",
        "[2-9]000": " \1 tusind",
        "100": " hundrede ",
        "[2-9]00": " \1 hundrede",
        "(?<![0-9])([0-9])([0-9])(?![0-9])": "\2 og \1\0",
        "10": " ti ",
        "20": " tyve ",
        "30": " tredive ",
        "40": " fyrre ",
        "50": " halvtreds ",
        "60": " treds ",
        "70": " halvfjerds ",
        "80": " firs ",
        "90": " halvfems ",
        "0": " nul ",
        "1": " et ",
        "2": " to ",
        "3": " tre ",
        "4": " fire ",
        "5": " fem ",
        "6": " seks ",
        "7": " syv ",
        "8": " otte ",
        "9": " ni ",
    }
    for key, value in conversion_dict.items():
        doc = re.sub(key, value, doc)

    # Remove empty whitespace
    doc = re.sub("\u0301", " ", doc)
    doc = re.sub("\u200b", " ", doc)

    # Replace spaces with a pipe, to emphasise the word boundaries
    doc = re.sub(r" +", "|", doc)

    # Make the transcription lowercase and strip whitespace
    doc = doc.lower().strip().strip("|")

    return doc
