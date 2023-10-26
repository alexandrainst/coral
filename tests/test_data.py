"""Unit tests for the `data` module."""

import re

import pytest
from datasets import DatasetDict, IterableDatasetDict

from coral_models.data import clean_example


class TestLoadData:
    def test_dataset_type(self, dataset) -> None:
        assert isinstance(dataset, DatasetDict) or isinstance(
            dataset, IterableDatasetDict
        )

    def test_split_names(self, dataset) -> None:
        assert set(dataset.keys()) == {"train", "val", "test"}

    def test_train_samples(self, dataset, cfg) -> None:
        if cfg.model.clean_dataset:
            samples = [sample["text"] for sample in dataset["train"]]
            assert samples == [
                "hver rose på træet i haven havde sin historie",
                "min fortræffelige lille nattergal",
                "her er kommet gode klæder at slide for de fire børn",
                "jeg venter grumme meget af den",
                "men hendes vilje var fast som hendes tillid til vorherre",
            ]


class TestCleanExample:
    transcription = "\nThis is a (test) [sentence]\u0301 with \n{aa} and ğ. "

    empty_regex = re.compile(r"")
    parens_regex = re.compile(r"[\(\)\[\]\{\}]")
    newline_regex = re.compile(r"[\n\r]")

    empty_conversion_dict: dict[str, str] = {}
    diacritics_conversion_dict = {"aa": "å", "ğ": "g"}
    empty_whitespace_conversion_dict = {"\u0301": " "}

    @pytest.mark.parametrize(
        "transcription, non_standard_characters_regex, conversion_dict, expected",
        ids=[
            "empty-empty",
            "empty-diacritics",
            "empty-empty_whitespace",
            "parans-empty",
            "parans-diacritics",
            "parans-empty_whitespace",
            "newline-empty",
            "newline-diacritics",
            "newline-empty_whitespace",
        ],
        argvalues=[
            (
                transcription,
                empty_regex,
                empty_conversion_dict,
                "this is a (test) [sentence]\u0301 with \n{aa} and ğ.",
            ),
            (
                transcription,
                empty_regex,
                diacritics_conversion_dict,
                "this is a (test) [sentence]\u0301 with \n{å} and g.",
            ),
            (
                transcription,
                empty_regex,
                empty_whitespace_conversion_dict,
                "this is a (test) [sentence] with \n{aa} and ğ.",
            ),
            (
                transcription,
                parens_regex,
                empty_conversion_dict,
                "this is a test sentence\u0301 with \naa and ğ.",
            ),
            (
                transcription,
                parens_regex,
                diacritics_conversion_dict,
                "this is a test sentence\u0301 with \nå and g.",
            ),
            (
                transcription,
                parens_regex,
                empty_whitespace_conversion_dict,
                "this is a test sentence with \naa and ğ.",
            ),
            (
                transcription,
                newline_regex,
                empty_conversion_dict,
                "this is a (test) [sentence]\u0301 with {aa} and ğ.",
            ),
            (
                transcription,
                newline_regex,
                diacritics_conversion_dict,
                "this is a (test) [sentence]\u0301 with {å} and g.",
            ),
            (
                transcription,
                newline_regex,
                empty_whitespace_conversion_dict,
                "this is a (test) [sentence] with {aa} and ğ.",
            ),
        ],
    )
    def test_clean_example(
        self,
        transcription: str,
        non_standard_characters_regex: re.Pattern[str],
        conversion_dict: dict[str, str],
        expected: str,
    ) -> None:
        example = dict(text=transcription)
        cleaned_transcription = clean_example(
            example=example,
            non_standard_characters_regex=non_standard_characters_regex,
            conversion_dict=conversion_dict,
        )["text"]
        assert cleaned_transcription == expected
