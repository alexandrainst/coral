"""Unit tests for the `data` module."""

from collections.abc import Generator

import pytest
from coral.data import load_data_for_finetuning, process_dataset, process_example
from datasets import IterableDatasetDict


class TestLoadDataForFinetuning:
    """Unit tests for the `load_data` function."""

    @pytest.fixture(scope="class")
    def dataset(self, finetuning_config) -> Generator[IterableDatasetDict, None, None]:
        """Load the dataset for testing."""
        yield load_data_for_finetuning(config=finetuning_config)

    def test_dataset_type(self, dataset) -> None:
        """Test that the dataset is of the correct type."""
        assert isinstance(dataset, IterableDatasetDict)

    def test_split_names(self, dataset) -> None:
        """Test that the dataset has the correct split names."""
        assert set(dataset.keys()) == {"train", "val"}

    def test_train_samples(self, dataset, finetuning_config) -> None:
        """Test that the training dataset has the correct samples."""
        samples = [sample["text"] for sample in dataset["train"]]
        expected_samples = [
            "hver rose på træet i haven havde sin historie",
            "min fortræffelige lille nattergal",
            "her er kommet gode klæder at slide for de fire børn",
            "jeg venter grumme meget af den",
            "men hendes vilje var fast som hendes tillid til vorherre",
        ]
        if finetuning_config.model.lower_case:
            expected_samples = [sample.lower() for sample in expected_samples]
        assert samples == expected_samples


class TestProcessDataset:
    """Unit tests for the `process_dataset` function."""

    @pytest.fixture(scope="class")
    def dataset(self, finetuning_config) -> Generator[IterableDatasetDict, None, None]:
        """Load the dataset for testing."""
        yield load_data_for_finetuning(config=finetuning_config)

    def test_process_dataset(self, dataset):
        """Test that the `process_dataset` function works as expected."""
        processed_dataset = process_dataset(
            dataset=dataset,
            characters_to_keep=None,
            text_column="text",
            audio_column=None,
            lower_case=True,
        )
        expected_samples = [
            "hver rose på træet i haven havde sin historie",
            "min fortræffelige lille nattergal",
            "her er kommet gode klæder at slide for de fire børn",
            "jeg venter grumme meget af den",
            "men hendes vilje var fast som hendes tillid til vorherre",
        ]
        for sample, expected in zip(processed_dataset["train"], expected_samples):
            assert sample["text"] == expected


class TestProcessExample:
    """Unit tests for the `process_example` function."""

    transcription = "\nThis is a (test) [sentence]\u0301 with \n{aa} and ğ. "

    empty_conversion_dict: dict[str, str] = {}
    diacritics_conversion_dict = {"aa": "å", "ğ": "g"}
    empty_whitespace_conversion_dict = {"\u0301": " "}

    all_characters = (
        set(transcription)
        | set(empty_conversion_dict.values())
        | set(diacritics_conversion_dict.values())
        | set(empty_whitespace_conversion_dict.values())
    )
    no_parentheses = all_characters - set("()[]{}")
    no_newlines = all_characters - set("\n\r")

    @pytest.mark.parametrize(
        argnames=[
            "transcription",
            "characters_to_keep",
            "conversion_dict",
            "text_column",
            "lower_case",
            "expected",
        ],
        argvalues=[
            (
                transcription,
                all_characters,
                empty_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                all_characters,
                empty_conversion_dict,
                "text",
                False,
                "This is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                all_characters,
                empty_conversion_dict,
                "text2",
                True,
                "this is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                None,
                empty_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with\n{aa} and ğ.",
            ),
            (
                transcription,
                all_characters,
                diacritics_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with\n{å} and g.",
            ),
            (
                transcription,
                all_characters,
                empty_whitespace_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence] with\n{aa} and ğ.",
            ),
            (
                transcription,
                no_parentheses,
                empty_conversion_dict,
                "text",
                True,
                "this is a test sentence \u0301 with\naa and ğ.",
            ),
            (
                transcription,
                no_parentheses,
                diacritics_conversion_dict,
                "text",
                True,
                "this is a test sentence \u0301 with\nå and g.",
            ),
            (
                transcription,
                no_parentheses,
                empty_whitespace_conversion_dict,
                "text",
                True,
                "this is a test sentence with\naa and ğ.",
            ),
            (
                transcription,
                no_newlines,
                empty_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with {aa} and ğ.",
            ),
            (
                transcription,
                no_newlines,
                diacritics_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence]\u0301 with {å} and g.",
            ),
            (
                transcription,
                no_newlines,
                empty_whitespace_conversion_dict,
                "text",
                True,
                "this is a (test) [sentence] with {aa} and ğ.",
            ),
        ],
        ids=[
            "empty-empty",
            "empty-empty-no-lower-case",
            "empty-empty-different-text-column",
            "empty-empty-with-None-characters-to-keep",
            "empty-diacritics",
            "empty-empty_whitespace",
            "parans-empty",
            "parans-diacritics",
            "parans-empty_whitespace",
            "newline-empty",
            "newline-diacritics",
            "newline-empty_whitespace",
        ],
    )
    def test_clean_example(
        self,
        transcription,
        characters_to_keep,
        conversion_dict,
        text_column,
        lower_case,
        expected,
    ) -> None:
        """Test that the `clean_example` function works as expected."""
        example = {text_column: transcription}
        cleaned_transcription = process_example(
            example=example,
            characters_to_keep=characters_to_keep,
            conversion_dict=conversion_dict,
            text_column=text_column,
            lower_case=lower_case,
        )[text_column]
        assert cleaned_transcription == expected
