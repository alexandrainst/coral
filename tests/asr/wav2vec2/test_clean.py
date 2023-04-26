"""Unit tests for the `asr.wav2vec2.clean` module."""

from copy import deepcopy

import pytest

from coral_models.asr.wav2vec2.clean import clean_dataset, clean_transcription


def test_clean_dataset(dataset, cfg) -> None:
    # Extract a copy of the documents in the dataset
    train_sentences = deepcopy(dataset["train"][cfg.dataset.text_column])
    val_sentences = deepcopy(dataset["val"][cfg.dataset.text_column])
    test_sentences = deepcopy(dataset["test"][cfg.dataset.text_column])

    # Manually apply the cleaning function to the documents
    train_sentences = [clean_transcription(sentence) for sentence in train_sentences]
    val_sentences = [clean_transcription(sentence) for sentence in val_sentences]
    test_sentences = [clean_transcription(sentence) for sentence in test_sentences]

    # Clean the dataset with the `clean_dataset` function
    dataset = clean_dataset(dataset)

    # Check that the cleaned dataset is equal to the manually cleaned dataset
    assert dataset["train"][cfg.dataset.text_column] == train_sentences
    assert dataset["val"][cfg.dataset.text_column] == val_sentences
    assert dataset["test"][cfg.dataset.text_column] == test_sentences


@pytest.mark.parametrize(
    "transcription, expected",
    [
        ("", ""),
        (" ", ""),
        (" laaseșmeð\u0301", "låsesmed"),
        ("§14, stk. 5", "paragraf|14|stk|5"),
        ("Han haR 5‰", "han|har|5|promille"),
        ("Han haR 5%", "han|har|5|procent"),
        (" Han lèr konstãnt\u0301 lîge her", "han|ler|konstant|lige|her"),
        (" Vi tager den lige\u200b på gefühlen ", "vi|tager|den|lige|på|gefuehlen"),
        ("10μg\u200b dosis", "10|mikrogram|dosis"),
        ("Han vejer 10kg\u200b", "han|vejer|10|kilo"),
        (
            "株式会社ＫＡＤＯＫＡＷＡ Ｆｕｔｕｒｅ Ｐｕｂｌｉｓｈｉｎｇ",
            "株式会社kadokawa|future|publishing",
        ),
        ("１２３", "123"),
        ("＋－．～）｝", "plus|minus"),
    ],
)
def test_clean_transcription(transcription: str, expected: str) -> None:
    assert clean_transcription(transcription) == expected
