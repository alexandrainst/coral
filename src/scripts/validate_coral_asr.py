"""Automatic validation of an ASR dataset.

Usage:
    python src/scripts/validate_coral_asr.py \
        --model-id <model-id-to-use-for-validation> \
        --dataset-id <dataset-id-to-validate> \
        --output-dataset-id <output-dataset-id> \
        [--dataset-subset <subset-of-dataset>] \
        [--dataset-split <split-of-dataset>] \
        [--text-column <name-of-text-column-in-dataset>] \
        [--audio-column <name-of-audio-column-in-dataset>] \
        [--output-dataset-subset <output-dataset-subset>] \
        [--cache-dir <cache-directory>] \
        [--batch-size <batch-size>]
"""

import logging
import re
from functools import partial

import click
import evaluate
import numpy as np
import torch
from coral.data import clean_example
from coral.data_collators import DataCollatorCTCWithPadding
from coral.data_models import Processor
from datasets import Audio, Dataset, DatasetDict, load_dataset
from numpy.typing import NDArray
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Wav2Vec2ProcessorWithLM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_coral_asr")


@click.command()
@click.option(
    "--model-id",
    "-m",
    required=True,
    help="Hugging Face Hub ID of the ASR model to use for validation.",
)
@click.option(
    "--dataset-id",
    "-d",
    required=True,
    help="Hugging Face Hub ID of the ASR dataset to validate.",
)
@click.option(
    "--dataset-subset",
    default=None,
    show_default=True,
    help="Name of the subset of the dataset to validate. If None then we assume no "
    "subsets exist.",
)
@click.option(
    "--dataset-split",
    default=None,
    show_default=True,
    help="Name of the split of the dataset to validate. If None then all splits are "
    "validated.",
)
@click.option(
    "--text-column",
    "-t",
    default="text",
    show_default=True,
    help="Name of the column in the dataset containing the reference text.",
)
@click.option(
    "--audio-column",
    "-a",
    default="audio",
    show_default=True,
    help="Name of the column in the dataset containing the audio.",
)
@click.option(
    "--output-dataset-id",
    required=True,
    help="Hugging Face Hub ID of the dataset to save the validation results. If it "
    "already exists then it will be overwritten!",
)
@click.option(
    "--output-dataset-subset",
    default="default",
    show_default=True,
    help="Name of the subset of the dataset to save the validation results. If None "
    "then we assume no subsets exist.",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Directory to cache the dataset and the model in. Defaults to the Hugging Face "
    "default cache directory.",
)
@click.option(
    "--batch-size",
    default=1,
    show_default=True,
    help="Batch size to use for validation.",
)
def main(
    model_id: str,
    dataset_id: str,
    dataset_subset: str | None,
    dataset_split: str | None,
    text_column: str,
    audio_column: str,
    output_dataset_id: str,
    output_dataset_subset: str,
    cache_dir: str,
    batch_size: int,
):
    """Validate the samples of the CoRal ASR dataset using an ASR model."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Loading the {model_id!r} ASR model...")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCTC.from_pretrained(model_id, cache_dir=cache_dir).to(device)
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, max_seconds_per_example=10, padding="longest"
    )

    logger.info(f"Loading the {dataset_id!r} dataset...")
    dataset = load_dataset(
        path=dataset_id,
        name=dataset_subset,
        split=dataset_split,
        token=True,
        cache_dir=cache_dir,
    )
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({dataset_split: dataset})
    assert isinstance(dataset, DatasetDict)

    logger.info("Cleaning the dataset...")
    characters_to_keep = "".join(
        [
            tok
            for tok in processor.tokenizer.get_vocab().keys()
            if tok not in processor.tokenizer.all_special_tokens
        ]
    )
    dataset = clean_dataset(
        dataset=dataset, characters_to_keep=characters_to_keep, text_column=text_column
    )

    logger.info("Resampling audio to 16kHz...")
    dataset = dataset.cast_column(
        column=audio_column, feature=Audio(sampling_rate=16_000)
    )

    logger.info("Tokenising the dataset...")
    dataset = dataset.map(
        function=lambda example: dict(
            labels=processor(text=example[text_column], truncation=True).input_ids
        )
    )
    assert isinstance(dataset, DatasetDict)

    logger.info("Validating the dataset...")
    trainer = Trainer(
        args=TrainingArguments(
            output_dir=".",
            remove_unused_columns=False,
            report_to=[],
            per_device_eval_batch_size=batch_size,
        ),
        model=model,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    new_data_dict: dict[str, Dataset] = dict()
    for split_name, split in dataset.items():
        wers = get_wers(dataset=split, trainer=trainer, processor=processor)
        split = split.add_column(
            name="asr_wer", column=wers, new_fingerprint=split._fingerprint
        )
        split = split.add_column(
            name="asr_validation_model",
            column=[model_id] * len(split),
            new_fingerprint=split._fingerprint,
        )
        new_data_dict[split_name] = split

    logger.info(f"Uploading the validated dataset to {output_dataset_id!r}...")
    new_dataset = DatasetDict(new_data_dict)
    new_dataset.push_to_hub(
        repo_id=output_dataset_id,
        config_name=output_dataset_subset,
        commit_message="Add ASR validation",
        private=True,
    )


def clean_dataset(
    dataset: DatasetDict, characters_to_keep: str, text_column: str
) -> DatasetDict:
    """Clean a dataset for ASR.

    Args:
        dataset:
            The dataset to clean.
        characters_to_keep:
            The characters to keep in the transcriptions.
        text_column:
            The name of the column containing the transcriptions.

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
        f"[^{re.escape(characters_to_keep + ' |')}]"
    )

    cleaned_dataset = dataset.map(
        partial(
            clean_example,
            non_standard_characters_regex=non_standard_characters_regex,
            conversion_dict=conversion_dict,
            text_column=text_column,
        )
    )

    return cleaned_dataset


def get_wers(dataset: Dataset, trainer: Trainer, processor: Processor) -> list[float]:
    """Get the word error rates for each sample in the dataset.

    Args:
        dataset:
            The dataset to validate.
        trainer:
            The trainer to use for validation, which contains the model and data
            collator.
        processor:
            The processor used for processing the data.

    Returns:
        The word error rates for each sample in the dataset.
    """
    prediction_object = trainer.predict(test_dataset=dataset)
    predictions = prediction_object.predictions
    labels = prediction_object.label_ids
    assert isinstance(predictions, np.ndarray)
    assert isinstance(labels, np.ndarray)

    # If all the logits are -100 for a token, then we set the logit for the padding
    # token for that token to 0. This is to ensure that this token gets decoded to a
    # padding token, and are therefore ignored
    pad_token = processor.tokenizer.pad_token_id
    predictions[np.all(predictions == -100, axis=-1), pad_token] = 0

    # Decode the predictions to get the transcriptions. When a language model is
    # attached to the processor then we get the predicted string directly from the
    # logits. If the vocabulary dimension of the predictions is too small then we pad
    # with zeros to match the size of the vocabulary
    if isinstance(processor, Wav2Vec2ProcessorWithLM):
        vocab_size = processor.tokenizer.get_vocab()
        mismatch_dim = len(vocab_size) - predictions.shape[-1]
        predictions = np.pad(
            array=predictions,
            pad_width=((0, 0), (0, 0), (0, mismatch_dim)),
            mode="constant",
            constant_values=pad_token,
        )
        predictions_str = processor.batch_decode(predictions).text

    # Otherwise, if we are not using a language model, we need to convert the logits to
    # token IDs and then decode the token IDs to get the predicted string
    else:
        pred_ids: NDArray[np.int_] = np.argmax(predictions, axis=-1)
        predictions_str = processor.batch_decode(pred_ids)

    # Decode the ground truth labels
    labels[labels == -100] = pad_token
    labels_str = processor.tokenizer.batch_decode(sequences=labels, group_tokens=False)

    # Compute the word error rates
    # wer_metric = load_metric("wer", trust_remote_code=True)
    wer_metric = evaluate.load("wer")
    wers = [
        wer_metric.compute(predictions=[pred], references=[ref])
        for pred, ref in zip(predictions_str, labels_str)
    ]

    return wers


def preprocess_logits_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Workaround to avoid storing too many tensors that are not needed.

    Args:
        logits:
            The logits from the model.
        labels:
            The labels for the logits.

    Returns:
        The logits and labels to use for computing the metrics.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


if __name__ == "__main__":
    main()
