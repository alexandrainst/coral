"""Evaluation of ASR models."""

import itertools as it
import logging
import re

import numpy as np
import pandas as pd
from datasets import Dataset, Sequence, Value, load_metric
from dotenv import load_dotenv
from numpy.typing import NDArray
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments, Wav2Vec2ProcessorWithLM

from .data import load_data
from .data_models import Processor
from .model_setup import load_model_setup
from .utils import (
    DIALECT_MAP,
    convert_iterable_dataset_to_dataset,
    transformers_output_ignored,
)

load_dotenv()


logger = logging.getLogger(__package__)


def evaluate(config: DictConfig) -> pd.DataFrame:
    """Evaluate a model on a CoRal evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DataFrame with the evaluation scores.
    """
    with transformers_output_ignored():
        model_data = load_model_setup(config=config).load_saved()

    dataset = load_data(config=config)

    trainer = Trainer(
        args=TrainingArguments(".", remove_unused_columns=False, report_to=[]),
        model=model_data.model,
        data_collator=model_data.data_collator,
        tokenizer=model_data.processor.tokenizer,
    )

    split = config.evaluation_dataset.split

    logger.info(f"Converting iterable {split} dataset to a regular dataset.")
    test_dataset = convert_iterable_dataset_to_dataset(
        iterable_dataset=dataset[split], dataset_id=f"coral-{split}"
    )

    logger.info(f"Preprocessing the {split} split of the CoRal dataset.")
    test_dataset = preprocess_transcriptions(
        dataset=test_dataset, processor=model_data.processor
    )

    df = test_dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    # Make a new binary feature of whether the native language is Danish
    df["native_1"] = df.native_language_1.isin(["Danmark", "Denmark", "Dansk"])

    # Fix dialects
    df.dialect_1 = [
        re.sub(r"\(.*\)", "", dialect.lower()).strip() for dialect in df.dialect_1
    ]
    df.dialect_1 = [DIALECT_MAP.get(dialect, dialect) for dialect in df.dialect_1]

    # Get unique values for each category
    categories = ["age", "gender", "dialect", "native"]
    unique_category_values = [
        df[f"{category}_1"].unique().tolist() + [None] for category in categories
    ]

    # Get predictions
    prediction_object = trainer.predict(test_dataset=test_dataset)
    predictions = prediction_object.predictions
    labels = prediction_object.label_ids
    assert isinstance(predictions, np.ndarray)
    assert isinstance(labels, np.ndarray)

    # If all the logits are -100 for a token, then we set the logit for the padding
    # token for that token to 0. This is to ensure that this token gets decoded to a
    # padding token, and are therefore ignored
    pad_token = model_data.processor.tokenizer.pad_token_id
    predictions[np.all(predictions == -100, axis=-1), pad_token] = 0

    logger.info("Decoding the predictions to get the transcriptions.")

    # Decode the predictions to get the transcriptions. When a language model is
    # attached to the processor then we get the predicted string directly from the
    # logits. If the vocabulary dimension of the predictions is too small then we pad
    # with zeros to match the size of the vocabulary
    if isinstance(model_data.processor, Wav2Vec2ProcessorWithLM):
        vocab_size = model_data.processor.tokenizer.get_vocab()
        mismatch_dim = len(vocab_size) - predictions.shape[-1]
        predictions = np.pad(
            array=predictions,
            pad_width=((0, 0), (0, 0), (0, mismatch_dim)),
            mode="constant",
            constant_values=pad_token,
        )
        predictions_str = model_data.processor.batch_decode(predictions).text

    # Otherwise, if we are not using a language model, we need to convert the logits to
    # token IDs and then decode the token IDs to get the predicted string
    else:
        pred_ids: NDArray[np.int_] = np.argmax(predictions, axis=-1)
        predictions_str = model_data.processor.tokenizer.batch_decode(pred_ids)

    # Decode the ground truth labels
    labels[labels == -100] = pad_token
    labels_str = model_data.processor.tokenizer.batch_decode(
        sequences=labels, group_tokens=False
    )

    # Load metrics
    wer_metric = load_metric("wer", trust_remote_code=True)
    cer_metric = load_metric("cer", trust_remote_code=True)
    assert wer_metric is not None
    assert cer_metric is not None

    # Iterate over all combinations of categories
    records = list()
    for combination in it.product(*unique_category_values):
        # Apply the combination of filters
        df_filtered = df.copy()
        skip_combination = False
        for key, value in zip(categories, combination):
            if value is not None:
                new_df_filtered = df_filtered.query(f"{key}_1 == @value")
                if (
                    len(new_df_filtered) == len(df_filtered)
                    or len(new_df_filtered) == 0
                ):
                    skip_combination = True
                df_filtered = new_df_filtered

        if skip_combination:
            continue

        # Compute the scores
        idxs = df_filtered.index.tolist()
        predictions_filtered = [predictions_str[idx] for idx in idxs]
        labels_filtered = [labels_str[idx] for idx in idxs]
        wer_computed = wer_metric.compute(
            predictions=predictions_filtered, references=labels_filtered
        )
        cer_computed = cer_metric.compute(
            predictions=predictions_filtered, references=labels_filtered
        )
        combination_scores = wer_computed | cer_computed

        # Add the combination to the records
        named_combination = dict(zip(categories, combination))
        records.append(named_combination | combination_scores)

        # Log the scores
        combination_str = ", ".join(
            f"{key}={value}"
            for key, value in named_combination.items()
            if value is not None
        )
        if combination_str == "":
            combination_str = f"entire {split} set"
        scores_str = ", ".join(
            f"{key}={value:.0%}" for key, value in combination_scores.items()
        )
        logger.info(f"Scores for {combination_str}: {scores_str}")

    score_df = pd.DataFrame.from_records(data=records)
    return score_df


def preprocess_transcriptions(dataset: Dataset, processor: Processor) -> Dataset:
    """Preprocess the transcriptions in the dataset.

    Args:
        dataset:
            The dataset to preprocess.
        processor:
            The processor to use for tokenization.

    Returns:
        The preprocessed dataset.
    """

    def tokenize_examples(example: dict) -> dict:
        example["labels"] = processor(text=example["text"], truncation=True).input_ids
        example["input_length"] = len(example["labels"])
        return example

    mapped = dataset.map(tokenize_examples)
    assert isinstance(mapped, Dataset)

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    mapped._info = dataset._info
    mapped._info.features["labels"] = Sequence(feature=Value(dtype="int64"), length=-1)
    mapped._info.features["input_length"] = Value(dtype="int64")
    mapped._info = dataset._info

    return mapped
