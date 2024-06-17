"""Evaluate a speech model.

Usage:
    python evaluate_model.py <key>=<value> <key>=<value> ...
"""

import itertools as it
import logging
from functools import partial

import hydra
import numpy as np
import pandas as pd
from coral.data import load_data
from coral.model_setup import load_model_setup
from coral.protocols import Processor
from coral.utils import convert_iterable_dataset_to_dataset, transformers_output_ignored
from datasets import DatasetDict, IterableDatasetDict, Sequence, Value
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import EvalPrediction, Trainer, TrainingArguments

load_dotenv()


logger = logging.getLogger("coral")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Evaluate a speech model on a dataset.

    Args:
        cfg:
            The Hydra configuration object.
    """
    with transformers_output_ignored():
        model_data = load_model_setup(cfg).load_saved()

    dataset: DatasetDict | IterableDatasetDict = load_data(cfg)
    dataset = preprocess_transcriptions(dataset=dataset, processor=model_data.processor)

    trainer = Trainer(
        args=TrainingArguments(".", remove_unused_columns=False, report_to=[]),
        model=model_data.model,
        data_collator=model_data.data_collator,
        compute_metrics=model_data.compute_metrics,
        eval_dataset=dataset,
        tokenizer=getattr(model_data.processor, "tokenizer"),
    )

    logger.info("Converting iterable test dataset to a regular dataset.")
    test_dataset = convert_iterable_dataset_to_dataset(
        iterable_dataset=dataset["test"].take(n=10)
    )
    prediction_object = trainer.predict(test_dataset=test_dataset)
    predictions = prediction_object.predictions
    labels = prediction_object.label_ids
    assert isinstance(predictions, np.ndarray)
    assert isinstance(labels, np.ndarray)

    df = test_dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    df["native_1"] = df.native_language_1 == "Denmark"

    categories = ["age", "gender", "dialect", "native"]
    unique_category_values = [
        df[f"{category}_1"].unique().tolist() + [None] for category in categories
    ]

    compute_metrics = partial(model_data.compute_metrics, log_examples=False)

    records = list()
    for combination in it.product(*unique_category_values):
        df_filtered = df.copy()
        for key, value in zip(categories, combination):
            if value is not None:
                df_filtered = df_filtered.query(f"{key}_1 == '{value}'")
        if not len(df_filtered):
            continue
        idxs = df_filtered.index.tolist()
        combination_scores = compute_metrics(
            EvalPrediction(predictions=predictions[idxs], label_ids=labels[idxs])
        )
        named_combination = dict(zip(categories, combination))
        records.append(named_combination | combination_scores)
        logger.info(f"Scores for {named_combination}: {combination_scores}")

    score_df = pd.DataFrame.from_records(data=records)
    score_df.to_csv(f"{cfg.pipeline_id}_scores.csv", index=False)


def preprocess_transcriptions(
    dataset: DatasetDict | IterableDatasetDict, processor: Processor
) -> DatasetDict | IterableDatasetDict:
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

    # After calling `map` the DatasetInfo is lost, so we need to add it back in
    for split in dataset.keys():
        mapped[split]._info = dataset[split]._info
        mapped[split]._info.features["labels"] = Sequence(
            feature=Value(dtype="int64"), length=-1
        )
        mapped[split]._info.features["input_length"] = Value(dtype="int64")
        mapped[split]._info = dataset[split]._info

    return mapped


if __name__ == "__main__":
    main()
