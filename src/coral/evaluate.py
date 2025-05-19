"""Evaluation of ASR models."""

import itertools as it
import logging
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import random

from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)

from .compute_metrics import compute_metrics_of_dataset_using_pipeline
from .data import load_dataset_for_evaluation
from .utils import transformers_output_ignored, create_mappings_for_categories_in_df

load_dotenv()


logger = logging.getLogger(__package__)


def evaluate(config: DictConfig, dataset_config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a model on the CoRal evaluation dataset.

    Args:
        config:
            The Hydra configuration object.
        dataset_config:
            The Hydra configuration object specific to the dataset.

    Returns:
        A DataFrame with the evaluation scores.

    """
    assert (
        config.model_id is not None
    ), "`model_id` must be set to perform an evaluation!"

    logger.info("Loading the dataset...")
    dataset = load_dataset_for_evaluation(config=config, dataset_config=dataset_config)

    # Pick a random subset if subset_size is given
    if config.get("subset_size", None):
        subset_size = min(config.get("subset_size", None), len(dataset))
        logger.info(f"Using subset of size {subset_size}")
        dataset = dataset.select(random.sample(range(len(dataset)), subset_size))

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    transcriber = load_asr_pipeline(model_id=config.model_id, no_lm=config.no_lm)

    logger.info("Computing the scores...")
    preds, labels, all_scores = compute_metrics_of_dataset_using_pipeline(
        dataset=dataset,
        transcriber=transcriber,
        metric_names=config.metrics,
        characters_to_keep=config.characters_to_keep,
        text_column=dataset_config.text_column,
        audio_column=dataset_config.audio_column,
        batch_size=config.batch_size,
    )

    # Create a DataFrame with predictions, labels, and scores
    df_predictions = pd.DataFrame(
        {
            "predictions": preds,
            "labels": labels,
            **{metric: all_scores[metric] for metric in config.metrics},
        }
    )

    if not config.detailed:
        logger.info("Bootstrapping the scores...")
        bootstrap_scores = defaultdict(list)
        bootstrap_std_errs = defaultdict(list)
        for metric in config.metrics:
            for bidx in range(config.bootstrap_samples):
                rng = np.random.default_rng(seed=bidx)
                bootstrap_sample = rng.choice(
                    all_scores[metric], size=len(all_scores[metric]), replace=True
                )
                mean_score = np.mean(bootstrap_sample)
                std_error = np.std(bootstrap_sample) / np.sqrt(len(bootstrap_sample))
                bootstrap_scores[metric].append(mean_score)
                bootstrap_std_errs[metric].append(std_error)
        mean_scores = {
            metric: np.mean(bootstrap_scores[metric]) for metric in config.metrics
        }
        std_errs = {
            metric: np.mean(bootstrap_std_errs[metric]) for metric in config.metrics
        }
        score_string = "\n- ".join(
            [
                f"{metric.upper()}={mean_score:.1%} ± {1.96 * std_err:.1%}"
                for metric, mean_score, std_err in zip(
                    config.metrics, mean_scores.values(), std_errs.values()
                )
            ]
        )
        logger.info(
            f"Bootstrap scores of {config.model_id} on {dataset_config.id}::{dataset_config.subset}:\n"
            f"- {score_string}"
        )
        df_scores = pd.DataFrame(
            {
                "model": [config.model_id],
                "dataset": [dataset_config.id +"::"+str(dataset_config.subset)],
                **{
                    f"{metric}_mean": [mean_scores[metric]] for metric in config.metrics
                },
                **{
                    f"{metric}_std_err": [std_errs[metric]] for metric in config.metrics
                },
            }
        )
        return df_scores, df_predictions

    logger.info(
        "Converting the dataset to a dataframe computing the scores for each "
        "metadata category..."
    )

    df = dataset.to_pandas()
    eval_categories = dataset_config.eval_categories if "eval_categories" in list(dataset_config.keys()) else {}

    df = create_mappings_for_categories_in_df(df=df, eval_categories=eval_categories)

    for metric_name in config.metrics:
        df[metric_name] = all_scores[metric_name]

    df_scores = get_score_df(
        df=df,
        categories=list(eval_categories.keys()),
        metric_names=config.metrics,
    )
    return df_scores, df_predictions


def load_asr_pipeline(model_id: str, no_lm: bool) -> AutomaticSpeechRecognitionPipeline:
    """Load the ASR pipeline.

    Args:
        model_id:
            The model ID to load.
        no_lm:
            Whether to load the ASR pipeline without a language model. Only applicable
            to Wav2Vec 2.0 models.

    Returns:
        The ASR pipeline.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with transformers_output_ignored():
        if no_lm:
            model = Wav2Vec2ForCTC.from_pretrained(model_id)
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            transcriber = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
        else:
            transcriber = pipeline(
                task="automatic-speech-recognition", model=model_id, device=device
            )

    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)
    return transcriber


def get_score_df(
    df: pd.DataFrame, categories: list[str], metric_names: list[str]
) -> pd.DataFrame:
    """Get the score DataFrame for the evaluation dataset.

    Args:
        df:
            The evaluation dataframe, containing all the metadata columns and columns
            for each metric.
        categories:
            The categories to evaluate.
        metric_names:
            The names of the metrics to use for evaluation.

    Returns:
        The score DataFrame.
    """
    unique_category_values = [
        df[f"{category}"].unique().tolist() + [None] for category in categories
    ]

    records = list()
    for combination in it.product(*unique_category_values):
        # Apply the combination of filters
        df_filtered = df.copy()
        skip_combination = False
        for key, value in zip(categories, combination):
            if value is None:
                continue
            new_df_filtered = df_filtered.query(f"{key} == @value")
            if len(new_df_filtered) == len(df_filtered) or len(new_df_filtered) == 0:
                skip_combination = True
            df_filtered = new_df_filtered

        if skip_combination:
            continue

        # Add the combination to the records
        named_combination = dict(zip(categories, combination))
        score_dict = {
            metric_name: df_filtered[metric_name].mean() for metric_name in metric_names
        }
        records.append(named_combination | score_dict)

        # Log the scores
        combination_str = ", ".join(
            f"{key}={value}"
            for key, value in named_combination.items()
            if value is not None
        )
        if combination_str == "":
            combination_str = "entire dataset"
        score_str = ", ".join(
            f"{metric_name.upper()}={df_filtered[metric_name].mean():.1%}"
            for metric_name in metric_names
        )
        logger.info(f"Scores for {combination_str}: {score_str}")

    score_df = pd.DataFrame.from_records(data=records)
    return score_df
