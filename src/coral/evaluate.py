"""Evaluation of ASR models."""

import itertools as it
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
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
from .utils import transformers_output_ignored

load_dotenv()


logger = logging.getLogger(__package__)


def evaluate(config: DictConfig) -> pd.DataFrame:
    """Evaluate a model on the CoRal evaluation dataset.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        A DataFrame with the evaluation scores.
    """
    assert (
        config.model_id is not None
    ), "`model_id` must be set to perform an evaluation!"

    logger.info("Loading the dataset...")
    dataset = load_dataset_for_evaluation(config=config)

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    transcriber = load_asr_pipeline(model_id=config.model_id, no_lm=config.no_lm)

    logger.info("Computing the scores...")
    _, _, all_scores = compute_metrics_of_dataset_using_pipeline(
        dataset=dataset,
        transcriber=transcriber,
        metric_names=config.metrics,
        characters_to_keep=config.characters_to_keep,
        text_column=config.text_column,
        audio_column=config.audio_column,
        batch_size=config.batch_size,
    )

    logger.info("Bootstrapping the scores...")
    if not config.detailed or "coral" not in config.dataset:
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
            f"Bootstrap scores of {config.model_id} on {config.dataset}:\n"
            f"- {score_string}"
        )
        df = pd.DataFrame(
            {
                "model": [config.model_id],
                "dataset": [config.dataset],
                **{
                    f"{metric}_mean": [mean_scores[metric]] for metric in config.metrics
                },
                **{
                    f"{metric}_std_err": [std_errs[metric]] for metric in config.metrics
                },
            }
        )
        return df

    logger.info(
        "Converting the dataset to a dataframe computing the scores for each "
        "metadata category..."
    )
    df = convert_evaluation_dataset_to_df(
        dataset=dataset, sub_dialect_to_dialect_mapping=config.sub_dialect_to_dialect
    )
    for metric_name in config.metrics:
        df[metric_name] = all_scores[metric_name]
    score_df = get_score_df(
        df=df,
        categories=["age_group", "gender", "dialect"],
        metric_names=config.metrics,
    )
    return score_df


def convert_evaluation_dataset_to_df(
    dataset: Dataset, sub_dialect_to_dialect_mapping: dict[str, str]
) -> pd.DataFrame:
    """Convert the evaluation dataset to a DataFrame.

    Args:
        dataset:
            The evaluation dataset.
        sub_dialect_to_dialect_mapping:
            The mapping from sub-dialect to dialect.

    Returns:
        A DataFrame with the evaluation dataset.
    """
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    age_group_mapping = {"0-25": (0, 25), "25-50": (26, 50), "50+": (50, None)}
    df["age_group"] = df.age.map(
        lambda x: next(
            group
            for group, (start, end) in age_group_mapping.items()
            if (start is None or x >= start) and (end is None or x < end)
        )
    )

    df.dialect = df.dialect.map(sub_dialect_to_dialect_mapping)

    # For non-native speakers, we use the accent as the dialect
    df.country_birth = df.country_birth.map(lambda x: "DK" if x is None else x)
    df.loc[df.country_birth != "DK", "dialect"] = "Non-native"

    return df


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
