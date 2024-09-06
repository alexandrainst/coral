"""Evaluation of ASR models."""

import itertools as it
import logging

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import AutomaticSpeechRecognitionPipeline, pipeline

from .compute_metrics import compute_metrics_of_dataset_using_pipeline
from .data import load_dataset_for_evaluation

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

    # TEMP
    df = convert_evaluation_dataset_to_df(
        dataset=dataset, sub_dialect_to_dialect_mapping=config.sub_dialect_to_dialect
    )
    breakpoint()

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    transcriber = load_asr_pipeline(model_id=config.model_id)

    logger.info("Computing the scores...")
    _, _, all_scores = compute_metrics_of_dataset_using_pipeline(
        dataset=dataset,
        transcriber=transcriber,
        metric_names=[config.metric],
        characters_to_keep=config.characters_to_keep,
        text_column="text",
        audio_column="audio",
        batch_size=config.batch_size,
    )

    logger.info(
        "Converting the dataset to a dataframe computing the scores for each "
        "metadata category..."
    )
    df = convert_evaluation_dataset_to_df(
        dataset=dataset, sub_dialect_to_dialect_mapping=config.sub_dialect_to_dialect
    )
    df["score"] = all_scores[config.metric]
    score_df = get_score_df(
        df=df,
        categories=["age_group", "gender", "dialect", "accent"],
        metric_name=config.metric,
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

    df["accent"] = df.country_birth.map(lambda x: "native" if x == "DK" else "foreign")

    age_group_mapping = {"0-25": (0, 25), "25-50": (26, 50), "50+": (50, None)}
    df["age_group"] = df.age.map(
        lambda x: next(
            group
            for group, (start, end) in age_group_mapping.items()
            if (start is None or x >= start) and (end is None or x < end)
        )
    )

    df.dialect = df.dialect.map(sub_dialect_to_dialect_mapping)
    return df


def load_asr_pipeline(model_id: str) -> AutomaticSpeechRecognitionPipeline:
    """Load the ASR pipeline.

    Args:
        model_id:
            The model ID to load.

    Returns:
        The ASR pipeline.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    transcriber = pipeline(
        task="automatic-speech-recognition", model=model_id, device=device
    )
    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)
    return transcriber


def get_score_df(
    df: pd.DataFrame, categories: list[str], metric_name: str
) -> pd.DataFrame:
    """Get the score DataFrame for the evaluation dataset.

    Args:
        df:
            The evaluation dataframe, containing all the metadata columns and a 'score'
            column.
        categories:
            The categories to evaluate.
        metric_name:
            The name of the metric to evaluate.

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
        records.append(named_combination | {"score": df_filtered.score.mean()})

        # Log the scores
        combination_str = ", ".join(
            f"{key}={value}"
            for key, value in named_combination.items()
            if value is not None
        )
        if combination_str == "":
            combination_str = "entire dataset"
        score_str = f"{metric_name}={df_filtered.score.mean():.2f}"
        logger.info(f"Scores for {combination_str}: {score_str}")

    score_df = pd.DataFrame.from_records(data=records)
    return score_df
