"""Evaluation of ASR models."""

import itertools as it
import logging

import pandas as pd
import torch
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

    dataset = load_dataset_for_evaluation(config=config)

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    transcriber = pipeline(
        task="automatic-speech-recognition", model=config.model_id, device=device
    )
    assert isinstance(transcriber, AutomaticSpeechRecognitionPipeline)

    # Get metrics
    _, _, all_scores = compute_metrics_of_dataset_using_pipeline(
        dataset=dataset,
        transcriber=transcriber,
        metric_names=[config.metric],
        characters_to_keep=config.characters_to_keep,
        text_column="text",
        audio_column="audio",
        batch_size=config.batch_size,
    )

    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    df["score"] = all_scores[config.metric]

    # Make a new binary feature of whether the native language is Danish
    df["accent"] = df.country_birth.map(lambda x: "native" if x == "DK" else "foreign")

    age_group_mapping = {"0-25": (0, 25), "25-50": (26, 50), "50+": (50, None)}
    df["age_group"] = df.age.map(
        lambda x: next(
            group
            for group, (start, end) in age_group_mapping.items()
            if (start is None or x >= start) and (end is None or x < end)
        )
    )

    # Get unique values for each category
    categories = ["age_group", "gender", "dialect", "accent"]
    unique_category_values = [
        df[f"{category}"].unique().tolist() + [None] for category in categories
    ]

    # Iterate over all combinations of categories
    records = list()
    for combination in it.product(*unique_category_values):
        # Apply the combination of filters
        df_filtered = df.copy()
        skip_combination = False
        for key, value in zip(categories, combination):
            if value is None:
                continue
            new_df_filtered = df_filtered.query(f"{key}_1 == @value")
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
        score_str = f"{config.metric}={df_filtered.score.mean():.2f}"
        logger.info(f"Scores for {combination_str}: {score_str}")

    score_df = pd.DataFrame.from_records(data=records)
    return score_df
