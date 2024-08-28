"""Evaluation of ASR models."""

import itertools as it
import logging
import re

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from evaluate import load as load_metric
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutomaticSpeechRecognitionPipeline, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from .data import load_dataset_for_evaluation

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
    _, _, all_scores = compute_metrics(
        dataset=dataset,
        transcriber=transcriber,
        metric_names=[config.metric],
        characters_to_keep=config.characters_to_keep,
        text_column=config.text_column,
        audio_column=config.audio_column,
        batch_size=config.batch_size,
    )

    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)

    df["score"] = all_scores[config.metric]

    # Make a new binary feature of whether the native language is Danish
    df["accent"] = "native" if df.country_birth == "DK" else "foreign"

    # Get unique values for each category
    categories = ["age", "gender", "dialect", "native"]
    unique_category_values = [
        df[f"{category}_1"].unique().tolist() + [None] for category in categories
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


def compute_metrics(
    dataset: Dataset,
    transcriber: AutomaticSpeechRecognitionPipeline,
    metric_names: list[str],
    characters_to_keep: str,
    text_column: str,
    audio_column: str,
    batch_size: int,
) -> tuple[list[str], list[str], dict[str, list[float]]]:
    """Compute the metrics for the dataset.

    Args:
        dataset:
            The dataset to validate.
        transcriber:
            The transcriber used for transcribing the audio.
        metric_names:
            The names of the metrics to compute. Needs to be compatible with the name of
            the metric in the `evaluate` library.
        characters_to_keep:
            The characters to keep in the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        audio_column:
            The name of the column containing the audio samples.
        batch_size:
            The batch size to use for transcribing the audio.

    Returns:
        A triple (predictions, labels, all_scores) where:
            predictions:
                The transcriptions predicted by the model.
            labels:
                The ASR-processed ground-truth labels for each sample.
            all_scores:
                A dictionary containing the computed scores for each metric.
    """
    # This contains all the punctuation characters that will be removed from the
    # transcriptions, as they do not have an influence on the pronunciation of the
    # words.
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(characters_to_keep + ' |')}]"
    )

    labels: list[str] = [lbl.strip().lower() for lbl in dataset[text_column]]
    predictions: list[str] = list()

    with tqdm(total=len(dataset), desc="Transcribing") as pbar:
        for out in transcriber(
            KeyDataset(dataset=dataset, key=audio_column), batch_size=batch_size
        ):
            prediction = re.sub(
                pattern=non_standard_characters_regex,
                repl="",
                string=out["text"].strip().lower(),
            )
            predictions.append(prediction.strip())
            pbar.update()

    all_scores: dict[str, list[float]] = dict()
    for metric_name in metric_names:
        metric = load_metric(metric_name)
        scores = [
            metric.compute(predictions=[pred], references=[ref])
            for pred, ref in zip(
                tqdm(predictions, desc=f"Computing {metric_name.upper()}s"), labels
            )
        ]

        # Ensure that the scores are indeed floats, as `compute` returns a dictionary
        # for some metrics
        scores = [score if isinstance(score, float) else -100.0 for score in scores]
        assert all(score >= 0 for score in scores), (
            f"The number of {metric_name.upper()}s should be equal to the number "
            f"of predictions - found {len(scores):,} {metric_name.upper()}s and "
            f"{len(predictions):,} predictions."
        )

        all_scores[metric_name] = scores

    return predictions, labels, all_scores
