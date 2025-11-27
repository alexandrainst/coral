"""Evaluation of ASR models."""

import itertools as it
import logging

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.pipelines import pipeline
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

from .data import DEFAULT_CONVERSION_DICT, load_dataset_for_evaluation, process_example
from .metrics import cer, wer
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
    assert config.model_id is not None, (
        "`model_id` must be set to perform an evaluation!"
    )

    logger.info("Loading the dataset...")
    dataset = load_dataset_for_evaluation(config=config)

    logger.info(f"Loading the {config.model_id!r} ASR model...")
    transcriber = load_asr_pipeline(model_id=config.model_id, no_lm=config.no_lm)

    predictions: list[str] = list()
    with (
        tqdm(total=len(dataset), desc="Transcribing") as pbar,
        transformers_output_ignored(),
    ):
        for out in transcriber(
            KeyDataset(dataset=dataset, key=config.audio_column),  # type: ignore[arg-type]
            batch_size=config.batch_size,
            generate_kwargs=dict(language="danish", task="transcribe"),
        ):
            prediction = process_example(
                example=dict(text=out["text"]),
                characters_to_keep="".join(config.characters_to_keep),
                conversion_dict=DEFAULT_CONVERSION_DICT,
                text_column="text",
                audio_column=None,
                lower_case=True,
                convert_numerals=True,
                processor=None,
            )["text"]
            predictions.append(prediction)
            pbar.update()

    logger.info(
        "Converting the dataset to a dataframe computing the scores for each "
        "metadata category..."
    )
    df = convert_evaluation_dataset_to_df(
        dataset=dataset, sub_dialect_to_dialect_mapping=config.sub_dialect_to_dialect
    )
    score_df = get_score_df(df=df, categories=["age_group", "gender", "dialect"])
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


def get_score_df(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Get the score DataFrame for the evaluation dataset.

    Args:
        df:
            The evaluation dataframe, containing all the metadata columns and columns
            for each metric.
        categories:
            The categories to evaluate.

    Returns:
        The score DataFrame.
    """
    unique_category_values = [
        df[category].unique().tolist() + [None] for category in categories
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
        score_dict = dict(
            cer=cer(predictions=df_filtered.prediction, labels=df_filtered.text),
            wer=wer(predictions=df_filtered.prediction, labels=df_filtered.text),
        )
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
            f"{key.upper()} = {value:.4f}" for key, value in score_dict.items()
        )
        logger.info(f"Scores for {combination_str}: {score_str}")

    score_df = pd.DataFrame.from_records(data=records)
    return score_df
