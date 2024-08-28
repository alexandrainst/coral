"""Automatic validation of an ASR dataset.

Usage:
    python src/scripts/validate_coral_asr.py [key=value] [key=value] ...
"""

import logging
import re
import warnings
from time import sleep

import evaluate
import hydra
import torch
from coral.data import filter_dataset, process_dataset
from datasets import Dataset, DatasetDict, enable_progress_bar, load_dataset
from omegaconf import DictConfig
from requests import HTTPError
from tqdm.auto import tqdm
from transformers import AutomaticSpeechRecognitionPipeline, pipeline
from transformers.pipelines.base import KeyDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_coral_asr")

warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(
    config_path="../../config", config_name="dataset_validation", version_base=None
)
def main(config: DictConfig) -> None:
    """Validate the samples of an ASR dataset using an ASR model.

    Args:
        config:
            The Hydra configuration object.
    """
    enable_progress_bar()

    logger.info(f"Loading the {config.dataset_id!r} dataset...")
    dataset = load_dataset(
        path=config.dataset_id,
        name=config.dataset_subset,
        revision=config.dataset_revision,
        token=True,
        cache_dir=config.cache_dir,
    )
    if isinstance(dataset, Dataset):
        dataset = DatasetDict(dict(train=dataset))
    assert isinstance(dataset, DatasetDict)

    if config.filter_dataset:
        dataset = filter_dataset(
            dataset=dataset,
            audio_column=config.audio_column,
            min_seconds_per_example=config.min_seconds_per_example,
            max_seconds_per_example=config.max_seconds_per_example,
            train_name=config.train_name,
        )

    # This contains all the punctuation characters that will be removed from the
    # transcriptions, as they do not have an influence on the pronunciation of the
    # words.
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(config.characters_to_keep + ' |')}]"
    )

    if config.process_dataset:
        dataset = process_dataset(
            dataset=dataset,
            characters_to_keep=config.characters_to_keep,
            text_column=config.text_column,
            audio_column=config.audio_column,
            lower_case=True,
            cast_to_sampling_rate=config.sampling_rate,
        )

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

    metric_names = [metric.name.lower() for metric in config.metrics]
    for split_name, split in dataset.items():
        logger.info(f"Validating the {split_name} split of the dataset...")
        predictions, labels, score_dict = compute_metrics(
            dataset=split,
            transcriber=transcriber,
            metric_names=metric_names,
            non_standard_characters_regex=non_standard_characters_regex,
            text_column=config.text_column,
            audio_column=config.audio_column,
            batch_size=config.batch_size,
        )

        # Create a new split with the predictions, labels, and scores
        dataset[split_name] = (
            dataset[split_name]
            .add_column(name="asr_prediction", column=predictions)
            .add_column(name="asr_label", column=labels)
            .add_column(
                name="asr_validation_model", column=[config.model_id] * len(split)
            )
        )
        for metric_name, scores in score_dict.items():
            dataset[split_name] = dataset[split_name].add_column(
                name=f"asr_{metric_name.lower()}", column=scores
            )

    # We upload here as well as at the end in case we run into an error during the final
    # filtering step
    logger.info(f"Uploading the validated dataset to {config.output_dataset_id!r}...")
    for _ in range(60):
        try:
            dataset.push_to_hub(
                repo_id=config.output_dataset_id,
                config_name=config.output_dataset_subset,
                max_shard_size="500MB",
                commit_message="Add ASR validation",
                private=True,
            )
            logger.info("All done!")
            break
        except (RuntimeError, HTTPError) as e:
            logger.info(f"Error while pushing to hub: {e}")
            logger.info("Waiting a minute before trying again...")
            sleep(60)
            logger.info("Retrying...")
    else:
        logger.error("Failed to upload the dataset to the Hugging Face Hub.")

    # Filter the dataset based on the metrics from the validation model
    num_samples_before = sum(len(split) for split in dataset.values())
    dataset = dataset.filter(
        lambda sample: sample["asr_cer"] < config.max_cer,
        desc=f"Removing samples with CER >= {config.max_cer}",
    )
    num_samples_removed = num_samples_before - sum(
        len(split) for split in dataset.values()
    )
    logger.info(
        f"Removed {num_samples_removed:,} samples based on the validation model."
    )

    logger.info(
        f"Uploading the filtered validated dataset to {config.output_dataset_id!r}..."
    )
    for _ in range(60):
        try:
            dataset.push_to_hub(
                repo_id=config.output_dataset_id,
                config_name=config.output_dataset_subset,
                max_shard_size="500MB",
                commit_message="Filter samples based on the validation model",
                private=True,
            )
            logger.info("All done!")
            break
        except (RuntimeError, HTTPError) as e:
            logger.info(f"Error while pushing to hub: {e}")
            logger.info("Waiting a minute before trying again...")
            sleep(60)
            logger.info("Retrying...")
    else:
        logger.error("Failed to upload the dataset to the Hugging Face Hub.")

    logger.info("All done!")


def compute_metrics(
    dataset: Dataset,
    transcriber: AutomaticSpeechRecognitionPipeline,
    metric_names: list[str],
    non_standard_characters_regex: re.Pattern[str],
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
        non_standard_characters_regex:
            Regular expression that matches all characters that should be removed from
            the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        audio_column:
            The name of the column containing the audio samples.
        batch_size:
            The batch size to use for transcribing the audio.

    Returns:
        A triple (predictions, labels, cers, wers) where:
            predictions:
                The transcriptions predicted by the model.
            labels:
                The ASR-processed ground-truth labels for each sample.
            cers:
                The word error rates for each sample.
            wers:
                The word error rates for each sample.
    """
    labels: list[str] = [lbl.strip().lower() for lbl in dataset[text_column]]
    predictions: list[str] = list()

    with tqdm(total=len(dataset), desc="Transcribing") as pbar:
        for out in transcriber(
            KeyDataset(dataset, audio_column), batch_size=batch_size
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
        metric = evaluate.load(metric_name)
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


if __name__ == "__main__":
    main()
