"""Automatic validation of an ASR dataset.

Usage:
    python src/scripts/validate_coral_asr.py [key=value] [key=value] ...
"""

import logging
import warnings
from time import sleep

import hydra
import torch
from coral.compute_metrics import compute_metrics_of_dataset_using_pipeline
from coral.data import filter_dataset, process_dataset
from datasets import Dataset, DatasetDict, enable_progress_bar, load_dataset
from omegaconf import DictConfig
from requests import HTTPError
from transformers import AutomaticSpeechRecognitionPipeline, pipeline

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
        predictions, labels, score_dict = compute_metrics_of_dataset_using_pipeline(
            dataset=split,
            transcriber=transcriber,
            metric_names=metric_names,
            characters_to_keep=config.characters_to_keep,
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


if __name__ == "__main__":
    main()
