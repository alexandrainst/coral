"""Automatic validation of an ASR dataset.

Usage:
    python src/scripts/validate_coral_asr.py [key=value] [key=value] ...
"""

import logging
from time import sleep

import hydra
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig
from requests import HTTPError

from coral.data import filter_dataset
from coral.utils import interpret_dataset_name
from coral.validation import add_validations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(name)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_coral_asr")


@hydra.main(
    config_path="../../config", config_name="dataset_validation", version_base=None
)
def main(config: DictConfig) -> None:
    """Validate the samples of an ASR dataset using an ASR model.

    Args:
        config:
            The Hydra configuration object.
    """
    dataset_id, dataset_subset, dataset_revision = interpret_dataset_name(
        dataset_name=config.dataset
    )

    logger.info(f"Loading the {config.dataset_id!r} dataset...")
    dataset = load_dataset(
        path=dataset_id,
        name=dataset_subset,
        revision=dataset_revision,
        token=True,
        cache_dir=config.cache_dir,
    )
    if isinstance(dataset, Dataset):
        dataset = DatasetDict(dict(train=dataset))
    assert isinstance(dataset, DatasetDict)

    dataset = filter_dataset(
        dataset=dataset,
        audio_column=config.audio_column,
        min_seconds_per_example=config.min_seconds_per_example,
        max_seconds_per_example=config.max_seconds_per_example,
        is_main_process=True,
    )

    dataset = add_validations(
        dataset=dataset,
        text_column=config.text_column,
        audio_column=config.audio_column,
        model_id=config.model_id,
        clean_text=config.clean_text,
        lower_case=config.lower_case,
        sampling_rate=config.sampling_rate,
        characters_to_keep=config.characters_to_keep,
        batch_size=config.batch_size,
        max_cer=config.max_cer,
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
