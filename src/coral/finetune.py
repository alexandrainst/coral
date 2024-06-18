"""Finetuning ASR models."""

import logging
import os
from functools import partial
from typing import Callable

from omegaconf import DictConfig
from transformers import EarlyStoppingCallback, TrainerCallback
from wandb.sdk.wandb_init import init as wandb_init
from wandb.sdk.wandb_run import finish as wandb_finish

from .data import load_data
from .data_models import ModelSetup
from .model_setup import load_model_setup
from .utils import disable_tqdm

logger = logging.getLogger(__package__)


def prepare_dataset_example(example: dict, processor: Callable) -> dict:
    """Prepare a dataset example for the model.

    Args:
        example: The example from the dataset.
        processor: The processor to use.

    Returns:
        The prepared example.
    """
    # Prepare audio
    audio = example["audio"]
    sampling_rate = audio["sampling_rate"]
    processed = processor(audio["array"], sampling_rate=sampling_rate)
    if "input_values" in processed:
        example["input_values"] = processed.input_values[0]
        example["num_seconds"] = len(example["input_values"]) / sampling_rate
    if "input_features" in processed:
        example["input_features"] = processed.input_features[0]
        example["num_seconds"] = len(example["input_features"]) / sampling_rate

    # Prepare transcriptions
    example["labels"] = processor(text=example["text"], truncation=True).input_ids
    example["input_length"] = len(example["labels"])

    return example


def example_audio_is_short(example: dict, max_seconds_per_example: int) -> bool:
    """Check if the example audio is too short.

    Args:
        example: The example from the dataset.
        max_seconds_per_example: The maximum number of seconds per example.

    Returns:
        Whether the example audio is too short.
    """
    return example["num_seconds"] <= max_seconds_per_example


def finetune(cfg: DictConfig) -> None:
    """Finetune a model on a dataset.

    Args:
        cfg: The Hydra cfguration object.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    model_setup: ModelSetup = load_model_setup(cfg)
    processor = model_setup.load_processor()
    processor.save_pretrained(cfg.model_dir)
    model = model_setup.load_model()
    dataset = load_data(cfg)

    dataset = dataset.map(
        function=partial(prepare_dataset_example, processor=processor),
        remove_columns=dataset["train"].column_names,
    )
    dataset = dataset.filter(
        function=partial(
            example_audio_is_short, max_seconds_per_example=cfg.max_seconds_per_example
        )
    )

    if cfg.wandb and is_main_process:
        wandb_init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

    if "val" not in dataset and is_main_process:
        logger.info("No validation set found. Disabling early stopping.")

    trainer = model_setup.load_trainer_class()(
        model=model,
        data_collator=model_setup.load_data_collator(),
        args=model_setup.load_training_arguments(),
        compute_metrics=model_setup.load_compute_metrics(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if "val" in dataset else None,
        tokenizer=getattr(processor, "tokenizer"),
        callbacks=load_early_stopping_callback(cfg) if "val" in dataset else None,
    )

    with disable_tqdm():
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    if is_main_process:
        wandb_finish()
        model.save_pretrained(cfg.model_dir)
        if cfg.push_to_hub:
            trainer.push_to_hub()


def load_early_stopping_callback(cfg: DictConfig) -> list[TrainerCallback]:
    """Load the early stopping callback for the trainer.

    Args:
        cfg: The Hydra configuration object.

    Returns:
        The callbacks.
    """
    callbacks: list[TrainerCallback] = list()
    if cfg.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience
        )
        callbacks = [early_stopping_callback]
    return callbacks
