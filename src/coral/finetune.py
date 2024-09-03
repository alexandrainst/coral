"""Finetuning ASR models."""

import logging
import os
import warnings

from omegaconf import DictConfig
from transformers import EarlyStoppingCallback, TrainerCallback
from wandb.sdk.wandb_init import init as wandb_init
from wandb.sdk.wandb_run import finish as wandb_finish

from .data import load_data_for_finetuning
from .data_models import ModelSetup
from .model_setup import load_model_setup
from .ngram import train_ngram_model
from .utils import disable_tqdm

logger = logging.getLogger(__package__)


def finetune(config: DictConfig) -> None:
    """Finetune a model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    model_dir = config.model_dir
    model_setup: ModelSetup = load_model_setup(config=config)
    processor = model_setup.load_processor()
    processor.save_pretrained(model_dir)
    model = model_setup.load_model()
    dataset = load_data_for_finetuning(config=config, processor=processor)
    breakpoint()

    if config.wandb and is_main_process:
        wandb_init(
            project=config.wandb_project,
            group=config.wandb_group,
            name=config.wandb_name,
            config=dict(config),
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
        callbacks=load_early_stopping_callback(config) if "val" in dataset else None,
    )

    with disable_tqdm():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    if is_main_process:
        wandb_finish()
        model.save_pretrained(model_dir)
        if config.push_to_hub:
            trainer.push_to_hub()

    if hasattr(config.model, "decoder") and config.model.decoder is not None:
        train_ngram_model(config=config)


def load_early_stopping_callback(config: DictConfig) -> list[TrainerCallback]:
    """Load the early stopping callback for the trainer.

    Args:
        config:
            The Hydra configuration object.

    Returns:
        The callbacks.
    """
    callbacks: list[TrainerCallback] = list()
    if config.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience
        )
        callbacks = [early_stopping_callback]
    return callbacks
