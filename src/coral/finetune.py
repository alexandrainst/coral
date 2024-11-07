"""Finetuning ASR models."""

import logging
import os

from omegaconf import DictConfig
from transformers import EarlyStoppingCallback, TrainerCallback

from .data import load_data_for_finetuning
from .data_models import ModelSetup
from .experiment_tracking.extracking_factory import load_extracking_setup
from .model_setup import load_model_setup
from .ngram import train_and_store_ngram_model
from .utils import block_terminal_output, disable_tqdm, push_model_to_hub

logger = logging.getLogger(__package__)


def finetune(config: DictConfig) -> None:
    """Finetune a model on a dataset.

    Args:
        config:
            The Hydra configuration object.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    model_setup: ModelSetup = load_model_setup(config=config)
    processor = model_setup.load_processor()
    processor.save_pretrained(save_directory=config.model_dir)
    model = model_setup.load_model()
    dataset = load_data_for_finetuning(config=config, processor=processor)

    if bool(config.experiment_tracking) and is_main_process:
        extracking_setup = load_extracking_setup(config=config)
        extracking_setup.run_initialization()

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

    block_terminal_output()
    with disable_tqdm():
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    if bool(config.experiment_tracking) and is_main_process:
        extracking_setup.run_finalization()

    model.save_pretrained(save_directory=config.model_dir)

    if hasattr(config.model, "use_decoder") and config.model.use_decoder:
        train_and_store_ngram_model(config=config)

    if config.push_to_hub:
        push_model_to_hub(
            trainer=trainer,
            model_name=config.model_id,
            finetuned_from=config.model.pretrained_model_id,
            create_pr=config.create_pr,
        )


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
