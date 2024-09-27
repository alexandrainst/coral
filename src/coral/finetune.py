"""Finetuning ASR models."""

import logging
import os

from omegaconf import DictConfig
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    EarlyStoppingCallback,
    TrainerCallback,
)
from wandb.sdk.wandb_init import init as wandb_init

from .data import load_data_for_finetuning
from .data_models import ModelSetup
from .model_setup import load_model_setup
from .utils import push_model_to_hub

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

    # TEMP
    model = AutoModelForCTC.from_pretrained(config.model_dir)
    processor = AutoProcessor.from_pretrained(config.model_dir)
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

    # block_terminal_output()
    # with disable_tqdm():
    #     trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    # if config.wandb and is_main_process:
    #     wandb_finish()

    # model.save_pretrained(save_directory=config.model_dir)

    # if hasattr(config.model, "use_decoder") and config.model.use_decoder:
    #     train_and_store_ngram_model(config=config)

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
