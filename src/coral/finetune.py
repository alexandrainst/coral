"""Finetuning ASR models."""

import logging
import os

from omegaconf import DictConfig
from transformers.trainer_callback import EarlyStoppingCallback

from .data import load_data_for_finetuning
from .data_models import ModelSetup
from .experiment_tracking import ExTrackingSetup, load_extracking_setup
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

    extracking_setup: ExTrackingSetup | None = None
    if config.enable_experiment_tracking and is_main_process:
        extracking_setup = load_extracking_setup(config=config)
        extracking_setup.run_initialization()

    vals = {
        split_name: split
        for split_name, split in dataset.items()
        if split_name.startswith("val")
    }
    match len(vals):
        case 0:
            eval_dataset = None
        case 1:
            eval_dataset = list(vals.values())[0]
        case _:
            eval_dataset = vals

    if eval_dataset is None and is_main_process:
        logger.info("No validation set found. Disabling early stopping.")

    trainer = model_setup.load_trainer_class()(
        model=model,
        data_collator=model_setup.load_data_collator(),
        args=model_setup.load_training_arguments(),
        compute_metrics=model_setup.load_compute_metrics(),
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=getattr(processor, "tokenizer"),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ]
        if eval_dataset is not None and config.early_stopping
        else None,
    )

    block_terminal_output()
    with disable_tqdm():
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    if extracking_setup is not None and is_main_process:
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
