"""Functions related to the finetuning of Wav2Vec 2.0 models on ASR datasets."""

import logging

from datasets import Audio
from omegaconf import DictConfig
from transformers import EarlyStoppingCallback, TrainerCallback
from wandb.sdk.wandb_init import init as wandb_init

from .data import clean_dataset, load_data
from .model_setup import load_model_setup
from .protocols import ModelSetup
from .utils import disable_tqdm

logger = logging.getLogger(__name__)


def finetune(cfg: DictConfig) -> None:
    """Finetune a model on a dataset.

    Args:
        cfg (DictConfig):
            The Hydra cfguration object.
    """
    model_setup: ModelSetup = load_model_setup(cfg)
    processor = model_setup.load_processor()
    processor.save_pretrained(cfg.model_dir)
    model = model_setup.load_model()

    dataset = load_data(cfg)
    if cfg.model.clean_dataset:
        dataset = clean_dataset(cfg, dataset=dataset)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=cfg.model.sampling_rate))

    def prepare_dataset(example: dict) -> dict:
        # Prepare audio
        audio = example["audio"]
        processed = processor(audio["array"], sampling_rate=audio["sampling_rate"])
        if "input_values" in processed:
            example["input_values"] = processed.input_values[0]
        if "input_features" in processed:
            example["input_features"] = processed.input_features[0]

        # Prepare transcriptions
        example["labels"] = processor(
            text=example[cfg.dataset.text_column], truncation=True
        ).input_ids
        example["input_length"] = len(example["labels"])

        return example

    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

    if cfg.wandb:
        wandb_init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
        )

    trainer = model_setup.load_trainer_class()(
        model=model,
        data_collator=model_setup.load_data_collator(),
        args=model_setup.load_training_arguments(),
        compute_metrics=model_setup.load_compute_metrics(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=getattr(processor, "tokenizer"),
        callbacks=load_callbacks(cfg),
    )

    with disable_tqdm():
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    model.save_pretrained(cfg.model_dir)
    if cfg.push_to_hub:
        trainer.push_to_hub()


def load_callbacks(cfg: DictConfig) -> list[TrainerCallback]:
    """Load the callbacks for the Trainer.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        list of TrainerCallback:
            The callbacks.
    """
    callbacks: list[TrainerCallback] = list()
    if cfg.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience
        )
        callbacks = [early_stopping_callback]
    return callbacks
