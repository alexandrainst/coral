"""Functions related to the finetuning of Wav2Vec 2.0 models on ASR datasets."""

import logging

import wandb
from datasets import Audio
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
from transformers.trainer import OptimizerNames

from .data import clean_dataset, load_data
from .protocols import ModelSetup
from .wav2vec2 import Wav2Vec2ModelSetup

logger = logging.getLogger(__name__)


def load_model_setup(cfg: DictConfig) -> ModelSetup:
    """Get the model setup for the given configuration.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        ModelSetup:
            The model setup.
    """
    model_type: str = cfg.model.type
    if model_type == "wav2vec2":
        return Wav2Vec2ModelSetup(cfg)
    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")


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
    dataset = clean_dataset(cfg, dataset=dataset)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=cfg.model.sampling_rate))

    def tokenize_examples(example: dict) -> dict:
        example["labels"] = processor(
            text=example[cfg.dataset.text_column], truncation=True
        ).input_ids
        example["input_length"] = len(example["labels"])
        return example

    dataset = dataset.map(tokenize_examples)

    trainer = model_setup.load_trainer_class()(
        model=model,
        data_collator=model_setup.load_data_collator(),
        args=load_training_args(cfg),
        compute_metrics=model_setup.load_compute_metrics(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=processor.tokenizer,
        callbacks=load_callbacks(cfg),
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    model.save_pretrained(cfg.model_dir)
    if cfg.push_to_hub:
        trainer.push_to_hub()

    # TODO: Add ngram model


def load_training_args(cfg: DictConfig) -> TrainingArguments:
    """Load the training arguments for the Trainer.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        TrainingArguments:
            The training arguments.
    """
    if cfg.wandb:
        wandb.init(project=cfg.pipeline_id, name=cfg.wandb_name)

    logger.debug("Initialising training arguments...")
    return TrainingArguments(
        output_dir=cfg.model_dir,
        hub_model_id=cfg.hub_id,
        per_device_train_batch_size=cfg.model.batch_size,
        per_device_eval_batch_size=cfg.model.batch_size,
        gradient_accumulation_steps=cfg.model.gradient_accumulation,
        learning_rate=cfg.model.learning_rate,
        warmup_steps=cfg.model.warmup_steps,
        max_steps=cfg.model.max_steps,
        fp16=cfg.model.fp16 and not mps_is_available(),
        push_to_hub=cfg.push_to_hub,
        evaluation_strategy="steps",
        eval_steps=cfg.model.eval_steps,
        save_steps=cfg.model.save_steps,
        logging_steps=cfg.model.logging_steps,
        length_column_name="input_length",
        gradient_checkpointing=True,
        save_total_limit=cfg.model.save_total_limit,
        load_best_model_at_end=cfg.model.early_stopping,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=4242,
        remove_unused_columns=False,
        optim=OptimizerNames.ADAMW_TORCH,
        use_mps_device=mps_is_available(),
        report_to=["wandb"] if cfg.wandb else [],
        ignore_data_skip=cfg.ignore_data_skip,
    )


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
    if cfg.model.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.model.early_stopping_patience
        )
        callbacks = [early_stopping_callback]
    return callbacks
