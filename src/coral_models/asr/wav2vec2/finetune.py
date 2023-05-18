"""Functions related to the finetuning of Wav2Vec 2.0 models on ASR datasets."""

import logging
from functools import partial

from datasets import DatasetDict, IterableDatasetDict
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2ForCTC,
)
from transformers.trainer import OptimizerNames

from ...data import load_data
from ...utils import transformers_output_ignored
from ..utils import dump_vocabulary
from .clean import clean_dataset
from .compute_metrics import compute_metrics
from .data_collator import DataCollatorCTCWithPadding
from .preprocess import ModifiedWav2Vec2Processor, load_processor

logger = logging.getLogger(__name__)


def finetune_wav2vec2(cfg: DictConfig) -> None:
    """Finetune a Wav2Vec 2.0 model on a given ASR dataset.

    Args:
        cfg (DictConfig):
            The Hydra cfguration object.
    """
    dataset, processor = load_preprocessed_dataset(cfg)
    model = load_model(cfg, processor=processor)

    logger.debug("Initialising trainer...")
    trainer = Trainer(
        model=model,
        data_collator=load_data_collator(processor=processor),
        args=load_training_args(cfg),
        compute_metrics=partial(compute_metrics, processor=processor),
        train_dataset=dataset["train"],  # type: ignore
        eval_dataset=dataset["val"],  # type: ignore
        tokenizer=processor.tokenizer,
        callbacks=load_callbacks(cfg),
    )

    logger.debug("Saving preprocessor...")
    processor.save_pretrained(cfg.model_dir)

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    logger.info("Saving model...")
    model.save_pretrained(cfg.model_dir)

    if cfg.push_to_hub:
        logger.info("Pushing model to the hub...")
        trainer.push_to_hub()

    # TODO: Add ngram model


def load_preprocessed_dataset(
    cfg: DictConfig,
) -> tuple[DatasetDict | IterableDatasetDict, ModifiedWav2Vec2Processor]:
    """Load the dataset, clean it and preprocess it.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        pair of DatasetDict or IterableDataset, and ModifiedWav2Vec2Processor:
            The preprocessed dataset, and the processor used to preprocess it.
    """
    logger.info("Loading dataset...")
    dataset = load_data(cfg)

    logger.info("Setting up dataset...")

    logger.debug("Cleaning vocabulary...")
    dataset = clean_dataset(cfg, dataset=dataset)

    logger.debug("Dumping vocabulary...")
    dump_vocabulary(cfg)

    logger.debug("Preprocessing dataset...")
    processor = load_processor(cfg)
    dataset = processor.preprocess(dataset=dataset)

    return dataset, processor


def load_training_args(cfg: DictConfig) -> TrainingArguments:
    """Load the training arguments for the Trainer.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        TrainingArguments:
            The training arguments.
    """
    logger.debug("Initialising training arguments...")
    return TrainingArguments(
        output_dir=cfg.model_dir,
        hub_model_id=cfg.hub_id,
        per_device_train_batch_size=cfg.model.batch_size,
        per_device_eval_batch_size=cfg.model.batch_size,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
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
    )


def load_model(cfg: DictConfig, processor: ModifiedWav2Vec2Processor) -> Wav2Vec2ForCTC:
    """Load the Wav2Vec 2.0 model.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
        processor (ModifiedWav2Vec2Processor):
            The processor object.

    Returns:
        Wav2Vec2ForCTC:
            The Wav2Vec 2.0 model.
    """
    with transformers_output_ignored():
        logger.info("Initialising model...")
        model = Wav2Vec2ForCTC.from_pretrained(
            cfg.model.pretrained_model_id,
            activation_dropout=cfg.model.activation_dropout,
            attention_dropout=cfg.model.attention_dropout,
            hidden_dropout=cfg.model.hidden_dropout,
            feat_proj_dropout=cfg.model.feat_proj_dropout,
            final_dropout=cfg.model.final_dropout,
            mask_time_prob=cfg.model.mask_time_prob,
            mask_feature_prob=cfg.model.mask_feature_prob,
            mask_feature_length=cfg.model.mask_feature_length,
            layerdrop=cfg.model.layerdrop,
            ctc_loss_reduction=cfg.model.ctc_loss_reduction,
            pad_token_id=processor.tokenizer.pad_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            vocab_size=len(processor.tokenizer.get_vocab()),
        )
        assert isinstance(model, Wav2Vec2ForCTC)

    if cfg.model.freeze_feature_encoder:
        logger.debug("Freezing feature encoder...")
        for param in model.wav2vec2.parameters():
            param.requires_grad = False

    return model


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
        logger.debug("Initialising early stopping callback...")
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.model.early_stopping_patience
        )
        callbacks = [early_stopping_callback]
    return callbacks


def load_data_collator(
    processor: ModifiedWav2Vec2Processor,
) -> DataCollatorCTCWithPadding:
    """Load the data collator.

    Args:
        processor (ModifiedWav2Vec2Processor):
            The processor object.

    Returns:
        DataCollatorCTCWithPadding:
            The data collator.
    """
    logger.debug("Initialising data collator...")
    return DataCollatorCTCWithPadding(processor=processor, padding="longest")
