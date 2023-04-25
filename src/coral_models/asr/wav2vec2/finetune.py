"""Functions related to the finetuning of Wav2Vec 2.0 models on ASR datasets."""

import logging
from functools import partial

from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2ForCTC,
)
from transformers.trainer import OptimizerNames

from ...data import load_data
from ...utils import ignore_transformers_output
from ..utils import dump_vocabulary
from .clean import clean_dataset
from .compute_metrics import compute_metrics
from .data_collator import DataCollatorCTCWithPadding
from .preprocess import load_processor

# Set up logging
logger = logging.getLogger(__name__)


def finetune_wav2vec2(cfg: DictConfig) -> None:
    """Finetune a Wav2Vec 2.0 model on a given ASR dataset.

    Args:
        cfg (DictConfig):
            The Hydra cfguration object.
    """
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_data(cfg)

    logger.info("Setting up dataset...")

    # Clean the dataset
    logger.debug("Cleaning vocabulary...")
    dataset = clean_dataset(dataset=dataset)

    # Dump the vocabulary
    logger.debug("Dumping vocabulary...")
    dump_vocabulary(cfg, dataset=dataset[cfg.dataset.train_name])

    # Preprocess the dataset
    logger.debug("Preprocessing dataset...")
    processor = load_processor(cfg)
    dataset = processor.preprocess(dataset=dataset)

    # Initialise data collator
    logger.debug("Initialising data collator...")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

    # Initialise the model
    logger.debug("Initialising model")
    tokenizer: PreTrainedTokenizerBase = processor.tokenizer
    with ignore_transformers_output():
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
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            vocab_size=len(tokenizer.get_vocab()),
        )
        assert isinstance(model, Wav2Vec2ForCTC)

    # Freeze the feature encoder
    if cfg.model.freeze_feature_encoder:
        logger.debug("Freezing feature encoder...")
        model.freeze_feature_encoder()

    # Initialise training arguments
    logger.debug("Initialising training arguments...")
    training_args = TrainingArguments(
        output_dir=cfg.model_dir,
        hub_model_id=cfg.pipeline_id,
        per_device_train_batch_size=cfg.model.batch_size,
        per_device_eval_batch_size=cfg.model.batch_size,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
        learning_rate=cfg.model.learning_rate,
        warmup_steps=cfg.model.warmup_steps,
        num_train_epochs=cfg.model.epochs,
        fp16=cfg.model.fp16 and not mps_is_available(),
        push_to_hub=cfg.push_to_hub,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        group_by_length=True,
        length_column_name="input_length",
        gradient_checkpointing=True,
        save_total_limit=2,
        load_best_model_at_end=cfg.model.early_stopping,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=4242,
        remove_unused_columns=False,
        optim=OptimizerNames.ADAMW_TORCH,
        use_mps_device=mps_is_available(),
        report_to=[],
    )

    # Create early stopping callback
    logger.debug("Initialising early stopping callback...")
    callbacks: list[TrainerCallback] = list()
    if cfg.model.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.model.early_stopping_patience
        )
        callbacks = [early_stopping_callback]

    # Initialise the trainer
    logger.debug("Initialising trainer...")
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=partial(compute_metrics, processor=processor),
        train_dataset=dataset["train"],  # type: ignore
        eval_dataset=dataset["val"],  # type: ignore
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Save the preprocessor
    logger.debug("Saving preprocessor...")
    processor.save_pretrained(cfg.model_dir)

    # Train the model
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # Save the model
    logger.info("Saving model...")
    model.save_pretrained(cfg.model_dir)

    # Push the model to the hub
    if cfg.push_to_hub:
        logger.info("Pushing model to the hub...")
        trainer.push_to_hub()
