"""Functions related to the finetuning of Wav2Vec 2.0 models on ASR datasets."""

from functools import partial

from omegaconf import DictConfig
from transformers import (
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2ForCTC,
)

from ..data import load_data
from .compute_metrics import compute_metrics
from .data_collator import DataCollatorCTCWithPadding
from .preprocess import Processor


def finetune(cfg: DictConfig) -> None:
    """Finetune a Wav2Vec 2.0 model on a given ASR dataset.

    Args:
        cfg (DictConfig):
            The Hydra cfguration object.
    """
    # Load dataset
    dataset = load_data(cfg)

    # Preprocess the dataset
    processor = Processor(cfg)
    dataset = processor.preprocess_audio(dataset=dataset)
    dataset = processor.preprocess_transcriptions(dataset=dataset)

    # Initialise data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor.wav2vec2_processor, padding="longest"
    )

    # Initialise the model
    tokenizer: PreTrainedTokenizerBase = processor.wav2vec2_processor.tokenizer
    model = Wav2Vec2ForCTC.from_pretrained(
        cfg.model.pretrained_model_id,
        activation_dropout=cfg.activation_dropout,
        attention_dropout=cfg.attention_dropout,
        hidden_dropout=cfg.hidden_dropout,
        feat_proj_dropout=cfg.feat_proj_dropout,
        final_dropout=cfg.final_dropout,
        mask_time_prob=cfg.mask_time_prob,
        mask_feature_prob=cfg.mask_feature_prob,
        mask_feature_length=cfg.mask_feature_length,
        layerdrop=cfg.layerdrop,
        ctc_loss_reduction=cfg.ctc_loss_reduction,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=len(tokenizer.get_vocab()),
    )

    if not isinstance(model, Wav2Vec2ForCTC):
        raise TypeError("The model must be a Wav2Vec2ForCTC model.")

    # Freeze the feature encoder
    if cfg.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # Initialise training arguments
    training_args = TrainingArguments(
        output_dir=cfg.finetuned_model_id.split("/")[-1],
        hub_model_id=cfg.finetuned_model_id,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.epochs,
        fp16=cfg.fp16,
        push_to_hub=cfg.push_to_hub,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        group_by_length=True,
        length_column_name="input_length",
        gradient_checkpointing=True,
        save_total_limit=2,
        load_best_model_at_end=cfg.early_stopping,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=4242,
        remove_unused_columns=False,
    )

    # Create early stopping callback
    callbacks: list[TrainerCallback] = list()
    if cfg.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience
        )
        callbacks = [early_stopping_callback]

    # Initialise the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=partial(
            compute_metrics, processor=processor.wav2vec2_processor
        ),
        train_dataset=dataset["train"],  # type: ignore
        eval_dataset=dataset["val"],  # type: ignore
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # Save the preprocessor
    processor.wav2vec2_processor.save_pretrained(cfg.finetuned_model_id.split("/")[-1])

    # Train the model
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    # Save the model
    model.save_pretrained(cfg.finetuned_model_id.split("/")[-1])

    # Push the model to the hub
    if cfg.push_to_hub:
        trainer.push_to_hub()
