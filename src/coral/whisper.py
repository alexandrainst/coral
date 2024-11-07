"""Model setup for Whisper models."""

import logging
import os
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Type

import torch
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    EvalPrediction,
    SchedulerType,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer import OptimizerNames

from .compute_metrics import compute_wer_metrics
from .data_collators import DataCollatorSpeechSeq2SeqWithPadding
from .data_models import ModelSetup, PreTrainedModelData, Processor
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


class WhisperModelSetup(ModelSetup):
    """Model setup for Whisper models."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        self.config = config
        self.processor: WhisperProcessor
        self.is_main_process = os.getenv("RANK", "0") == "0"

    def load_processor(self) -> WhisperProcessor:
        """Return the processor for the model."""
        processor_or_tup = WhisperProcessor.from_pretrained(
            self.config.model.pretrained_model_id, language="Danish", task="transcribe"
        )
        assert isinstance(processor_or_tup, WhisperProcessor)
        self.processor = processor_or_tup

        # Whisper tokenizers are misconfigured with a max_length that is too high, but
        # the correct max_length is stored in the model config, so we'll update it here.
        hf_config = AutoConfig.from_pretrained(self.config.model.pretrained_model_id)
        self.processor.tokenizer.model_max_length = min(
            self.processor.tokenizer.model_max_length, hf_config.max_length
        )

        return self.processor

    def load_model(self) -> WhisperForConditionalGeneration:
        """Return the model for the setup."""
        with transformers_output_ignored():
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model.pretrained_model_id,
                dropout=self.config.model.dropout,
                activation_dropout=self.config.model.activation_dropout,
                attention_dropout=self.config.model.attention_dropout,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                apply_spec_augment=True,
                mask_time_prob=self.config.model.mask_time_prob,
                mask_time_length=self.config.model.mask_time_length,
                mask_feature_prob=self.config.model.mask_feature_prob,
                mask_feature_length=self.config.model.mask_feature_length,
                encoder_layerdrop=self.config.model.layerdrop,
                decoder_layerdrop=self.config.model.layerdrop,
            )
            assert isinstance(model, WhisperForConditionalGeneration)

        if self.config.model.freeze_feature_encoder:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.proj_out.parameters():
                param.requires_grad = True

        # The Whisper model has token ids that are forced as model outputs before
        # autoregressive generation is started (forced_decoder_ids). These token ids
        # control the transcription language and task for zero-shot ASR. For
        # fine-tuning, we'll set these ids to None, as we'll train the model to predict
        # the correct language and task. There are also tokens that are completely
        # suppressed during generation (suppress_tokens). These tokens have their log
        # probabilities set to -inf, such that they are never sampled. We'll override
        # these tokens to an empty list, meaning no tokens are suppressed.
        # Source: https://hf.co/blog/fine-tune-whisper#load-a-pre-trained-checkpoint
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        # Disabling cache as this is incompatible with gradient checkpointing
        model.config.use_cache = False

        return model

    def load_data_collator(self) -> DataCollatorSpeechSeq2SeqWithPadding:
        """Return the data collator for the model."""
        return DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            max_seconds_per_example=self.config.max_seconds_per_example,
            padding=self.config.padding,
        )

    def load_trainer_class(self) -> Type[Trainer]:
        """Return the trainer class used to train the model."""
        return Seq2SeqTrainer

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        """Return the function used to compute metrics during training."""
        return partial(compute_wer_metrics, processor=self.processor)

    def load_training_arguments(self) -> TrainingArguments:
        """Return the training arguments for the model."""
        # Compute the gradient accumulation based on the total batch size in the config
        num_devices = max(torch.cuda.device_count(), 1)
        per_device_total_batch_size = self.config.total_batch_size // num_devices
        gradient_accumulation_steps = (
            per_device_total_batch_size // self.config.per_device_batch_size
        )

        if gradient_accumulation_steps == 0:
            if self.is_main_process:
                logger.warning(
                    "Your `total_batch_size` is too small "
                    f"({self.config.total_batch_size}), relative to the number of "
                    f"devices ({num_devices}) and your `per_device_batch_size` "
                    f"({self.config.per_device_batch_size}). It has been set to "
                    "`per_device_batch_size * num_devices` = "
                    f"{self.config.per_device_batch_size * num_devices}."
                )
            gradient_accumulation_steps = 1

        fp16 = False
        bf16 = False
        if not mps_is_available():
            if self.config.bf16_allowed and torch.cuda.is_bf16_supported():
                bf16 = True
                if self.is_main_process:
                    logger.info("Mixed precision training with BF16 enabled.")
            elif self.config.fp16_allowed and torch.cuda.is_available():
                fp16 = True
                if self.is_main_process:
                    logger.info("Mixed precision training with FP16 enabled.")

        if self.config.early_stopping:
            self.config.save_total_limit = max(self.config.save_total_limit, 1)

        args = Seq2SeqTrainingArguments(
            output_dir=self.config.model_dir,
            hub_model_id=f"{self.config.hub_organisation}/{self.config.model_id}",
            hub_private_repo=self.config.private,
            per_device_train_batch_size=self.config.per_device_batch_size,
            per_device_eval_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.config.model.learning_rate,
            lr_scheduler_type=SchedulerType.COSINE,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            fp16=fp16,
            bf16=bf16,
            push_to_hub=False,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_strategy="no" if self.config.save_total_limit == 0 else "steps",
            logging_steps=self.config.logging_steps,
            length_column_name="input_length",
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.early_stopping,
            metric_for_best_model="wer",
            greater_is_better=False,
            seed=self.config.seed,
            remove_unused_columns=False,
            optim=OptimizerNames.ADAMW_TORCH,
            adam_beta1=self.config.adam_first_momentum,
            adam_beta2=self.config.adam_second_momentum,
            report_to=[self.config.experiment_tracking.type]
            if self.config.experiment_tracking
            else [],
            ignore_data_skip=self.config.ignore_data_skip,
            save_safetensors=True,
            predict_with_generate=True,
            generation_max_length=self.config.model.max_length,
            use_cpu=hasattr(sys, "_called_from_test"),
            dataloader_num_workers=self.config.dataloader_num_workers,
            ddp_find_unused_parameters=False,
            dispatch_batches=False,
        )
        return args

    def load_saved(self) -> PreTrainedModelData:
        """Load the model setup."""
        if Path(self.config.model_dir).exists():
            model_path = self.config.model_dir
        else:
            model_path = f"{self.config.hub_organisation}/{self.config.model_id}"

        processor: Processor
        processor_or_tup = WhisperProcessor.from_pretrained(model_path, token=True)
        assert isinstance(processor_or_tup, WhisperProcessor)
        processor = processor_or_tup

        model_or_tup = WhisperForConditionalGeneration.from_pretrained(
            model_path, token=True
        )
        assert isinstance(model_or_tup, WhisperForConditionalGeneration)
        model = model_or_tup

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            max_seconds_per_example=self.config.max_seconds_per_example,
            padding=self.config.padding,
        )
        compute_metrics = partial(compute_wer_metrics, processor=processor)
        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
