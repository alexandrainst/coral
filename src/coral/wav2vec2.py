"""Model setup for Wav2Vec 2.0 models."""

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Type

import torch
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    EvalPrediction,
    SchedulerType,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
)
from transformers.trainer import OptimizerNames

from .compute_metrics import compute_wer_metrics
from .data_collators import DataCollatorCTCWithPadding
from .data_models import ModelSetup, PreTrainedModelData, Processor
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


class Wav2Vec2ModelSetup(ModelSetup):
    """Model setup for Wav2Vec 2.0 models."""

    def __init__(self, config: DictConfig) -> None:
        """Initialise the model setup.

        Args:
            config:
                The Hydra configuration object.
        """
        self.config = config
        self.processor: Processor
        self.is_main_process = os.getenv("RANK", "0") == "0"

    def load_processor(self) -> Wav2Vec2Processor:
        """Return the processor for the model."""
        # We dump the vocabulary to a file since the tokenizer uses this file during
        # initialisation
        while True:
            try:
                dump_vocabulary(self.config)
                tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                    self.config.model_dir,
                    pad_token="<pad>",
                    unk_token="<unk>",
                    bos_token="<s>",
                    eos_token="</s>",
                    word_delimiter_token="|",
                    replace_word_delimiter_char=" ",
                )
                break
            except json.decoder.JSONDecodeError:
                log_message = "JSONDecodeError while loading tokenizer"
                process_id = os.getenv("RANK")
                if process_id is not None:
                    log_message += f" in process {process_id}"
                log_message += ". Retrying in a second."
                if self.is_main_process:
                    logger.warning(log_message)
                time.sleep(1)

        # Set the `model_max_length` attribute of the tokenizer, if it hasn't been set,
        # to ensure that truncation is done correctly
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e6:
            tokenizer.model_max_length = 512

        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.config.model.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=extractor, tokenizer=tokenizer
        )

        return self.processor

    def load_model(self) -> Wav2Vec2ForCTC:
        """Return the model for the model."""
        with transformers_output_ignored():
            model = Wav2Vec2ForCTC.from_pretrained(
                self.config.model.pretrained_model_id,
                activation_dropout=self.config.model.activation_dropout,
                attention_dropout=self.config.model.attention_dropout,
                hidden_dropout=self.config.model.hidden_dropout,
                feat_proj_dropout=self.config.model.feat_proj_dropout,
                final_dropout=self.config.model.final_dropout,
                apply_spec_augment=True,
                mask_time_prob=self.config.model.mask_time_prob,
                mask_time_length=self.config.model.mask_time_length,
                mask_feature_prob=self.config.model.mask_feature_prob,
                mask_feature_length=self.config.model.mask_feature_length,
                layerdrop=self.config.model.layerdrop,
                ctc_loss_reduction=self.config.model.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                vocab_size=len(self.processor.tokenizer.get_vocab()),
                ctc_zero_infinity=True,
            )
        assert isinstance(model, Wav2Vec2ForCTC)

        if self.config.model.freeze_feature_encoder:
            for param in model.wav2vec2.parameters():
                param.requires_grad = False

        return model

    def load_data_collator(self) -> DataCollatorCTCWithPadding:
        """Return the data collator for the model."""
        return DataCollatorCTCWithPadding(
            processor=self.processor,
            max_seconds_per_example=self.config.max_seconds_per_example,
            padding=self.config.padding,
        )

    def load_trainer_class(self) -> Type[Trainer]:
        """Return the trainer class for the model."""
        return Trainer

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        """Return the compute metrics function for the model."""
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

        args = TrainingArguments(
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
            use_cpu=hasattr(sys, "_called_from_test"),
            dataloader_num_workers=self.config.dataloader_num_workers,
            ddp_find_unused_parameters=False,
            dispatch_batches=False,
        )
        return args

    def load_saved(self) -> PreTrainedModelData:
        """Return the saved model data for the model."""
        if Path(self.config.model_dir).exists():
            model_path = self.config.model_dir
        else:
            model_path = f"{self.config.hub_organisation}/{self.config.model_id}"

        processor: Wav2Vec2Processor | Wav2Vec2ProcessorWithLM
        if self.config.model.decoder is not None:
            try:
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                    model_path, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
                )
            except (FileNotFoundError, ValueError):
                raise FileNotFoundError(
                    "The model was trained with a language model decoder, but the "
                    "language model decoder was not found."
                )
        else:
            processor_or_tup = Wav2Vec2Processor.from_pretrained(
                model_path, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
            )
            assert not isinstance(processor_or_tup, tuple)
            processor = processor_or_tup

        model_or_tup = Wav2Vec2ForCTC.from_pretrained(
            model_path, token=os.getenv("HUGGINGFACE_HUB_TOKEN", True)
        )
        assert isinstance(model_or_tup, Wav2Vec2ForCTC)
        model = model_or_tup

        data_collator = DataCollatorCTCWithPadding(
            processor=processor,
            max_seconds_per_example=self.config.max_seconds_per_example,
            padding=self.config.padding,
        )

        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=data_collator,
            compute_metrics=partial(compute_wer_metrics, processor=processor),
        )


def dump_vocabulary(config: DictConfig) -> None:
    """Extracts the vocabulary from the dataset and dumps it to a file.

    It will dump the file to `${config.model_dir}/vocab.json`.

    Args:
        config:
            The Hydra configuration object.
    """
    # Build the set of all unique characters in the dataset
    unique_characters: set[str] = set(config.characters_to_keep + "|")
    sorted_unique_characters: list[str] = sorted(unique_characters)

    # Build vocabulary
    vocab = {char: idx for idx, char in enumerate(sorted_unique_characters)}

    # Dump the vocabulary to a json file
    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocab, f)
