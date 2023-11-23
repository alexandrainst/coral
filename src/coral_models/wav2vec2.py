"""Model setup for Wav2Vec 2.0 models."""

import json
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Type
import time
import os

import torch
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    BatchEncoding,
    BatchFeature,
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
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer import OptimizerNames

from .compute_metrics import compute_wer_metrics
from .protocols import PreTrainedModelData, Processor
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


@dataclass
class DataCollatorCTCWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor (Wav2Vec2Processor)
            The processor used for proccessing the data.
        padding (bool, str or PaddingStrategy, optional):
            Select a strategy to pad the returned sequences (according to the model's
            padding side and padding index) among:
            * True or 'longest':
                Pad to the longest sequence in the batch (or no padding if only a
                single sequence if provided).
            * 'max_length':
                Pad to a maximum length specified with the argument max_length or to
                the maximum acceptable input length for the model if that argument is
                not provided.
            * False or 'do_not_pad':
                No padding (i.e., can output a batch with sequences of different
                lengths).
            Defaults to True.
    """

    processor: Wav2Vec2Processor
    padding: bool | str
    return_tensors: str = "pt"

    def torch_call(self, features: list[dict]) -> BatchFeature:
        """Collate the features.

        Args:
            features (list of dict):
                A list of feature dicts.

        Returns:
            BatchFeature:
                A dictionary of the collated features.
        """
        if "input_values" in features[0]:
            audio_features = [dict(input_values=f["input_values"]) for f in features]
        elif "audio" in features[0]:
            audio_features = [dict(audio=f["audio"]["array"]) for f in features]
        else:
            raise ValueError(
                "Features must contain either 'input_values' or 'audio' key."
            )
        batch: BatchFeature = self.processor.pad(
            audio_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=16_000 * 10,
        )

        label_features = [dict(input_ids=feature["labels"]) for feature in features]
        labels_batch: BatchEncoding = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
            max_length=512,
        )

        # Replace padding with -100 to ignore loss correctly
        non_one_entries: torch.Tensor = labels_batch.attention_mask.ne(1)
        labels: torch.Tensor = labels_batch.input_ids.masked_fill(non_one_entries, -100)

        batch["labels"] = labels
        return batch


class Wav2Vec2ModelSetup:
    """Model setup for Wav2Vec 2.0 models.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.processor: Wav2Vec2Processor

    def load_processor(self) -> Wav2Vec2Processor:
        # We dump the vocabulary to a file since the tokenizer uses this file during
        # initialisation
        while True:
            try:
                dump_vocabulary(self.cfg)
                tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                    self.cfg.model_dir,
                    unk_token="<unk>",
                    pad_token="<pad>",
                    bos_token="<s>",
                    eos_token="</s>",
                    word_delimiter_token=" ",
                )
                break
            except json.decoder.JSONDecodeError:
                process_id = os.getenv("RANK", 0)
                logger.warning(
                    f"JSONDecodeError while loading tokenizer on process {process_id}. "
                    "Retrying in a second."
                )
                time.sleep(1)

        # Set the `model_max_length` attribute of the tokenizer, if it hasn't been set,
        # to ensure that truncation is done correctly
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e6:
            tokenizer.model_max_length = 512

        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.cfg.model.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=extractor, tokenizer=tokenizer
        )
        return self.processor

    def load_model(self) -> Wav2Vec2ForCTC:
        with transformers_output_ignored():
            model = Wav2Vec2ForCTC.from_pretrained(
                self.cfg.model.pretrained_model_id,
                activation_dropout=self.cfg.model.activation_dropout,
                attention_dropout=self.cfg.model.attention_dropout,
                hidden_dropout=self.cfg.model.hidden_dropout,
                feat_proj_dropout=self.cfg.model.feat_proj_dropout,
                final_dropout=self.cfg.model.final_dropout,
                apply_spec_augment=True,
                mask_time_prob=self.cfg.model.mask_time_prob,
                mask_time_length=self.cfg.model.mask_time_length,
                mask_feature_prob=self.cfg.model.mask_feature_prob,
                mask_feature_length=self.cfg.model.mask_feature_length,
                layerdrop=self.cfg.model.layerdrop,
                ctc_loss_reduction=self.cfg.model.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                vocab_size=len(self.processor.tokenizer.get_vocab()),
                ctc_zero_infinity=True,
            )
            assert isinstance(model, Wav2Vec2ForCTC)

        if self.cfg.model.freeze_feature_encoder:
            for param in model.wav2vec2.parameters():
                param.requires_grad = False

        return model

    def load_data_collator(self) -> DataCollatorCTCWithPadding:
        return DataCollatorCTCWithPadding(
            processor=self.processor, padding=self.cfg.padding
        )

    def load_trainer_class(self) -> Type[Trainer]:
        return Trainer

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        return partial(compute_wer_metrics, processor=self.processor)

    def load_training_arguments(self) -> TrainingArguments:
        # Compute the gradient accumulation based on the total batch size in the config
        num_devices = max(torch.cuda.device_count(), 1)
        per_device_total_batch_size = self.cfg.total_batch_size // num_devices
        gradient_accumulation_steps = (
            per_device_total_batch_size // self.cfg.per_device_batch_size
        )

        if gradient_accumulation_steps == 0:
            logger.warning(
                f"Your `total_batch_size` is too small ({self.cfg.total_batch_size}), "
                f"relative to the number of devices ({num_devices}) and your "
                f"`per_device_batch_size` ({self.cfg.per_device_batch_size}). It has "
                f"been set to `per_device_batch_size * num_devices` = "
                f"{self.cfg.per_device_batch_size * num_devices}."
            )
            gradient_accumulation_steps = 1

        do_eval = any(
            [
                dataset_cfg.val_name is not None
                for dataset_cfg in self.cfg.datasets.values()
            ]
        )
        args = TrainingArguments(
            output_dir=self.cfg.model_dir,
            hub_model_id=self.cfg.hub_id,
            per_device_train_batch_size=self.cfg.per_device_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            lr_scheduler_type=SchedulerType.COSINE,
            warmup_steps=self.cfg.warmup_steps,
            max_steps=self.cfg.max_steps,
            fp16=self.cfg.fp16 and not mps_is_available(),
            push_to_hub=self.cfg.push_to_hub,
            evaluation_strategy="steps" if do_eval else "no",
            eval_steps=self.cfg.eval_steps if do_eval else None,
            save_steps=self.cfg.save_steps,
            save_strategy="no" if self.cfg.save_total_limit == 0 else "steps",
            logging_steps=self.cfg.logging_steps,
            length_column_name="input_length",
            gradient_checkpointing=True,
            save_total_limit=self.cfg.save_total_limit,
            load_best_model_at_end=self.cfg.early_stopping if do_eval else False,
            metric_for_best_model="wer" if do_eval else None,
            greater_is_better=False if do_eval else None,
            seed=self.cfg.seed,
            remove_unused_columns=False,
            optim=OptimizerNames.ADAMW_TORCH,
            adam_beta1=self.cfg.adam_first_momentum,
            adam_beta2=self.cfg.adam_second_momentum,
            report_to=["wandb"] if self.cfg.wandb else [],
            ignore_data_skip=self.cfg.ignore_data_skip,
            save_safetensors=True,
            use_cpu=hasattr(sys, "_called_from_test"),
            dataloader_num_workers=self.cfg.dataloader_num_workers,
            ddp_find_unused_parameters=False,
        )
        return args

    def load_saved(self) -> PreTrainedModelData:
        processor: Processor
        if self.cfg.model.language_model_decoder is not None:
            try:
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                    self.cfg.hub_id, token=True
                )
            except (FileNotFoundError, ValueError):
                processor = Wav2Vec2Processor.from_pretrained(
                    self.cfg.hub_id, token=True
                )
        else:
            processor = Wav2Vec2Processor.from_pretrained(self.cfg.hub_id, token=True)

        model = Wav2Vec2ForCTC.from_pretrained(self.cfg.hub_id, token=True)
        data_collator = DataCollatorCTCWithPadding(
            processor=processor, padding=self.cfg.padding
        )
        compute_metrics = partial(compute_wer_metrics, processor=processor)
        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )


def dump_vocabulary(cfg: DictConfig) -> None:
    """Extracts the vocabulary from the dataset and dumps it to a file.

    It will dump the file to `${cfg.model_dir}/vocab.json`.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """
    # Build the set of all unique characters in the dataset
    unique_characters: set[str] = set(cfg.characters_to_keep)

    # Build vocabulary
    vocab = {char: idx for idx, char in enumerate(unique_characters)}
    for tok in ["<unk>", "<pad>", "<s>", "</s>"]:
        vocab[tok] = len(vocab)

    # Dump the vocabulary to a json file
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = model_dir / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocab, f)
