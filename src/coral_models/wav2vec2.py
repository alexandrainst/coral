"""Model setup for Wav2Vec 2.0 models."""

import json
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Type

import torch
from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    BatchEncoding,
    BatchFeature,
    EvalPrediction,
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
from wandb.sdk.wandb_init import init as wandb_init

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
    padding: bool | str = True
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
        audio_features = [
            dict(input_values=feature["input_values"]) for feature in features
        ]
        batch: BatchFeature = self.processor.pad(
            audio_features, padding=self.padding, return_tensors="pt"
        )

        with self.processor.as_target_processor():
            label_features = [dict(input_ids=feature["labels"]) for feature in features]
            labels_batch: BatchEncoding = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
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
        dump_vocabulary(self.cfg)
        tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            self.cfg.model_dir,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            word_delimiter_token="|",
        )

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
                mask_time_prob=self.cfg.model.mask_time_prob,
                mask_feature_prob=self.cfg.model.mask_feature_prob,
                mask_feature_length=self.cfg.model.mask_feature_length,
                layerdrop=self.cfg.model.layerdrop,
                ctc_loss_reduction=self.cfg.model.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                vocab_size=len(self.processor.tokenizer.get_vocab()),
            )
            assert isinstance(model, Wav2Vec2ForCTC)

        if self.cfg.model.freeze_feature_encoder:
            for param in model.wav2vec2.parameters():
                param.requires_grad = False

        return model

    def load_data_collator(self) -> DataCollatorCTCWithPadding:
        return DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def load_trainer_class(self) -> Type[Trainer]:
        return Trainer

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        return partial(compute_wer_metrics, processor=self.processor)

    def load_training_arguments(self) -> TrainingArguments:
        if self.cfg.wandb:
            wandb_init(project=self.cfg.pipeline_id, name=self.cfg.wandb_name)

        args = TrainingArguments(
            output_dir=self.cfg.model_dir,
            hub_model_id=self.cfg.hub_id,
            per_device_train_batch_size=self.cfg.model.batch_size,
            per_device_eval_batch_size=self.cfg.model.batch_size,
            gradient_accumulation_steps=self.cfg.model.gradient_accumulation,
            learning_rate=self.cfg.model.learning_rate,
            warmup_steps=self.cfg.model.warmup_steps,
            max_steps=self.cfg.model.max_steps,
            fp16=self.cfg.model.fp16 and not mps_is_available(),
            push_to_hub=self.cfg.push_to_hub,
            evaluation_strategy="steps",
            eval_steps=self.cfg.eval_steps,
            save_steps=self.cfg.save_steps,
            logging_steps=self.cfg.logging_steps,
            length_column_name="input_length",
            gradient_checkpointing=True,
            save_total_limit=self.cfg.save_total_limit,
            load_best_model_at_end=self.cfg.model.early_stopping,
            metric_for_best_model="wer",
            greater_is_better=False,
            seed=self.cfg.seed,
            remove_unused_columns=False,
            optim=OptimizerNames.ADAMW_TORCH,
            use_mps_device=mps_is_available(),
            report_to=["wandb"] if self.cfg.wandb else [],
            ignore_data_skip=self.cfg.ignore_data_skip,
            save_safetensors=True,
            no_cuda=hasattr(sys, "_called_from_test"),
        )
        return args

    def load_saved(self) -> PreTrainedModelData:
        processor: Processor
        if self.cfg.model.language_model_decoder is not None:
            try:
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                    self.cfg.hub_id, use_auth_token=True
                )
            except (FileNotFoundError, ValueError):
                processor = Wav2Vec2Processor.from_pretrained(
                    self.cfg.hub_id, use_auth_token=True
                )
        else:
            processor = Wav2Vec2Processor.from_pretrained(
                self.cfg.hub_id, use_auth_token=True
            )

        model = Wav2Vec2ForCTC.from_pretrained(self.cfg.hub_id, use_auth_token=True)
        data_collator = DataCollatorCTCWithPadding(
            processor=processor, padding="longest"
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
    unique_characters: set[str] = set(cfg.model.characters_to_keep)

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