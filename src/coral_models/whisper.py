"""Model setup for Whisper models."""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Type

from omegaconf import DictConfig
from torch.backends.mps import is_available as mps_is_available
from transformers import (
    BatchFeature,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer import OptimizerNames
from wandb.sdk.wandb_init import init as wandb_init

from .compute_metrics import compute_wer_metrics
from .protocols import PreTrainedModelData, Processor
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding(DataCollatorMixin):
    """Data collator that will dynamically pad the inputs received.

    Args:
        processor (WhisperProcessor)
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

    processor: WhisperProcessor
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
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods. First treat the audio inputs by simply returning
        # torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class WhisperModelSetup:
    """Model setup for Whisper models.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.processor: WhisperProcessor

    def load_processor(self) -> WhisperProcessor:
        # We dump the vocabulary to a file since the tokenizer uses this file during
        # initialisation
        tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(
            self.cfg.model.pretrained_model_id, language="Danish", task="transcribe"
        )

        # Set the `model_max_length` attribute of the tokenizer, if it hasn't been set,
        # to ensure that truncation is done correctly
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 1e6:
            tokenizer.model_max_length = 512

        extractor = WhisperFeatureExtractor.from_pretrained(
            self.cfg.model.pretrained_model_id
        )
        self.processor = WhisperProcessor(
            feature_extractor=extractor, tokenizer=tokenizer
        )
        return self.processor

    def load_model(self) -> WhisperForConditionalGeneration:
        with transformers_output_ignored():
            model = WhisperForConditionalGeneration.from_pretrained(
                self.cfg.model.pretrained_model_id,
                activation_dropout=self.cfg.model.activation_dropout,
                attention_dropout=self.cfg.model.attention_dropout,
                mask_time_prob=self.cfg.model.mask_time_prob,
                mask_feature_prob=self.cfg.model.mask_feature_prob,
                mask_feature_length=self.cfg.model.mask_feature_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            assert isinstance(model, WhisperForConditionalGeneration)

        if self.cfg.model.freeze_feature_encoder:
            for param in model.wav2vec2.parameters():
                param.requires_grad = False

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
        return DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor, padding=True
        )

    def load_trainer_class(self) -> Type[Trainer]:
        return Seq2SeqTrainer

    def load_compute_metrics(self) -> Callable[[EvalPrediction], dict]:
        return partial(compute_wer_metrics, processor=self.processor)

    def load_training_arguments(self) -> TrainingArguments:
        if self.cfg.wandb:
            wandb_init(project=self.cfg.pipeline_id, name=self.cfg.wandb_name)
        args = Seq2SeqTrainingArguments(
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
            predict_with_generate=True,
            generation_max_length=self.cfg.model.generation_max_length,
        )
        return args

    def load_saved(self) -> PreTrainedModelData:
        processor: Processor
        processor = WhisperProcessor.from_pretrained(
            self.cfg.hub_id, use_auth_token=True
        )

        model = WhisperForConditionalGeneration.from_pretrained(
            self.cfg.hub_id, use_auth_token=True
        )
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor, padding="longest"
        )
        compute_metrics = partial(compute_wer_metrics, processor=processor)
        return PreTrainedModelData(
            processor=processor,
            model=model,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
