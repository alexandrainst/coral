"""Function used to compute metrics during ASR training of Wav2Vec 2.0 models."""

import logging
import os
from collections.abc import Iterable

import numpy as np
from datasets import Dataset
from numpy.typing import NDArray
from tqdm.auto import tqdm
from transformers import Wav2Vec2ProcessorWithLM
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction

from .data import DEFAULT_CONVERSION_DICT, process_example
from .data_models import Processor
from .metrics import cer, wer
from .utils import transformers_output_ignored

logger = logging.getLogger(__package__)


def compute_wer_metrics(
    pred: EvalPrediction, processor: Processor, log_examples: bool = True
) -> dict[str, float]:
    """Compute the word error rate of predictions.

    Args:
        pred:
            Prediction output of the speech recognition model.
        processor:
            Audio and transcription processor.
        log_examples:
            Whether to log examples of the predictions and the ground truth labels.

    Returns:
        dict:
            dictionary with 'wer' as the key and the word error rate as the value.
    """
    tokenizer: PreTrainedTokenizerBase = getattr(processor, "tokenizer")
    pad_token = tokenizer.pad_token_id

    # Shape: [batch_size, seq_len, vocab_size] or [batch_size, seq_len]
    predictions: NDArray[np.number] = pred.predictions  # type: ignore[assignment]

    # Set the ground truth labels with label id -100 to be the padding token id. This
    # ensures that the WER metric does not consider these labels in its computation.
    labels = pred.label_ids
    assert isinstance(labels, np.ndarray)
    labels[labels == -100] = pad_token

    # Whisper decoding
    pred_ids: NDArray[np.int_]
    if predictions.ndim == 2:
        if isinstance(processor, Wav2Vec2ProcessorWithLM):
            predictions_str = processor.batch_decode(predictions)
        else:
            predictions_str = processor.batch_decode(
                predictions, skip_special_tokens=True
            )
        labels_str = tokenizer.batch_decode(sequences=labels, skip_special_tokens=True)

    # Wav2Vec2 decoding
    elif predictions.ndim == 3:
        # If all the logits are -100 for a token, then we set the logit for the padding
        # token for that token to 0. This is to ensure that this token gets decoded to a
        # padding token, and are therefore ignored
        predictions[np.all(predictions == -100, axis=-1), pad_token] = 0

        pred_ids = np.argmax(predictions, axis=-1)
        predictions_str = processor.batch_decode(pred_ids)
        labels_str = tokenizer.batch_decode(sequences=labels, group_tokens=False)

    else:
        raise ValueError(
            f"Expected predictions to have either 2 or 3 dimensions, but found "
            f"{predictions.ndim} dimensions."
        )

    # Lower case and strip both the predictions and the ground truth labels, to ensure
    # that the WER and CER metrics are computed fairly
    predictions_str = [pred.lower().strip() for pred in predictions_str]
    labels_str = [lbl.lower().strip() for lbl in labels_str]

    # Log both the predictions and the ground truth labels
    is_main_process = os.getenv("RANK", "0") == "0"
    if is_main_process and log_examples:
        random_idx = np.random.randint(0, len(predictions_str))
        logger.info(f"Random sample document: {labels_str[random_idx]}")
        logger.info(f"Associated prediction: {predictions_str[random_idx]}")

    metrics: dict[str, float] = dict(
        cer=cer(predictions=predictions_str, labels=labels_str),
        wer=wer(predictions=predictions_str, labels=labels_str),
    )
    return metrics


def compute_metrics_of_dataset_using_pipeline(
    dataset: Dataset,
    transcriber: AutomaticSpeechRecognitionPipeline,
    characters_to_keep: Iterable[str],
    text_column: str,
    audio_column: str,
    batch_size: int,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Compute the metrics for the dataset using a pipeline.

    Args:
        dataset:
            The dataset to validate.
        transcriber:
            The transcriber used for transcribing the audio.
        characters_to_keep:
            The characters to keep in the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        audio_column:
            The name of the column containing the audio samples.
        batch_size:
            The batch size to use for transcribing the audio.

    Returns:
        A triple (predictions, labels, scores) where:
            predictions:
                The transcriptions predicted by the model.
            labels:
                The ASR-processed ground-truth labels for each sample.
            scores:
                A dictionary containing the computed scores for each metric.
    """
    characters_to_keep = "".join(characters_to_keep)

    labels: list[str] = [lbl.strip().lower() for lbl in dataset[text_column]]
    predictions: list[str] = list()

    dataset = dataset.shuffle(seed=42).select(range(100))

    with (
        tqdm(total=len(dataset), desc="Transcribing") as pbar,
        transformers_output_ignored(),
    ):
        for out in transcriber(
            KeyDataset(dataset=dataset, key=audio_column),  # type: ignore[arg-type]
            batch_size=batch_size,
            generate_kwargs=dict(language="danish", task="transcribe"),
        ):
            prediction = process_example(
                example=dict(text=out["text"]),
                characters_to_keep=characters_to_keep,
                conversion_dict=DEFAULT_CONVERSION_DICT,
                text_column="text",
                audio_column=None,
                lower_case=True,
                convert_numerals=True,
                processor=None,
            )["text"]
            predictions.append(prediction)
            pbar.update()

    scores = dict(
        cer=cer(predictions=predictions, labels=labels),
        wer=wer(predictions=predictions, labels=labels),
    )
    return predictions, labels, scores
