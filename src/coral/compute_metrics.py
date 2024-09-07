"""Function used to compute metrics during ASR training of Wav2Vec 2.0 models."""

import logging
import os
import re
from collections.abc import Iterable

import numpy as np
from datasets import Dataset
from evaluate.loading import load as load_metric
from numpy.typing import NDArray
from tqdm.auto import tqdm
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    EvalPrediction,
    PreTrainedTokenizerBase,
)
from transformers.pipelines.pt_utils import KeyDataset

from .data_models import Processor
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
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    tokenizer: PreTrainedTokenizerBase = getattr(processor, "tokenizer")
    pad_token = tokenizer.pad_token_id

    # Shape: [batch_size, seq_len, vocab_size] or [batch_size, seq_len]
    predictions: NDArray[np.number] = pred.predictions

    # Set the ground truth labels with label id -100 to be the padding token id. This
    # ensures that the WER metric does not consider these labels in its computation.
    labels = pred.label_ids
    assert isinstance(labels, np.ndarray)
    labels[labels == -100] = pad_token

    # Whisper decoding
    pred_ids: NDArray[np.int_]
    if predictions.ndim == 2:
        predictions_str = processor.batch_decode(predictions, skip_special_tokens=True)
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

    # Log both the predictions and the ground truth labels
    is_main_process = os.getenv("RANK", "0") == "0"
    if is_main_process and log_examples:
        random_idx = np.random.randint(0, len(predictions_str))
        logger.info(f"Sample document: {labels_str[random_idx]}")
        logger.info(f"Predicted: {predictions_str[random_idx]}")

    metrics: dict[str, float] = dict()

    # Compute the word error rate
    wer_computed = wer_metric.compute(
        predictions=predictions_str, references=labels_str
    )
    assert wer_computed is not None
    if not isinstance(wer_computed, dict):
        metrics = metrics | dict(wer=wer_computed)
    else:
        metrics = metrics | wer_computed

    # Compute the character error rate
    cer_computed = cer_metric.compute(
        predictions=predictions_str, references=labels_str
    )
    assert cer_computed is not None
    if not isinstance(cer_computed, dict):
        metrics = metrics | dict(cer=cer_computed)
    else:
        metrics = metrics | cer_computed

    return metrics


def compute_metrics_of_dataset_using_pipeline(
    dataset: Dataset,
    transcriber: AutomaticSpeechRecognitionPipeline,
    metric_names: list[str],
    characters_to_keep: Iterable[str],
    text_column: str,
    audio_column: str,
    batch_size: int,
) -> tuple[list[str], list[str], dict[str, list[float]]]:
    """Compute the metrics for the dataset using a pipeline.

    Args:
        dataset:
            The dataset to validate.
        transcriber:
            The transcriber used for transcribing the audio.
        metric_names:
            The names of the metrics to compute. Needs to be compatible with the name of
            the metric in the `evaluate` library.
        characters_to_keep:
            The characters to keep in the transcriptions.
        text_column:
            The name of the column containing the transcriptions.
        audio_column:
            The name of the column containing the audio samples.
        batch_size:
            The batch size to use for transcribing the audio.

    Returns:
        A triple (predictions, labels, all_scores) where:
            predictions:
                The transcriptions predicted by the model.
            labels:
                The ASR-processed ground-truth labels for each sample.
            all_scores:
                A dictionary containing the computed scores for each metric.
    """
    characters_to_keep = "".join(characters_to_keep)

    # This contains all the punctuation characters that will be removed from the
    # transcriptions, as they do not have an influence on the pronunciation of the
    # words.
    non_standard_characters_regex = re.compile(
        f"[^{re.escape(characters_to_keep + ' |')}]"
    )

    labels: list[str] = [lbl.strip().lower() for lbl in dataset[text_column]]
    predictions: list[str] = list()

    with (
        tqdm(total=len(dataset), desc="Transcribing") as pbar,
        transformers_output_ignored(),
    ):
        for out in transcriber(
            KeyDataset(dataset=dataset, key=audio_column), batch_size=batch_size
        ):
            prediction = re.sub(
                pattern=non_standard_characters_regex,
                repl="",
                string=out["text"].strip().lower(),
            )
            predictions.append(prediction.strip())
            pbar.update()

    all_scores: dict[str, list[float]] = dict()
    for metric_name in metric_names:
        metric = load_metric(metric_name)
        scores = [
            metric.compute(predictions=[pred], references=[ref])
            for pred, ref in zip(
                tqdm(predictions, desc=f"Computing {metric_name.upper()}s"), labels
            )
        ]

        # Ensure that the scores are indeed floats, as `compute` returns a dictionary
        # for some metrics
        scores = [score if isinstance(score, float) else -100.0 for score in scores]
        assert all(score >= 0 for score in scores), (
            f"The number of {metric_name.upper()}s should be equal to the number "
            f"of predictions - found {len(scores):,} {metric_name.upper()}s and "
            f"{len(predictions):,} predictions."
        )

        all_scores[metric_name] = scores

    return predictions, labels, all_scores
