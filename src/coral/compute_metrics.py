"""Function used to compute metrics during ASR training of Wav2Vec 2.0 models."""

import logging
import os

import numpy as np
from evaluate.loading import load as load_metric
from numpy.typing import NDArray
from transformers import (
    EvalPrediction,
    PreTrainedTokenizerBase,
    Wav2Vec2ProcessorWithLM,
)

from .data_models import Processor

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

    # If all the logits are -100 for a token, then we set the logit for the padding
    # token for that token to 0. This is to ensure that this token gets decoded to a
    # padding token, and are therefore ignored
    predictions[np.all(predictions == -100, axis=-1), pad_token] = 0

    # Decode the predictions to get the transcriptions. When a language model is
    # attached to the processor then we get the predicted string directly from the
    # logits. If the vocabulary dimension of the predictions is too small then we pad
    # with zeros to match the size of the vocabulary
    if isinstance(processor, Wav2Vec2ProcessorWithLM):
        vocab_size = tokenizer.get_vocab()
        mismatch_dim = len(vocab_size) - predictions.shape[-1]
        predictions = np.pad(
            array=predictions,
            pad_width=((0, 0), (0, 0), (0, mismatch_dim)),
            mode="constant",
            constant_values=pad_token,
        )
        predictions_str = processor.batch_decode(predictions).text

    # Otherwise, if we are not using a language model, we need to convert the logits to
    # token IDs and then decode the token IDs to get the predicted string
    else:
        pred_ids: NDArray[np.int_] = np.argmax(predictions, axis=-1)
        predictions_str = processor.batch_decode(pred_ids)

    # Set the ground truth labels with label id -100 to be the padding token id. This
    # ensures that the WER metric does not consider these labels in its computation.
    labels = pred.label_ids
    assert isinstance(labels, np.ndarray)
    labels[labels == -100] = pad_token

    # Decode the ground truth labels
    labels_str = tokenizer.batch_decode(sequences=labels, group_tokens=False)

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
