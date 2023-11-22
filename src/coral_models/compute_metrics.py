"""Function used to compute metrics during ASR training of Wave2Vec 2.0 models."""

import numpy as np
from evaluate.loading import load as load_metric
from numpy.typing import NDArray
from transformers import EvalPrediction, PreTrainedTokenizerBase
import logging

from .protocols import Processor

logger = logging.getLogger(__name__)


def compute_wer_metrics(pred: EvalPrediction, processor: Processor) -> dict[str, float]:
    """Compute the word error rate of predictions.

    Args:
        pred (EvalPrediction):
            Prediction output of the speech recognition model.
        processor (Processor):
            Audio and transcription processor.

    Returns:
        dict:
            dictionary with 'wer' as the key and the word error rate as the value.
    """
    wer_metric = load_metric("wer")
    tokenizer: PreTrainedTokenizerBase = getattr(processor, "tokenizer")
    pad_token = tokenizer.pad_token_id

    # Shape: [batch_size, seq_len, vocab_size] or [batch_size, seq_len]
    predictions: NDArray[np.int_] | NDArray[np.float_] = pred.predictions

    if len(predictions.shape) == 3:
        # Decode the predictions to get the transcriptions. When a language model is
        # attached to the processor then we get the predicted string directly from the
        # logits. If the vocabulary dimension of the predictions is too small then we
        # pad with zeros to match the size of the vocabulary
        if predictions.dtype == np.int_:
            vocab_size = tokenizer.get_vocab()
            mismatch_dim = len(vocab_size) - predictions.shape[-1]
            predictions = np.pad(predictions, ((0, 0), (0, 0), (0, mismatch_dim)))
            predictions_str = tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )

        # Otherwise, if we are not using a language model, we need to convert the
        # logits to token IDs and then decode the token IDs to get the predicted string
        else:
            pred_ids: NDArray[np.int_] = np.argmax(predictions, axis=-1)
            predictions_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    elif len(predictions.shape) == 2 and predictions.dtype == np.int_:
        predictions_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    else:
        raise ValueError(
            f"Predictions have an unexpected shape {predictions.shape} and dtype "
            f"{predictions.dtype}."
        )

    # Set the ground truth labels with label id -100 to be the padding token id. This
    # ensures that the WER metric does not consider these labels in its computation.
    labels = pred.label_ids
    assert isinstance(labels, np.ndarray)
    labels[labels == -100] = pad_token

    # Decode the ground truth labels
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # TEMP: Log both the predictions and the ground truth labels
    random_idx = np.random.randint(0, len(predictions_str))
    logger.info(f"Sample document: {labels_str[random_idx]}")
    logger.info(f"Predicted: {predictions_str[random_idx]}")

    # Compute the word error rate
    computed = wer_metric.compute(predictions=predictions_str, references=labels_str)
    assert computed is not None

    # Ensure that `wer` is a dict, as metrics in the `evaluate` library can either be
    # dicts or floats
    if not isinstance(computed, dict):
        return dict(wer=computed)
    else:
        return computed
