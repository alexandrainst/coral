"""Function used to compute metrics during ASR training of Wave2Vec 2.0 models."""

import numpy as np
from evaluate.loading import load as load_metric
from numpy.typing import NDArray
from transformers import EvalPrediction, ProcessorMixin, Wav2Vec2ProcessorWithLM


def compute_wer_metrics(
    pred: EvalPrediction, processor: ProcessorMixin
) -> dict[str, float]:
    """Compute the word error rate of predictions.

    Args:
        pred (EvalPrediction):
            Prediction output of the speech recognition model.
        processor (ProcessorMixin):
            Audio and transcription processor.

    Returns:
        dict:
            dictionary with 'wer' as the key and the word error rate as the
            value.
    """
    wer_metric = load_metric("wer")
    pad_token = processor.tokenizer.pad_token_id

    pred_logits = pred.predictions
    assert not isinstance(pred_logits, tuple)

    # In cases where the prediction logits are -100 for all characters, convert the
    # logits for the padding character to be 100, to make sure that padding is chosen
    pad_arr = pred_logits[:, :, pad_token]
    pred_logits[:, :, pad_token] = np.where(pad_arr == -100, 100, pad_arr)

    # Decode the predictions to get the transcriptions. When a language model is
    # attached to the processor then we get the predicted string directly from the
    # logits
    if isinstance(processor, Wav2Vec2ProcessorWithLM):
        try:
            pred_str: str = processor.batch_decode(pred_logits).text

        # This error occurs when there are too few logits compared to the size of the
        # vocabulary. We fix this by simply adding zero logits padding to match the
        # size of the vocabulary.
        except ValueError:
            vocab_size = processor.tokenizer.get_vocab()
            mismatch_dim = len(vocab_size) - pred_logits.shape[-1]
            pred_logits = np.pad(pred_logits, ((0, 0), (0, 0), (0, mismatch_dim)))
            pred_str = processor.batch_decode(pred_logits).text

    # Otherwise, if we are not using a language model, we need to convert the logits to
    # token IDs and then decode the token IDs to get the predicted string
    else:
        pred_ids: NDArray[np.int_] = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)

    # Set the ground truth labels with label id -100 to be the padding token id. This
    # ensures that the WER metric does not consider these labels in its computation.
    labels = pred.label_ids
    assert isinstance(labels, np.ndarray)
    labels[labels == -100] = pad_token

    # Decode the ground truth labels
    label_str = processor.tokenizer.batch_decode(labels, group_tokens=False)

    # Compute the word error rate
    computed = wer_metric.compute(predictions=pred_str, references=label_str)
    assert computed is not None

    # Ensure that `wer` is a dict, as metrics in the `evaluate` library can either be
    # dicts or floats
    if not isinstance(computed, dict):
        return dict(wer=computed)
    else:
        return computed
