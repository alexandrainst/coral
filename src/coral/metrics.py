"""Metrics for automatic speech recognition."""

import collections.abc as c

import jiwer


def cer(
    predictions: c.Iterable[str], labels: c.Iterable[str], normalise: bool = True
) -> float:
    """Compute the character error rate.

    Args:
        predictions:
            The model predictions.
        labels:
            The ground-truth labels.
        normalise (optional):
            Whether to normalise the error rate to ensure that it is always been 0% and
            100%. Defaults to True.

    Returns:
        The aggregated character error rate.
    """
    incorrect: int = 0
    total: int = 0
    for prediction, label in zip(predictions, labels):
        measures = jiwer.process_characters(reference=label, hypothesis=prediction)
        incorrect += measures.substitutions + measures.deletions + measures.insertions
        total += measures.substitutions + measures.deletions + measures.hits
        if normalise:
            total += measures.insertions
    return incorrect / total


def wer(
    predictions: c.Iterable[str], labels: c.Iterable[str], normalise: bool = True
) -> float:
    """Compute the word error rate.

    Args:
        predictions:
            The model predictions.
        labels:
            The ground-truth labels.
        normalise (optional):
            Whether to normalise the error rate to ensure that it is always been 0% and
            100%. Defaults to True.

    Returns:
        The aggregated word error rate.
    """
    incorrect: int = 0
    total: int = 0
    for prediction, label in zip(predictions, labels):
        measures = jiwer.process_words(reference=label, hypothesis=prediction)
        incorrect += measures.substitutions + measures.deletions + measures.insertions
        total += measures.substitutions + measures.deletions + measures.hits
        if normalise:
            total += measures.insertions
    return incorrect / total
