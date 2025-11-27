"""Metrics for automatic speech recognition."""

import collections.abc as c

import numpy as np


def cer(
    predictions: c.Iterable[str],
    labels: c.Iterable[str],
    normalise: bool = True,
    aggregator: c.Callable[[list[float]], float] = lambda x: np.median(x).item(),
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
        aggregator (optional):
            The function which aggregates all the sample error rate into a combined
            score. Defaults to the median.

    Returns:
        The aggregated character error rate.
    """
    raise NotImplementedError
