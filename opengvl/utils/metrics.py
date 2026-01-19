from collections.abc import Sequence

import numpy as np


def spearman_dense_correlation(values: Sequence[float]) -> float:
    """Compute a Spearman-style correlation of `values` against their order.

    Uses dense ranks (argsort(argsort())) and returns NaN if undefined
    (fewer than 2 items or constant input).
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 2 or np.allclose(arr, arr[0]):
        return float("nan")
    ranks = arr.argsort().argsort().astype(float)
    target = np.arange(n, dtype=float)
    denom = np.std(ranks) * np.std(target)
    if denom == 0:
        return float("nan")
    return float(np.corrcoef(ranks, target)[0, 1])
