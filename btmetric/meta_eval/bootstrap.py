"""Paired bootstrap resampling over verses, for whether one system really scores higher than another."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Maps per-verse statistics (one row per verse) to a corpus-level score. A mean suits metrics averaged over
# verses; corpus chrF needs its n-gram counts summed before the F-score is computed.
Aggregator = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class PairedBootstrapResult:
    delta: float
    lower: float
    upper: float
    p_value: float


class PairedBootstrap:
    """Paired bootstrap test (Koehn 2004) of the score difference A − B on the same test verses."""

    def __init__(self, aggregate: Aggregator, samples: int = 1000, confidence: float = 0.95, seed: int = 0) -> None:
        self._aggregate = aggregate
        self._samples = samples
        self._confidence = confidence
        self._seed = seed

    def compare(self, stats_a: np.ndarray, stats_b: np.ndarray) -> PairedBootstrapResult:
        stats_a = np.asarray(stats_a, dtype=float)
        stats_b = np.asarray(stats_b, dtype=float)
        if len(stats_a) != len(stats_b):
            raise ValueError("Both systems must be scored on the same verses.")
        rng = np.random.default_rng(self._seed)
        n = len(stats_a)
        deltas = np.empty(self._samples)
        for s in range(self._samples):
            # The same verses are drawn for both systems, which is what makes the test paired.
            indices = rng.integers(0, n, n)
            deltas[s] = self._aggregate(stats_a[indices]) - self._aggregate(stats_b[indices])
        tail = (1 - self._confidence) / 2 * 100
        lower, upper = np.percentile(deltas, [tail, 100 - tail])
        delta = self._aggregate(stats_a) - self._aggregate(stats_b)
        # One-sided: how often resampling erases or reverses the observed difference.
        p_value = float(np.mean(deltas <= 0)) if delta > 0 else float(np.mean(deltas >= 0))
        return PairedBootstrapResult(float(delta), float(lower), float(upper), p_value)
