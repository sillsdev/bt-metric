"""How often a metric agrees with a gold judgment about which of two outputs is better."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


class PairwiseAgreement:
    """Agreement between gold and metric score differences over a set of compared pairs.

    Each pair is one comparison (system A vs. B, or segment i vs. j) given as the gold difference A − B
    and the metric difference A − B.
    """

    def __init__(self, gold_deltas: Sequence[float], metric_deltas: Sequence[float]) -> None:
        self._gold = np.asarray(gold_deltas, dtype=float)
        self._metric = np.asarray(metric_deltas, dtype=float)
        if self._gold.shape != self._metric.shape:
            raise ValueError("Gold and metric deltas must have the same length.")

    @classmethod
    def all_pairs(cls, gold_scores: Sequence[float], metric_scores: Sequence[float]) -> "PairwiseAgreement":
        """Every pair among the items scored by both the gold standard and the metric."""
        gold = np.asarray(gold_scores, dtype=float)
        metric = np.asarray(metric_scores, dtype=float)
        i, j = np.triu_indices(len(gold), k=1)
        return cls(gold[i] - gold[j], metric[i] - metric[j])

    def accuracy(self) -> float:
        """Pairwise accuracy (Kocmi et al. 2021): pairs with a gold tie are skipped, metric ties count as wrong."""
        decided = self._gold != 0
        if not decided.any():
            return float("nan")
        return float(np.mean(np.sign(self._gold[decided]) == np.sign(self._metric[decided])))

    def accuracy_with_ties(self, epsilon: float = 0.0) -> float:
        """Pairwise accuracy with tie calibration (acc_eq, Deutsch et al. 2023).

        A metric difference within epsilon counts as a tie, and a tie agrees only with a gold tie.
        """
        if len(self._gold) == 0:
            return float("nan")
        metric_signs = np.where(np.abs(self._metric) <= epsilon, 0.0, np.sign(self._metric))
        return float(np.mean(np.sign(self._gold) == metric_signs))

    def best_tie_epsilon(self, candidates: Sequence[float]) -> Tuple[float, float]:
        """The epsilon among the candidates that maximizes acc_eq, and the accuracy it gives."""
        scored = [(self.accuracy_with_ties(epsilon), epsilon) for epsilon in candidates]
        accuracy, epsilon = max(scored)
        return epsilon, accuracy

    def bootstrap_accuracy(self, samples: int = 1000, confidence: float = 0.95, seed: int = 0) -> Interval:
        """Percentile bootstrap interval for accuracy(), resampling the compared pairs."""
        rng = np.random.default_rng(seed)
        n = len(self._gold)
        estimates = [
            PairwiseAgreement(self._gold[indices], self._metric[indices]).accuracy()
            for indices in (rng.integers(0, n, n) for _ in range(samples))
        ]
        tail = (1 - confidence) / 2 * 100
        lower, upper = np.nanpercentile(estimates, [tail, 100 - tail])
        return Interval(self.accuracy(), float(lower), float(upper))
