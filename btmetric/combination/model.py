"""The combined metric: a weighted sum of normalized features (PLAN.md 2.1, 4.1).

Weights are learned from compared pairs (an original and its perturbed copy, or two hypotheses a human ranked)
with a pairwise logistic ranking loss, constrained non-negative and L2-regularized.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PerGroupScale:
    """Scales each feature by its standard deviation within a group (a test set or language), so a weight means the
    same thing everywhere. Only the scale matters for pairwise differences; any per-group offset cancels."""

    def __init__(self) -> None:
        self._scales: Dict[str, Dict[str, float]] = {}

    def fit(self, values: pd.DataFrame, groups: Sequence[str], features: List[str]) -> "PerGroupScale":
        frame = values[features].assign(_group=list(groups))
        for group, rows in frame.groupby("_group"):
            self._scales[str(group)] = {
                feature: float(rows[feature].std()) if rows[feature].std() > 0 else 1.0 for feature in features
            }
        return self

    def apply(self, deltas: pd.DataFrame, groups: Sequence[str], features: List[str]) -> np.ndarray:
        scales = np.array([[self._scales.get(str(group), {}).get(f, 1.0) for f in features] for group in groups])
        scaled = deltas[features].to_numpy(dtype=float) / scales
        # A missing score is no evidence either way.
        return np.nan_to_num(scaled, nan=0.0)


class PairwiseLogLinear:
    """Weights w >= 0 minimizing mean log(1 + exp(-y * w . delta)) + l2 * |w|^2 over compared pairs.

    delta is the scaled feature difference A - B of a pair and y is +1 when A is the better side, -1 otherwise.
    """

    def __init__(self, l2: float = 0.01) -> None:
        self._l2 = l2
        self.weights = np.array([])

    def fit(self, deltas: np.ndarray, labels: np.ndarray) -> "PairwiseLogLinear":
        deltas = np.asarray(deltas, dtype=float)
        labels = np.asarray(labels, dtype=float)

        def loss_and_gradient(w: np.ndarray):
            margins = labels * (deltas @ w)
            loss = np.mean(np.logaddexp(0.0, -margins)) + self._l2 * w @ w
            # d/dw log(1 + exp(-m)) = -sigmoid(-m) * dm/dw
            weights_per_pair = -labels / (1.0 + np.exp(margins))
            gradient = deltas.T @ weights_per_pair / len(labels) + 2 * self._l2 * w
            return loss, gradient

        k = deltas.shape[1]
        result = minimize(loss_and_gradient, np.full(k, 1.0 / k), jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * k)
        self.weights = result.x
        return self

    def score(self, deltas: np.ndarray) -> np.ndarray:
        return np.asarray(deltas, dtype=float) @ self.weights

    def weight_shares(self) -> np.ndarray:
        """The weights normalized to sum to 1; the scale of a ranking metric doesn't matter."""
        total = self.weights.sum()
        return self.weights / total if total > 0 else self.weights


def leave_one_group_out(deltas: np.ndarray, labels: np.ndarray, groups: Sequence[str], l2: float = 0.01) -> np.ndarray:
    """Out-of-group predictions: each group's pairs are scored by a model fit on all other groups. This tests
    whether weights carry over to a language the model hasn't seen."""
    groups = np.asarray(groups)
    predictions = np.full(len(labels), np.nan)
    for group in np.unique(groups):
        held_out = groups == group
        model = PairwiseLogLinear(l2).fit(deltas[~held_out], labels[~held_out])
        predictions[held_out] = model.score(deltas[held_out])
    return predictions
