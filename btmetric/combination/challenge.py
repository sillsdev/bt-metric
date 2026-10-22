"""Fits the combined metric on challenge-set row scores and checks it per perturbation, against each feature alone.

Every row compares an original with its perturbed copy, so the original is always the better side. Accuracy is
from leave-one-test-set-out predictions: the weights used for a test set never saw that set's language.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .model import PairwiseLogLinear, PerGroupScale, leave_one_group_out

LOGGER = logging.getLogger(__name__)


class ChallengeRowScores:
    def __init__(self, table: pd.DataFrame) -> None:
        self._table = table

    @classmethod
    def load(cls, path: Path) -> "ChallengeRowScores":
        return cls(pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""]))

    def features(self) -> List[str]:
        return sorted({column.split("|")[0] for column in self._table.columns if column.endswith("|original")})

    def deltas(self, features: List[str]) -> pd.DataFrame:
        return pd.DataFrame({f: self._table[f"{f}|original"] - self._table[f"{f}|perturbed"] for f in features}).astype(
            float
        )

    def values(self, features: List[str]) -> pd.DataFrame:
        """Both sides' scores stacked, for estimating each feature's spread per test set."""
        original = self._table[[f"{f}|original" for f in features]].set_axis(features, axis=1)
        perturbed = self._table[[f"{f}|perturbed" for f in features]].set_axis(features, axis=1)
        return pd.concat([original, perturbed], ignore_index=True).astype(float)

    def groups(self) -> np.ndarray:
        return self._table["test_hash"].to_numpy()

    def perturbations(self) -> np.ndarray:
        return self._table["perturbation"].to_numpy()


def accuracy_by_perturbation(scores: np.ndarray, perturbations: np.ndarray) -> pd.Series:
    """Share of rows where the original scores strictly higher, as in the per-feature challenge evaluation."""
    return pd.Series(scores > 0).groupby(perturbations).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the combined metric on challenge-set row scores")
    parser.add_argument("--rows", type=Path, default=Path("data/results/challenge_set_rows.tsv"))
    parser.add_argument("--features", nargs="+", help="Default: every feature in the file")
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("data/results/combination_challenge.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = ChallengeRowScores.load(args.rows)
    features = args.features or rows.features()
    groups, perturbations = rows.groups(), rows.perturbations()
    scale = PerGroupScale().fit(rows.values(features), np.concatenate([groups, groups]), features)
    deltas = scale.apply(rows.deltas(features), groups, features)
    labels = np.ones(len(deltas))
    # With every pair oriented original-first, half are flipped so the model also sees perturbed-first pairs.
    flip = np.random.default_rng(0).random(len(labels)) < 0.5
    deltas[flip] *= -1
    labels[flip] = -1

    out_of_group = leave_one_group_out(deltas, labels, groups, args.l2) * labels
    table = pd.DataFrame({"combined": accuracy_by_perturbation(out_of_group, perturbations)})
    raw = rows.deltas(features)
    for feature in features:
        # Each feature alone is judged only on the rows it could score, as in btmetric-evaluate-challenge-set.
        scored = raw[feature].notna().to_numpy()
        table[feature] = accuracy_by_perturbation(raw[feature].to_numpy()[scored], perturbations[scored])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t")
    shares = PairwiseLogLinear(args.l2).fit(deltas, labels).weight_shares()
    LOGGER.info(f"Weights fit on all test sets: {dict(zip(features, shares.round(3)))}")
    LOGGER.info(
        "Share of rows where the original scores higher (combined: leave one test set out):\n"
        f"{table.round(3).to_string()}\nMean over perturbations:\n{table.mean().round(3).to_string()}"
    )


if __name__ == "__main__":
    main()
