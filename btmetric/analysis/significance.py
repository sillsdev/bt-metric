"""How large a metric difference has to be before humans reliably agree with it (PLAN.md 5; Kocmi et al. 2024).

For every pair of systems in the WMT MQM sets that human raters distinguished, the metric's system-level difference
is compared with the human preference. A logistic curve of P(humans agree with the metric's direction) against the
size of the difference then gives the difference needed for a chosen confidence.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..evaluation.loaders import WmtMqmLoader
from ..meta_eval.runner import FeatureRegistry

LOGGER = logging.getLogger(__name__)


class AgreementCurve:
    """P(humans agree with the metric's direction) = sigmoid(a + b * |difference|), fit by maximum likelihood."""

    def __init__(self) -> None:
        self.intercept = 0.0
        self.slope = 0.0

    def fit(self, differences: np.ndarray, agrees: np.ndarray) -> "AgreementCurve":
        x, y = np.abs(np.asarray(differences, dtype=float)), np.asarray(agrees, dtype=float)

        def negative_log_likelihood(params: np.ndarray) -> float:
            logits = params[0] + params[1] * x
            return float(np.sum(np.logaddexp(0.0, logits) - y * logits))

        result = minimize(negative_log_likelihood, np.array([0.0, 0.1]), method="L-BFGS-B")
        self.intercept, self.slope = result.x
        return self

    def probability(self, difference: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.intercept + self.slope * abs(difference)))))

    def difference_for(self, probability: float) -> float:
        """The smallest |difference| at which agreement reaches the probability; inf if it never does."""
        if self.slope <= 0:
            return float("inf")
        needed = (np.log(probability / (1 - probability)) - self.intercept) / self.slope
        return float(max(0.0, needed))


def system_pairs(outputs_list, feature, exclude: str = r"^(?:ref|Human)") -> pd.DataFrame:
    """Every pair of systems humans scored differently: the metric difference and whether its sign matches."""
    rows = []
    for outputs in outputs_list:
        evaluation_set = outputs.evaluation_set
        feature.prepare(evaluation_set.source_language, evaluation_set.target_language)
        systems = [s for s in outputs.systems() if not pd.Series([s]).str.match(exclude).iloc[0]]
        human = {s: outputs.human_system_score(s) for s in systems}
        metric = {
            s: feature.system_score(evaluation_set.sources, outputs.hypotheses[s], evaluation_set.references)
            for s in systems
        }
        for k, a in enumerate(systems):
            for b in systems[k + 1 :]:
                if human[a] == human[b]:
                    continue
                difference = metric[a] - metric[b]
                rows.append(
                    {
                        "dataset": evaluation_set.name,
                        "difference": difference,
                        "agrees": np.sign(difference) == np.sign(human[a] - human[b]),
                    }
                )
    return pd.DataFrame(rows)


def threshold_table(pairs: pd.DataFrame, probabilities: List[float]) -> dict:
    curve = AgreementCurve().fit(pairs["difference"].to_numpy(), pairs["agrees"].to_numpy())
    row = {"pairs": len(pairs), "agreement": float(pairs["agrees"].mean())}
    row.update({f"diff_for_{int(p * 100)}%": curve.difference_for(p) for p in probabilities})
    return row


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Metric differences needed for humans to agree, on WMT MQM")
    parser.add_argument("--mqm", type=Path, default=Path("data/external/wmt-mqm/wmt-mqm-human-evaluation.csv"))
    parser.add_argument("--features", nargs="+", default=["chrf3", "chrf3++", "m-chrf3", "bleu", "spbleu"])
    parser.add_argument("--probabilities", nargs="+", type=float, default=[0.7, 0.8, 0.9])
    parser.add_argument("--output", type=Path, default=Path("data/results/significance_thresholds.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    outputs_list = list(WmtMqmLoader(args.mqm).judged_outputs())
    rows, buckets = [], []
    for name in args.features:
        feature = registry.get(name)
        try:
            pairs = system_pairs(outputs_list, feature)
        finally:
            feature.close()
        rows.append({"feature": feature.name, **threshold_table(pairs, args.probabilities)})
        bins = pd.cut(pairs["difference"].abs(), [0, 0.5, 1, 2, 4, np.inf], include_lowest=True)
        bucket = pairs.groupby(bins, observed=True)["agrees"].agg(["mean", "size"])
        buckets.append(bucket.assign(feature=feature.name))
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(f"Metric difference needed for humans to agree with its direction:\n{table.round(2).to_string()}")
    LOGGER.info(f"Agreement by size of difference:\n{pd.concat(buckets).round(2).to_string()}")


if __name__ == "__main__":
    main()
