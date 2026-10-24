"""How large a metric difference has to be before humans reliably agree with it (PLAN.md 5; Kocmi et al. 2024).

For every pair of systems in the WMT MQM sets that human raters distinguished, the metric's system-level difference
is compared with the human preference. A logistic curve of P(humans agree with the metric's direction) against the
size of the difference then gives the difference needed for a chosen confidence.

For the Bible, where no human system rankings exist, the same is done on the silver table (PLAN.md 3.3c): pairs of
past silnlp systems on the same test set, with xCOMET-XL's preference in place of the humans'.
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


def bible_system_scores(table: pd.DataFrame, silver: str) -> pd.DataFrame:
    """Each system's score on its test set as users would see it: corpus chrF3, chrF-B (corpus chrF3 plus the
    coefficients times mean coverage and the mean key-term share, as ChrFB.system_score), and the mean silver label."""
    from sacrebleu.metrics import CHRF

    from ..features.chrf_b import COVERAGE_TARGET_COEFFICIENT, KEY_TERMS_COEFFICIENT

    chrf = CHRF(char_order=6, word_order=0, beta=3)
    rows = []
    for (test_hash, system), part in table.groupby(["test_hash", "system"]):
        corpus = chrf.corpus_score(part["hypothesis"].tolist(), [part["reference"].tolist()]).score
        coverage = part["coverage-target"].mean()
        key_terms = part["key-terms"].mean() if part["key-terms"].notna().any() else 0.0
        rows.append(
            {
                "dataset": test_hash,
                "system": system,
                "silver": part[silver].mean(),
                "chrF3": corpus,
                "chrF-B": corpus
                + COVERAGE_TARGET_COEFFICIENT * np.nan_to_num(coverage)
                + KEY_TERMS_COEFFICIENT * key_terms,
            }
        )
    return pd.DataFrame(rows)


def score_pairs(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Every pair of systems on a test set the silver label tells apart: the metric difference and whether its sign
    matches the silver preference."""
    rows = []
    for dataset, part in scores.groupby("dataset"):
        silver, values = part["silver"].to_numpy(), part[metric].to_numpy()
        i, j = np.triu_indices(len(part), k=1)
        decided = silver[i] != silver[j]
        difference = values[i][decided] - values[j][decided]
        agrees = np.sign(difference) == np.sign(silver[i][decided] - silver[j][decided])
        rows.append(pd.DataFrame({"dataset": dataset, "difference": difference, "agrees": agrees}))
    return pd.concat(rows, ignore_index=True)


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
    parser.add_argument(
        "--silver-rows",
        type=Path,
        help="Use the silver table's Bible system pairs (with chrF3 and chrF-B) instead of WMT MQM",
    )
    parser.add_argument("--silver", default="xCOMET-XL", help="With --silver-rows: the label column")
    parser.add_argument("--output", type=Path, default=Path("data/results/significance_thresholds.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.silver_rows:
        from .silver_agreement import SilverTable

        scores = bible_system_scores(SilverTable(args.silver_rows).load(), args.silver)
        pairs_by_feature = {metric: score_pairs(scores, metric) for metric in ["chrF3", "chrF-B"]}
    else:
        outputs_list = list(WmtMqmLoader(args.mqm).judged_outputs())
        pairs_by_feature = {}
        for name in args.features:
            feature = registry.get(name)
            try:
                pairs_by_feature[feature.name] = system_pairs(outputs_list, feature)
            finally:
                feature.close()
    rows, buckets = [], []
    for name, pairs in pairs_by_feature.items():
        rows.append({"feature": name, **threshold_table(pairs, args.probabilities)})
        bins = pd.cut(pairs["difference"].abs(), [0, 0.5, 1, 2, 4, np.inf], include_lowest=True)
        bucket = pairs.groupby(bins, observed=True)["agrees"].agg(["mean", "size"])
        buckets.append(bucket.assign(feature=name))
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(f"Metric difference needed for humans to agree with its direction:\n{table.round(2).to_string()}")
    LOGGER.info(f"Agreement by size of difference:\n{pd.concat(buckets).round(2).to_string()}")


if __name__ == "__main__":
    main()
