"""Baselines on pairs of real minority-language systems from past silnlp experiments.

There are no human judgments for these, so the questions are how often the baselines disagree on which system is
better, and how often a difference is significant at the test-set sizes these experiments use.
"""

import argparse
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from ..evaluation.data import JudgedOutputs
from ..evaluation.loaders import ExperimentOutputsLoader
from ..features.base import Feature
from ..meta_eval.runner import FeatureRegistry
from ..paths import EXPERIMENTS_DIR

LOGGER = logging.getLogger(__name__)


class OutputScores:
    """Segment and system scores of every output in one test set, per feature."""

    def __init__(self, outputs: JudgedOutputs, features: List[Feature]) -> None:
        evaluation_set = outputs.evaluation_set
        self.segment: Dict[str, Dict[str, np.ndarray]] = {}
        self.system: Dict[str, Dict[str, float]] = {}
        for feature in features:
            self.segment[feature.name] = {}
            self.system[feature.name] = {}
            for system, hypotheses in outputs.hypotheses.items():
                self.segment[feature.name][system] = feature.segment_scores(
                    evaluation_set.sources, hypotheses, evaluation_set.references
                )
                self.system[feature.name][system] = feature.system_score(
                    evaluation_set.sources, hypotheses, evaluation_set.references
                )


class SystemPairBaselines:
    def __init__(self, loader: ExperimentOutputsLoader, features: List[Feature], significance_feature: str) -> None:
        self._loader = loader
        self._features = features
        self._significance_feature = significance_feature

    def score(self, pairs: pd.DataFrame, workers: int) -> pd.DataFrame:
        groups = list(pairs.groupby("test_hash"))
        rows: List[dict] = []
        # Reading outputs from the mounted bucket is the slow part, so test sets load in parallel threads.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            loaded = executor.map(lambda item: self._load(item[0]), groups)
            for i, ((test_hash, group), outputs) in enumerate(zip(groups, loaded)):
                if outputs is None:
                    continue
                scores = OutputScores(outputs, self._features)
                rows.extend(self._pair_rows(group, scores))
                if (i + 1) % 100 == 0:
                    LOGGER.info(f"Scored {i + 1} of {len(groups)} test sets")
        return pd.DataFrame(rows)

    def _load(self, test_hash: str):
        try:
            return self._loader.outputs_for(test_hash)
        except (OSError, ValueError, KeyError) as e:
            LOGGER.warning(f"Skipping test set {test_hash}: {e}")
            return None

    def _pair_rows(self, group: pd.DataFrame, scores: OutputScores) -> List[dict]:
        rows = []
        segment = scores.segment[self._significance_feature]
        for pair in group.itertuples(index=False):
            if pair.system_a not in segment or pair.system_b not in segment:
                continue
            row = {"test_hash": pair.test_hash, "verses": pair.verses, "test_design": pair.test_design}
            row.update({"system_a": pair.system_a, "system_b": pair.system_b})
            for feature in self._features:
                system = scores.system[feature.name]
                row[f"delta_{feature.name}"] = system[pair.system_a] - system[pair.system_b]
            # A paired t-test on verse-level differences; for a mean over hundreds of verses it agrees closely
            # with the paired bootstrap and is far cheaper across tens of thousands of pairs.
            row["p_value"] = float(ttest_rel(segment[pair.system_a], segment[pair.system_b]).pvalue)
            rows.append(row)
        return rows


def summarize(table: pd.DataFrame, feature_names: List[str]) -> str:
    lines = []
    deltas = {name: np.sign(table[f"delta_{name}"]) for name in feature_names}
    agreement = pd.DataFrame(index=feature_names, columns=feature_names, dtype=float)
    for a, b in itertools.product(feature_names, repeat=2):
        agreement.loc[a, b] = float(np.mean(deltas[a] == deltas[b]))
    lines.append(f"Share of pairs on which two baselines pick the same winner:\n{agreement.round(3).to_string()}")
    table = table.assign(significant=table["p_value"] < 0.05)
    by_design = table.groupby("test_design").agg(pairs=("p_value", "size"), significant=("significant", "mean"))
    lines.append(f"Share of pairs whose verse-level difference is significant (p < 0.05):\n{by_design.round(3)}")
    sizes = pd.cut(table["verses"], [0, 100, 250, 500, 1000, 5000, np.inf])
    by_size = table.groupby(sizes, observed=True).agg(pairs=("p_value", "size"), significant=("significant", "mean"))
    lines.append(f"...by test-set size (verses):\n{by_size.round(3)}")
    return "\n\n".join(lines)


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Score past silnlp system pairs with the baselines")
    parser.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--index", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--pairs", type=Path, default=Path("data/system_pairs.tsv"))
    parser.add_argument("--features", nargs="+", default=["chrf3", "m-chrf3", "bleu", "spbleu"])
    parser.add_argument("--significance-feature", default="chrF3", help="Feature whose verse scores are tested")
    parser.add_argument("--relations", nargs="+", default=["config"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("data/results/system_pair_baselines.tsv"))
    parser.add_argument(
        "--resume", action="store_true", help="Score only test sets not already in --output, and append them"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = pd.read_csv(args.index, sep="\t", dtype={"step": str})
    pairs = pd.read_csv(args.pairs, sep="\t")
    usable = pairs[
        pairs["relation"].isin(args.relations) & ~pairs["near_identical"] & pairs["differing_fraction"].notna()
    ]
    previous = pd.DataFrame()
    if args.resume and args.output.is_file():
        previous = pd.read_csv(args.output, sep="\t")
        usable = usable[~usable["test_hash"].isin(previous["test_hash"])]
    LOGGER.info(f"{len(usable)} usable pairs in {usable['test_hash'].nunique()} test sets to score")
    features = [registry.get(name) for name in args.features]
    baselines = SystemPairBaselines(
        ExperimentOutputsLoader(args.experiments_root, index), features, args.significance_feature
    )
    table = pd.concat([previous, baselines.score(usable, args.workers)], ignore_index=True)
    for feature in features:
        feature.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(summarize(table, [feature.name for feature in features]))


if __name__ == "__main__":
    main()
