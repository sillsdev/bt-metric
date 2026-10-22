"""Meta-evaluates features against the human judgments in the downloaded datasets."""

import argparse
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ..evaluation.data import JudgedOutputs, SegmentJudgments
from ..evaluation.loaders import AfriMteLoader, WmtDaLoader, WmtMqmLoader
from ..features.base import Feature
from ..features.chrf import ChrF
from .agreement import PairwiseAgreement

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaEvalResult:
    dataset: str
    level: str
    statistic: str
    feature: str
    value: float
    lower: Optional[float]
    upper: Optional[float]
    comparisons: int


class SystemLevelEvaluator:
    """Pairwise accuracy of system rankings, per evaluation set and pooled over all of them."""

    def __init__(self, exclude_systems: str) -> None:
        # Human translations scored as systems (e.g. "refB", "Human-A") are excluded from ranking by default.
        self._exclude = re.compile(exclude_systems) if exclude_systems else None

    def evaluate(self, outputs_list: List[JudgedOutputs], feature: Feature, dataset: str) -> List[MetaEvalResult]:
        gold_deltas: List[np.ndarray] = []
        metric_deltas: List[np.ndarray] = []
        results = []
        for outputs in outputs_list:
            outputs = outputs.without_systems([s for s in outputs.systems() if self._is_excluded(s)])
            evaluation_set = outputs.evaluation_set
            systems = [system for system in outputs.systems() if system in outputs.human_scores]
            gold = [outputs.human_system_score(system) for system in systems]
            metric = [
                feature.system_score(evaluation_set.sources, outputs.hypotheses[system], evaluation_set.references)
                for system in systems
            ]
            agreement = PairwiseAgreement.all_pairs(gold, metric)
            i, j = np.triu_indices(len(systems), k=1)
            gold_deltas.append(np.asarray(gold)[i] - np.asarray(gold)[j])
            metric_deltas.append(np.asarray(metric)[i] - np.asarray(metric)[j])
            results.append(
                MetaEvalResult(
                    evaluation_set.name,
                    "system",
                    "pairwise_accuracy",
                    feature.name,
                    agreement.accuracy(),
                    None,
                    None,
                    len(i),
                )
            )
        pooled = PairwiseAgreement(np.concatenate(gold_deltas), np.concatenate(metric_deltas))
        interval = pooled.bootstrap_accuracy()
        results.append(
            MetaEvalResult(
                f"{dataset} (pooled)",
                "system",
                "pairwise_accuracy",
                feature.name,
                interval.estimate,
                interval.lower,
                interval.upper,
                sum(len(d) for d in gold_deltas),
            )
        )
        return results

    def _is_excluded(self, system: str) -> bool:
        return bool(self._exclude and self._exclude.search(system))


class SegmentLevelEvaluator:
    """Segment-level agreement: acc_eq over hypothesis pairs of the same item, or global Kendall's tau-b.

    Comparing hypotheses of the same source (WMT's "item-grouped" setting) is used when items have several
    hypotheses; single-hypothesis sets such as AfriMTE only allow a correlation across all segments.
    """

    def __init__(self, max_items: int, tie_epsilons: List[float]) -> None:
        self._max_items = max_items
        self._tie_epsilons = tie_epsilons

    def evaluate(self, judgments: SegmentJudgments, feature: Feature) -> MetaEvalResult:
        judgments = judgments.sample_items(self._max_items)
        metric = feature.segment_scores(
            judgments.column("source"), judgments.column("hypothesis"), judgments.column("reference")
        )
        human = judgments.human()
        gold_deltas, metric_deltas = self._item_pairs(judgments.items(), human, metric)
        if len(gold_deltas) > 0:
            epsilon, accuracy = PairwiseAgreement(gold_deltas, metric_deltas).best_tie_epsilon(self._tie_epsilons)
            LOGGER.debug(f"{judgments.name} {feature.name}: tie epsilon {epsilon}")
            return MetaEvalResult(
                judgments.name, "segment", "acc_eq", feature.name, accuracy, None, None, len(gold_deltas)
            )
        tau = kendalltau(human, metric).statistic
        return MetaEvalResult(
            judgments.name, "segment", "kendall_tau_b", feature.name, float(tau), None, None, len(human)
        )

    def _item_pairs(self, items: np.ndarray, human: np.ndarray, metric: np.ndarray):
        gold_deltas, metric_deltas = [], []
        for indices in pd.Series(np.arange(len(items))).groupby(items).groups.values():
            indices = np.asarray(indices)
            if len(indices) < 2:
                continue
            i, j = np.triu_indices(len(indices), k=1)
            gold_deltas.append(human[indices[i]] - human[indices[j]])
            metric_deltas.append(metric[indices[i]] - metric[indices[j]])
        if not gold_deltas:
            return np.array([]), np.array([])
        return np.concatenate(gold_deltas), np.concatenate(metric_deltas)


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: Dict[str, Feature] = {feature.name.lower(): feature for feature in (ChrF(0), ChrF(2))}

    def names(self) -> List[str]:
        return list(self._features)

    def get(self, name: str) -> Feature:
        return self._features[name.lower()]


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Meta-evaluate features against human judgments")
    parser.add_argument("--features", nargs="+", default=registry.names(), choices=registry.names())
    parser.add_argument("--data-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--output", type=Path, default=Path("data/results/meta_eval.tsv"))
    parser.add_argument("--max-items", type=int, default=2000, help="Items sampled per set for segment level")
    parser.add_argument("--exclude-systems", default=r"^(ref|Human)", help="Regex of systems not to rank")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features = [registry.get(name) for name in args.features]
    system_level = SystemLevelEvaluator(args.exclude_systems)
    segment_level = SegmentLevelEvaluator(args.max_items, tie_epsilons=[0.0, 0.5, 1.0, 2.0, 4.0])
    results: List[MetaEvalResult] = []

    mqm = list(WmtMqmLoader(args.data_dir / "wmt-mqm" / "wmt-mqm-human-evaluation.csv").judged_outputs())
    da = list(WmtDaLoader(args.data_dir / "wmt-da" / "wmt-da-human-evaluation.csv").segment_judgments())
    afrimte = list(AfriMteLoader(args.data_dir / "afrimte" / "AfriMTE-ade-devtest-v2-with-score").segment_judgments())
    for feature in features:
        LOGGER.info(f"Meta-evaluating {feature.name}")
        results.extend(system_level.evaluate(mqm, feature, "wmt-mqm"))
        for judgments in [outputs.segment_judgments() for outputs in mqm] + da + afrimte:
            results.append(segment_level.evaluate(judgments, feature))

    table = pd.DataFrame([asdict(result) for result in results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    pooled = table[table["dataset"].str.endswith("(pooled)")]
    LOGGER.info(f"System-level pairwise accuracy:\n{pooled.to_string(index=False)}")
    segment = table[table["level"] == "segment"]
    collection = segment["dataset"].str.extract(r"^(wmt\d+-\w+|afrimte)", expand=False)
    summary = segment.pivot_table(index=collection, columns=["statistic", "feature"], values="value")
    LOGGER.info(f"Segment-level agreement (mean over sets):\n{summary.round(3).to_string()}")
    LOGGER.info(f"All results -> {args.output}")


if __name__ == "__main__":
    main()
