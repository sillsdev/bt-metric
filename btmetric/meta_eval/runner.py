"""Meta-evaluates features against the human judgments in the downloaded datasets."""

import argparse
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ..evaluation.data import JudgedOutputs, SegmentJudgments
from ..evaluation.loaders import AfriMteLoader, WmtDaLoader, WmtMqmLoader
from ..features.alignment import AlignerCache, AlignmentCoverage
from ..features.base import Feature
from ..features.chrf import ChrF
from ..features.comet import Comet
from ..features.key_terms import InferredRenderings, KeyTermAccuracy
from ..features.lexical import Bleu, MeanChrF, NegativeTer
from ..features.naturalness import PerplexityRatio
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
            feature.prepare(evaluation_set.source_language, evaluation_set.target_language)
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
        feature.prepare(judgments.source_language, judgments.target_language)
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


class ResultsFile:
    """Accumulates results across runs: rerunning a feature replaces its earlier rows."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def merge(self, new_results: pd.DataFrame) -> pd.DataFrame:
        table = new_results
        if self._path.is_file():
            previous = pd.read_csv(self._path, sep="\t")
            table = pd.concat([previous[~previous["feature"].isin(new_results["feature"])], new_results])
        self._path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(self._path, sep="\t", index=False)
        return table


class FeatureRegistry:
    """Features by name, built on first use: some load models or download tokenizers when constructed."""

    def __init__(self) -> None:
        # Shared so that related features train each language's models once between them.
        self._aligners = AlignerCache()
        self._renderings = InferredRenderings()
        self._factories: Dict[str, Callable[[], Feature]] = {
            "chrf3": lambda: ChrF(0),
            "chrf3++": lambda: ChrF(2),
            "m-chrf3": lambda: MeanChrF(0),
            "bleu": lambda: Bleu("13a"),
            "spbleu": lambda: Bleu("flores200"),
            "ter": NegativeTer,
            "comet22": lambda: Comet("COMET-22", "Unbabel/wmt22-comet-da"),
            "africomet": lambda: Comet("AfriCOMET", "masakhane/africomet-stl-1.1"),
            # Gated on Hugging Face. XXL is left out: COMET loads models in fp32, and 10.7B parameters need ~43 GB.
            "xcomet-xl": lambda: Comet("xCOMET-XL", "Unbabel/XCOMET-XL", batch_size=16),
            "coverage-source": lambda: AlignmentCoverage("source", aligners=self._aligners),
            "coverage-target": lambda: AlignmentCoverage("target", aligners=self._aligners),
            "key-terms": lambda: KeyTermAccuracy(renderings=self._renderings),
            "key-terms-names": lambda: KeyTermAccuracy(categories=frozenset({"PN"}), renderings=self._renderings),
            "perplexity": PerplexityRatio,
        }
        # Run only when asked for: neural features need a GPU; sacrebleu's TER is pure Python and takes over an hour
        # on the WMT sets while ranking almost exactly like chrF and BLEU; coverage trains an aligner per language
        # pair from Bible text on the mounted bucket.
        self._opt_in = {
            "ter",
            "comet22",
            "africomet",
            "xcomet-xl",
            "coverage-source",
            "coverage-target",
            "key-terms",
            "key-terms-names",
            "perplexity",
        }

    def names(self) -> List[str]:
        return list(self._factories)

    def default_names(self) -> List[str]:
        return [name for name in self._factories if name not in self._opt_in]

    def get(self, name: str) -> Feature:
        return self._factories[name.lower()]()


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Meta-evaluate features against human judgments")
    parser.add_argument("--features", nargs="+", default=registry.default_names(), choices=registry.names())
    parser.add_argument("--data-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--output", type=Path, default=Path("data/results/meta_eval.tsv"))
    parser.add_argument("--max-items", type=int, default=2000, help="Items sampled per set for segment level")
    parser.add_argument("--exclude-systems", default=r"^(ref|Human)", help="Regex of systems not to rank")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    system_level = SystemLevelEvaluator(args.exclude_systems)
    segment_level = SegmentLevelEvaluator(args.max_items, tie_epsilons=[0.0, 0.5, 1.0, 2.0, 4.0])
    results_file = ResultsFile(args.output)

    mqm = list(WmtMqmLoader(args.data_dir / "wmt-mqm" / "wmt-mqm-human-evaluation.csv").judged_outputs())
    da = list(WmtDaLoader(args.data_dir / "wmt-da" / "wmt-da-human-evaluation.csv").segment_judgments())
    afrimte = list(AfriMteLoader(args.data_dir / "afrimte" / "AfriMTE-ade-devtest-v2-with-score").segment_judgments())
    for name in args.features:
        # One feature at a time, closed afterwards, so at most one model is on the GPU.
        feature = registry.get(name)
        results: List[MetaEvalResult] = []
        try:
            LOGGER.info(f"Meta-evaluating {feature.name}")
            results.extend(system_level.evaluate(mqm, feature, "wmt-mqm"))
            for judgments in [outputs.segment_judgments() for outputs in mqm] + da + afrimte:
                results.append(segment_level.evaluate(judgments, feature))
        finally:
            feature.close()
        # Saved per feature, so stopping a long run keeps every feature that finished.
        table = results_file.merge(pd.DataFrame([asdict(result) for result in results]))
        LOGGER.info(f"Saved {feature.name} results to {args.output}")

    pooled = table[table["dataset"].str.endswith("(pooled)")]
    LOGGER.info(f"System-level pairwise accuracy:\n{pooled.to_string(index=False)}")
    segment = table[table["level"] == "segment"]
    collection = segment["dataset"].str.extract(r"^(wmt\d+-\w+|afrimte)", expand=False)
    summary = segment.pivot_table(index=collection, columns=["statistic", "feature"], values="value")
    LOGGER.info(f"Segment-level agreement (mean over sets):\n{summary.round(3).to_string()}")
    LOGGER.info(f"All results -> {args.output}")


if __name__ == "__main__":
    main()
