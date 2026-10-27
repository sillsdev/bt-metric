"""Meta-evaluates features against the human judgments in the downloaded datasets."""

import argparse
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from machine.translation import SymmetrizationHeuristic
import pandas as pd
from scipy.stats import kendalltau

from ..evaluation.data import JudgedOutputs, SegmentJudgments
from ..evaluation.loaders import AfriMteLoader, WmtDaLoader, WmtMqmLoader
from ..features.alignment import AlignerCache, AlignmentCoverage, PivotCoverage, ReferenceAnchoredCoverage
from ..features.back_translation import BackTranslationInfo
from ..features.base import Feature
from ..features.chrf import ChrF
from ..features.chrf_b import ChrFB
from ..features.comet import Comet
from ..features.corpora import BibleCorpus, TrainingCorpus
from ..features.key_terms import InferredRenderings, KeyTermAccuracy
from ..features.lexical import Bleu, MeanChrF, NegativeTer
from ..features.negation import NegationConsistency
from ..features.unseen_words import UnseenWords, VocabularyCache
from ..features.naturalness import CharNgramModel, PerplexityRatio
from ..features.reverse_models import FineTunedNllb, ScratchReverseModel
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
        for k, outputs in enumerate(outputs_list, start=1):
            outputs = outputs.without_systems([s for s in outputs.systems() if self._is_excluded(s)])
            evaluation_set = outputs.evaluation_set
            LOGGER.info(f"{feature.name}: system level, set {k}/{len(outputs_list)} ({evaluation_set.name})")
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


# "direct" keeps only the source-to-target model, which links each target word to at most one source word or to nothing.
_HEURISTICS = {
    "och": SymmetrizationHeuristic.OCH,
    "intersection": SymmetrizationHeuristic.INTERSECTION,
    "direct": SymmetrizationHeuristic.NONE,
}


def _renamed(feature: Feature, name: str) -> Feature:
    feature.name = name
    return feature


class FeatureRegistry:
    """Features by name, built on first use: some load models or download tokenizers when constructed."""

    def __init__(self, corpus: Optional[TrainingCorpus] = None) -> None:
        """corpus: the training text for features that train models, in the domain being scored (Bible by default;
        news for WMT). Key terms always use Bible text, since they rely on verse references."""
        corpus = corpus or BibleCorpus()
        # Shared so that related features train each language's models once between them.
        self._aligners = AlignerCache(corpus)
        self._subword_aligners = AlignerCache(corpus, subword_vocab=8000)
        self._vocabularies = VocabularyCache(corpus)
        self._renderings = InferredRenderings()
        self._language_models: Dict[str, CharNgramModel] = {}
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
            # The same model without the reference (quality estimation), so its labels do not share our reference.
            "xcomet-xl-qe": lambda: Comet("xCOMET-XL-QE", "Unbabel/XCOMET-XL", uses_reference=False, batch_size=16),
            "coverage-source": lambda: AlignmentCoverage("source", aligners=self._aligners),
            "coverage-target": lambda: AlignmentCoverage("target", aligners=self._aligners),
            "key-terms": lambda: KeyTermAccuracy(renderings=self._renderings),
            "key-terms-names": lambda: KeyTermAccuracy(categories=frozenset({"PN"}), renderings=self._renderings),
            "perplexity": lambda: PerplexityRatio(corpus=corpus, models=self._language_models),
            "perplexity-one-sided": lambda: PerplexityRatio(
                corpus=corpus, one_sided=True, models=self._language_models
            ),
            "coverage-reference": lambda: ReferenceAnchoredCoverage(aligners=self._aligners),
            "coverage-source-detector": lambda: AlignmentCoverage(
                "source", aligners=self._aligners, detector_tolerance=0.3
            ),
            "coverage-target-detector": lambda: AlignmentCoverage(
                "target", aligners=self._aligners, detector_tolerance=0.3
            ),
            "bt-nllb": BackTranslationInfo,
            "bt-nllb-finetuned": lambda: BackTranslationInfo(FineTunedNllb(corpus)),
            "bt-scratch": lambda: BackTranslationInfo(ScratchReverseModel(corpus)),
            "coverage-reference-detector": lambda: ReferenceAnchoredCoverage(
                aligners=self._aligners, detector_tolerance=0.3
            ),
            # Symmetrization alternatives for coverage and key terms (the defaults above use grow-diag-final-and).
            "coverage-target-och": lambda: self._coverage_with("och"),
            "coverage-target-intersection": lambda: self._coverage_with("intersection"),
            "coverage-target-direct": lambda: self._coverage_with("direct"),
            # SentencePiece pieces instead of words: 8,000-piece unigram models trained on each aligner's text.
            "coverage-target-subword": lambda: _renamed(
                AlignmentCoverage("target", aligners=self._subword_aligners), "coverage-target-subword"
            ),
            "coverage-target-english": lambda: PivotCoverage(
                "en-BSB", "coverage-target-english", aligners=self._aligners
            ),
            "coverage-target-original": lambda: PivotCoverage(
                "orig-UHB_UGNT", "coverage-target-original", aligners=self._aligners
            ),
            "key-terms-och": lambda: _renamed(
                KeyTermAccuracy(renderings=self._renderings, heuristic=SymmetrizationHeuristic.OCH),
                "key-terms-och",
            ),
            "negation": lambda: NegationConsistency(renderings=self._renderings),
            "negation-missing": lambda: NegationConsistency(renderings=self._renderings, missing_only=True),
            "unseen-words": lambda: UnseenWords(vocabularies=self._vocabularies),
            "unseen-words-implausible": lambda: UnseenWords(implausible_only=True, vocabularies=self._vocabularies),
            # The recommended metric (PLAN.md 4.1): chrF3, coverage of the hypothesis and key terms, in chrF3 points.
            "chrf-b": lambda: ChrFB(
                ChrF(0),
                AlignmentCoverage("target", aligners=self._aligners),
                KeyTermAccuracy(renderings=self._renderings),
            ),
        }
        # Run only when asked for: neural features need a GPU; sacrebleu's TER is pure Python and takes over an hour
        # on the WMT sets while ranking almost exactly like chrF and BLEU; coverage trains an aligner per language
        # pair from Bible text on the mounted bucket.
        self._opt_in = {
            "ter",
            "comet22",
            "africomet",
            "xcomet-xl",
            "xcomet-xl-qe",
            "coverage-source",
            "coverage-target",
            "key-terms",
            "key-terms-names",
            "perplexity",
            "perplexity-one-sided",
            "coverage-reference",
            "coverage-source-detector",
            "coverage-target-detector",
            "coverage-reference-detector",
            "coverage-target-och",
            "coverage-target-intersection",
            "coverage-target-direct",
            "coverage-target-subword",
            "coverage-target-english",
            "coverage-target-original",
            "key-terms-och",
            "negation",
            "negation-missing",
            "unseen-words",
            "unseen-words-implausible",
            "bt-nllb",
            "bt-nllb-finetuned",
            "bt-scratch",
            "chrf-b",
        }

    def _coverage_with(self, heuristic: str) -> Feature:
        coverage = AlignmentCoverage("target", aligners=self._aligners, heuristic=_HEURISTICS[heuristic])
        return _renamed(coverage, f"coverage-target-{heuristic}")

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
            segment_sets = [outputs.segment_judgments() for outputs in mqm] + da + afrimte
            for k, judgments in enumerate(segment_sets, start=1):
                LOGGER.info(f"{feature.name}: segment level, set {k}/{len(segment_sets)} ({judgments.name})")
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
