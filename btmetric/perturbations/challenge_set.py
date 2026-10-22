"""Builds a Bible challenge set from real MT outputs, and checks which perturbations each metric notices.

Each row pairs an MT output for a verse (the original) with a copy carrying one injected error (the perturbed).
Scored against the same human reference, a metric should prefer the original; the share of rows where it does is
its accuracy for that error type.
"""

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..evaluation.data import JudgedOutputs
from ..evaluation.loaders import ExperimentOutputsLoader
from ..meta_eval.runner import FeatureRegistry
from ..terms.major_terms import MajorTermList, ProjectKeyTerms, find_renderings_file
from ..terms.project_alignment import ProjectAlignment
from .perturbations import Perturbation, VerseContext, all_perturbations

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChallengeRow:
    test_hash: str
    target: str
    vref: str
    perturbation: str
    severity: int
    source: str
    reference: str
    original: str
    perturbed: str


@dataclass(frozen=True)
class ProjectResources:
    key_terms: Optional[ProjectKeyTerms]
    key_terms_source: str
    negation_markers: Tuple[str, ...]


class TargetProjectResources:
    """Key terms and negation words for a target project.

    Curated renderings are used where the project has them. Otherwise renderings are inferred from an English-target
    alignment, as they must be for most projects; that alignment also gives the negation words.
    """

    def __init__(
        self,
        terms: MajorTermList,
        terms_dir: Path,
        scripture_dir: Path,
        vref_lines: List[str],
        english_lines: List[str],
        min_confidence: float,
        prefer_curated: bool = True,
    ) -> None:
        self._terms = terms
        self._terms_dir = terms_dir
        self._scripture_dir = scripture_dir
        self._vref_lines = vref_lines
        self._english_lines = english_lines
        self._min_confidence = min_confidence
        self._prefer_curated = prefer_curated

    def for_target(self, target: str, test_vrefs: Set[str]) -> ProjectResources:
        extract = target.split("+")[0]
        # Test verses are left out of the alignment, as they would be for any auxiliary model.
        alignment = ProjectAlignment.build(
            self._terms, self._vref_lines, self._english_lines, self._scripture_dir / f"{extract}.txt", test_vrefs
        )
        negation = tuple(alignment.negation_markers()) if alignment else ()
        curated_path = find_renderings_file(self._terms_dir, extract) if self._prefer_curated else None
        if curated_path:
            return ProjectResources(ProjectKeyTerms.load(self._terms, curated_path), "curated", negation)
        if alignment:
            return ProjectResources(alignment.key_terms(self._min_confidence), "inferred", negation)
        return ProjectResources(None, "none", negation)


class ChallengeSetBuilder:
    def __init__(
        self,
        perturbations: List[Perturbation],
        resources: TargetProjectResources,
        verses_per_set: int,
        seed: int,
        perturb_reference: bool = False,
    ) -> None:
        self._perturbations = perturbations
        self._resources = resources
        self._verses_per_set = verses_per_set
        self._rng = np.random.default_rng(seed)
        self._perturb_reference = perturb_reference

    def rows_for(self, outputs: JudgedOutputs, target: str) -> List[ChallengeRow]:
        evaluation_set = outputs.evaluation_set
        if self._perturb_reference:
            # The memorization gate (PLAN.md 3.3a) perturbs the human reference itself, to check whether the error
            # survives back-translation; metrics scored against the reference are trivially right on these rows.
            base = evaluation_set.references
        else:
            # One MT system per test set, picked at random so no metric is favored by how the base was chosen.
            systems = outputs.systems()
            base = outputs.hypotheses[systems[int(self._rng.integers(len(systems)))]]
        candidates = [
            i for i, (hyp, ref) in enumerate(zip(base, evaluation_set.references)) if hyp.strip() and ref.strip()
        ]
        if len(candidates) > self._verses_per_set:
            candidates = sorted(self._rng.choice(candidates, size=self._verses_per_set, replace=False))
        resources = self._resources.for_target(target, set(evaluation_set.segment_ids))
        LOGGER.info(
            f"{target}: key terms {resources.key_terms_source}, negation words {list(resources.negation_markers)}"
        )
        rows = []
        for i in candidates:
            vref = evaluation_set.segment_ids[i]
            context = VerseContext(
                source=evaluation_set.sources[i],
                other_verses=tuple(base[j] for j in self._neighbours(i, len(base))),
                key_terms=resources.key_terms.occurrences(vref, base[i], self._rng) if resources.key_terms else (),
                negation_markers=resources.negation_markers,
            )
            for perturbation in self._perturbations:
                perturbed = perturbation.apply(base[i], context, self._rng)
                if perturbed and perturbed.strip() and perturbed != base[i]:
                    rows.append(
                        ChallengeRow(
                            evaluation_set.name,
                            target,
                            vref,
                            perturbation.name,
                            perturbation.severity,
                            evaluation_set.sources[i],
                            evaluation_set.references[i],
                            base[i],
                            perturbed,
                        )
                    )
        return rows

    def _neighbours(self, i: int, n: int) -> range:
        # Nearby verses are on the same topic, so a wrong verse or added clause is plausible rather than random.
        return range(max(0, i - 10), min(n, i + 11))


def evaluate(
    challenge: pd.DataFrame,
    feature_names: List[str],
    registry: FeatureRegistry,
    languages: Optional[Dict[str, Tuple[str, str]]] = None,
    row_scores: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Per feature and perturbation: the share of rows where the original scores strictly higher, overall and per
    severity, and for graded perturbations the share of verses where the mild version outscores the severe one.

    With languages (test set -> source and target extract), each test set is scored separately after preparing
    the feature for its language pair, leaving that set's verses out of any model trained on Scripture.
    """
    if "severity" not in challenge:
        challenge = challenge.assign(severity=1)
    challenge = challenge.reset_index(drop=True)
    results = []
    for name in feature_names:
        feature = registry.get(name)
        original = np.full(len(challenge), np.nan)
        perturbed = np.full(len(challenge), np.nan)
        try:
            groups = challenge.groupby("test_hash").groups.items() if languages else [(None, challenge.index)]
            for test_hash, rows in groups:
                part = challenge.loc[rows]
                if test_hash is not None and test_hash in languages:
                    feature.prepare(*languages[test_hash], frozenset(part["vref"]))
                sources, references = part["source"].tolist(), part["reference"].tolist()
                original[rows] = feature.segment_scores(sources, part["original"].tolist(), references)
                perturbed[rows] = feature.segment_scores(sources, part["perturbed"].tolist(), references)
        finally:
            feature.close()
        if row_scores is not None:
            # Kept per row so features can be combined and weighted later (PLAN.md 4.1).
            row_scores[f"{feature.name}|original"] = original
            row_scores[f"{feature.name}|perturbed"] = perturbed
        # A feature may have no score for a row (no key term in the verse, no Bible text for the language); those
        # rows say nothing about it, so they are left out rather than counted as failures.
        valid = ~(np.isnan(original) | np.isnan(perturbed))
        prefers_original = np.where(valid, original > perturbed, np.nan)
        scored = challenge.assign(prefers_original=prefers_original, perturbed_score=perturbed)
        for perturbation, rows in scored.groupby("perturbation"):
            by_severity = rows.groupby("severity")["prefers_original"].mean()
            results.append(
                {
                    "feature": feature.name,
                    "perturbation": perturbation,
                    "accuracy": float(rows["prefers_original"].mean()),
                    "accuracy_mild": float(by_severity.get(1, np.nan)),
                    "accuracy_severe": float(by_severity.get(2, np.nan)),
                    "graded": _graded_share(rows),
                    "rows": len(rows),
                    "scored_rows": int(rows["prefers_original"].notna().sum()),
                }
            )
    return pd.DataFrame(results)


def _graded_share(rows: pd.DataFrame) -> float:
    """Share of verses where the mild version scores strictly higher than the severe one."""
    paired = rows.pivot_table(
        index=["test_hash", "vref"], columns="severity", values="perturbed_score", aggfunc="first"
    ).dropna()
    if 1 not in paired.columns or 2 not in paired.columns or paired.empty:
        return float("nan")
    return float((paired[1] > paired[2]).mean())


class HeldOutTestSets:
    """Picks held-out-book test sets from the experiment index, one per target language for diversity."""

    def __init__(self, index: pd.DataFrame) -> None:
        self._index = index

    def held_out_books(self, count: int, seed: int) -> Dict[str, str]:
        books = self._index[(self._index["test_design"] == "books") & self._index["trg"].notna()]
        books = books.assign(language=books["trg"].str.split("-").str[0])
        one_per_language = books.drop_duplicates("language").sample(frac=1, random_state=seed).head(count)
        return dict(zip(one_per_language["test_hash"], one_per_language["trg"]))


def build_main() -> None:
    parser = argparse.ArgumentParser(description="Build a Bible challenge set of perturbed MT outputs")
    parser.add_argument("--experiments-root", type=Path, default=Path("M:/MT/experiments"))
    parser.add_argument("--index", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--terms-dir", type=Path, default=Path("M:/MT/terms"))
    parser.add_argument("--assets-dir", type=Path, default=Path("../silnlp/silnlp/assets"))
    parser.add_argument("--scripture-dir", type=Path, default=Path("M:/MT/scripture"))
    parser.add_argument("--english", default="en-BSB", help="English extract aligned with each target")
    parser.add_argument(
        "--min-term-confidence", type=float, default=0.5, help="Minimum confidence of inferred renderings"
    )
    parser.add_argument("--infer-all-terms", action="store_true", help="Infer renderings even where curated ones exist")
    parser.add_argument("--test-sets", type=int, default=60, help="Held-out-book test sets, one per target language")
    parser.add_argument("--verses-per-set", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/challenge_set.tsv"))
    parser.add_argument(
        "--perturb-reference", action="store_true", help="Perturb the human reference instead of an MT output"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = pd.read_csv(args.index, sep="\t", dtype={"step": str})
    test_sets = HeldOutTestSets(index).held_out_books(args.test_sets, args.seed)
    loader = ExperimentOutputsLoader(args.experiments_root, index)
    resources = TargetProjectResources(
        MajorTermList.load(args.assets_dir),
        args.terms_dir,
        args.scripture_dir,
        (args.assets_dir / "vref.txt").read_text(encoding="utf-8").splitlines(),
        (args.scripture_dir / f"{args.english}.txt").read_text(encoding="utf-8-sig").splitlines(),
        args.min_term_confidence,
        prefer_curated=not args.infer_all_terms,
    )
    builder = ChallengeSetBuilder(
        all_perturbations(), resources, args.verses_per_set, args.seed, perturb_reference=args.perturb_reference
    )
    rows: List[ChallengeRow] = []
    for test_hash, target in test_sets.items():
        try:
            outputs = loader.outputs_for(test_hash)
        except (OSError, ValueError, KeyError) as e:
            LOGGER.warning(f"Skipping {test_hash}: {e}")
            continue
        if outputs.systems():
            rows.extend(builder.rows_for(outputs, target))
    table = pd.DataFrame([asdict(row) for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(
        f"{len(table)} challenge rows from {table['test_hash'].nunique()} test sets -> {args.output}\n"
        + table["perturbation"].value_counts().to_string()
    )


def evaluate_main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Check which perturbations each metric notices")
    parser.add_argument("--challenge-set", type=Path, default=Path("data/challenge_set.tsv"))
    parser.add_argument("--features", nargs="+", default=registry.default_names(), choices=registry.names())
    parser.add_argument("--output", type=Path, default=Path("data/results/challenge_set.tsv"))
    parser.add_argument("--error-rates", type=Path, default=Path("data/results/error_rates.tsv"))
    parser.add_argument(
        "--index", type=Path, default=Path("data/experiments.tsv"), help="For each test set's source and target"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    challenge = pd.read_csv(args.challenge_set, sep="\t", keep_default_na=False)
    languages = None
    if args.index.is_file():
        index = pd.read_csv(args.index, sep="\t", dtype={"step": str}).dropna(subset=["src", "trg"])
        first = index.drop_duplicates("test_hash")
        # Multi-source experiments list several extracts; the first stands in for the language pair.
        languages = {
            row.test_hash: (row.src.split("+")[0], row.trg.split("+")[0]) for row in first.itertuples(index=False)
        }
    row_scores: Dict[str, np.ndarray] = {}
    results = evaluate(challenge, args.features, registry, languages, row_scores)
    RowScores(args.output.with_name(args.output.stem + "_rows.tsv")).merge(challenge, row_scores)
    if args.output.is_file():
        previous = pd.read_csv(args.output, sep="\t")
        results = pd.concat([previous[~previous["feature"].isin(results["feature"])], results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, sep="\t", index=False)
    table = results.pivot(index="perturbation", columns="feature", values="accuracy")
    LOGGER.info(f"Share of verses where the metric prefers the unperturbed output:\n{table.round(3).to_string()}")
    graded = results.dropna(subset=["graded"]).pivot(index="perturbation", columns="feature", values="graded")
    if not graded.empty:
        LOGGER.info(f"Share of verses where a mild error outscores a severe one:\n{graded.round(3).to_string()}")
    if args.error_rates.is_file():
        weights = CalibratedWeights.from_error_rates(pd.read_csv(args.error_rates, sep="\t"))
        LOGGER.info(f"Accuracy weighted by how often each error occurs in real MT output:\n{weights.apply(table)}")


class RowScores:
    """Every feature's scores for each challenge row's original and perturbed text, accumulated across runs.

    Rows are matched by position, so a file only ever holds scores for one challenge set; a different set
    replaces it.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def merge(self, challenge: pd.DataFrame, scores: Dict[str, np.ndarray]) -> pd.DataFrame:
        keys = challenge[["test_hash", "vref", "perturbation", "severity"]].reset_index(drop=True)
        table = keys.assign(**scores)
        if self._path.is_file():
            previous = pd.read_csv(self._path, sep="\t", keep_default_na=False, na_values=[""])
            if len(previous) == len(keys) and (previous[keys.columns].astype(str) == keys.astype(str)).all().all():
                kept = [column for column in previous.columns if column not in table.columns]
                table = pd.concat([table, previous[kept]], axis=1)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(self._path, sep="\t", index=False)
        return table


class CalibratedWeights:
    """Weights perturbation types by how often their signature occurs in real MT output (calibration.py).

    Only types with a measurable signature are weighted; key terms, negation, word order and wrong verses have
    none, so they stay out of the calibrated score and are read from the per-type table instead.
    """

    def __init__(self, weights: Dict[str, float]) -> None:
        total = sum(weights.values())
        self._weights = {name: weight / total for name, weight in weights.items()} if total else {}

    @classmethod
    def from_error_rates(cls, error_rates: pd.DataFrame) -> "CalibratedWeights":
        # Each test set counts once, however many systems it has.
        rates = error_rates.drop(columns=["target", "system"]).groupby("test_hash").mean().mean()
        short = float(rates.get("too_short", 0.0))
        return cls(
            {
                "repetition": float(rates.get("loop", rates.get("repetition", 0.0))),
                "addition": float(rates.get("too_long", 0.0)),
                # A short output may have lost a clause or its ending; the two can't be told apart by length.
                "omission": short / 2,
                "truncation": short / 2,
                "source_copy": float(rates.get("source_copy", 0.0)),
            }
        )

    def apply(self, accuracy_table: pd.DataFrame) -> pd.Series:
        """Weighted accuracy per feature, from a perturbation x feature table of accuracies."""
        weights = pd.Series(self._weights).reindex(accuracy_table.index).fillna(0.0)
        return (accuracy_table.mul(weights, axis=0).sum() / weights.sum()).round(3)
