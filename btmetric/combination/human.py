"""Fits the combined metric on human judgments (WMT MQM and DA) and compares it with each feature alone.

Segment scores for every feature are computed once and saved, since some features train a model per language
pair. Training pairs are two hypotheses for the same source, labelled by which one the humans scored higher.
Accuracy is from leave-one-collection-out predictions (e.g. all of wmt21-mqm held out at once), so the weights
scoring a collection never saw its language pairs' judgments.
"""

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from ..evaluation.data import SegmentJudgments
from ..evaluation.loaders import WmtDaLoader, WmtMqmLoader
from ..features.corpora import NewsCorpus
from ..meta_eval.agreement import PairwiseAgreement
from ..meta_eval.runner import FeatureRegistry
from .model import PairwiseLogLinear, PerGroupScale, leave_one_group_out

LOGGER = logging.getLogger(__name__)


class SegmentScoreTable:
    """Every feature's score for each judged hypothesis, with its dataset, item, system and human score."""

    def __init__(self, path: Path, folds: int = 1) -> None:
        self._path = path
        self._folds = folds

    def build(self, judgments: List[SegmentJudgments], features: List[str], registry: FeatureRegistry) -> None:
        for part in self._scored(judgments, features, registry):
            # Appended per dataset, so a long run that stops keeps what it finished.
            part.to_csv(self._path, sep="\t", index=False, mode="a", header=not self._path.is_file())

    def add_columns(self, judgments: List[SegmentJudgments], features: List[str], registry: FeatureRegistry) -> None:
        """Adds features to an existing table, matching rows on dataset, item, system and human score.

        The judged sets must be the same ones the table was built from; DA sampling uses a fixed seed, so a rerun
        selects the same rows.
        """
        existing = self.load()
        added = pd.concat(list(self._scored(judgments, features, registry)), ignore_index=True)
        keys = ["dataset", "item", "system", "human"]
        labels = ["dataset", "item", "system"]

        def comparable(table: pd.DataFrame) -> np.ndarray:
            # Missing system names are saved empty and read back as NaN.
            return table[labels].fillna({"system": ""}).astype(str).to_numpy()

        same_rows = (
            len(added) == len(existing)
            and (comparable(added) == comparable(existing)).all()
            and np.allclose(added["human"].to_numpy(float), existing["human"].to_numpy(float), equal_nan=True)
        )
        if not same_rows:
            raise ValueError("The judged sets don't match the existing table's rows; rebuild it instead.")
        new_columns = [column for column in added.columns if column not in keys]
        merged = pd.concat([existing.drop(columns=new_columns, errors="ignore"), added[new_columns]], axis=1)
        merged.to_csv(self._path, sep="\t", index=False)

    def _scored(
        self, judgments: List[SegmentJudgments], features: List[str], registry: FeatureRegistry
    ) -> Iterator[pd.DataFrame]:
        loaded = [registry.get(name) for name in features]
        try:
            for i, judged in enumerate(judgments):
                sources, hypotheses = judged.column("source"), judged.column("hypothesis")
                references = judged.column("reference")
                # Each fold is scored by models that never saw its source sentences, for features trained on text
                # drawn from the judged sets themselves (NewsCorpus's fallback). Items stay within one fold.
                folds = np.array([int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % self._folds for s in sources])
                columns: Dict[str, np.ndarray] = {feature.name: np.full(len(sources), np.nan) for feature in loaded}
                for fold in range(self._folds):
                    rows = np.flatnonzero(folds == fold)
                    if len(rows) == 0:
                        continue
                    excluded = frozenset(sources[r] for r in rows)
                    part = ([sources[r] for r in rows], [hypotheses[r] for r in rows], [references[r] for r in rows])
                    for feature in loaded:
                        feature.prepare(judged.source_language, judged.target_language, excluded)
                        columns[feature.name][rows] = feature.segment_scores(*part)
                LOGGER.info(f"Scored {judged.name} ({i + 1} of {len(judgments)})")
                yield pd.DataFrame(
                    {
                        "dataset": judged.name,
                        "item": judged.items(),
                        "system": judged.systems(),
                        "human": judged.human(),
                        **columns,
                    }
                )
        finally:
            for feature in loaded:
                feature.close()

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._path, sep="\t", keep_default_na=False, na_values=[""])


def within_item_pairs(table: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Feature differences, human-preference labels and datasets for every pair of hypotheses of one source.

    Pairs the humans scored equally say nothing about which is better and are skipped.
    """
    deltas, labels, datasets = [], [], []
    for (dataset, _), rows in table.groupby(["dataset", "item"], sort=False):
        if len(rows) < 2:
            continue
        i, j = np.triu_indices(len(rows), k=1)
        human = rows["human"].to_numpy()
        values = rows[features].to_numpy(dtype=float)
        gold = human[i] - human[j]
        decided = gold != 0
        deltas.append(values[i][decided] - values[j][decided])
        labels.append(np.sign(gold[decided]))
        datasets.extend([dataset] * int(decided.sum()))
    return pd.DataFrame(np.vstack(deltas), columns=features), np.concatenate(labels), np.array(datasets)


def collection_of(dataset: str) -> str:
    """wmt21-mqm-en-de-news -> wmt21-mqm; wmt19-da-en-gu -> wmt19-da."""
    return "-".join(dataset.split("-")[:2])


def system_level_accuracy(
    table: pd.DataFrame, scores: np.ndarray, exclude: str = r"^(?:ref|Human)"
) -> PairwiseAgreement:
    """Pairwise accuracy of systems ranked by their mean segment score, pooled over the MQM datasets."""
    scored = table.assign(score=scores)
    scored = scored[scored["system"].astype(str).str.len() > 0]
    scored = scored[~scored["system"].astype(str).str.contains(exclude)]
    gold_deltas, metric_deltas = [], []
    for _, rows in scored.groupby("dataset"):
        means = rows.groupby("system")[["human", "score"]].mean()
        i, j = np.triu_indices(len(means), k=1)
        gold_deltas.append(means["human"].to_numpy()[i] - means["human"].to_numpy()[j])
        metric_deltas.append(means["score"].to_numpy()[i] - means["score"].to_numpy()[j])
    return PairwiseAgreement(np.concatenate(gold_deltas), np.concatenate(metric_deltas))


def in_domain_pool(judged: List[SegmentJudgments]) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    """Each language pair's distinct source-reference pairs across all judged sets: in-domain training text where no
    news corpus exists for the pair. Scoring by folds keeps a segment's own reference out of its models."""
    pool: Dict[Tuple[str, str], Dict[str, str]] = {}
    for judgments in judged:
        pairs = pool.setdefault((judgments.source_language, judgments.target_language), {})
        for source, reference in zip(judgments.column("source"), judgments.column("reference")):
            pairs.setdefault(source, reference)
    return {pair: list(sources.items()) for pair, sources in pool.items()}


def judged_sets(data_dir: Path, max_items: int) -> Iterator[SegmentJudgments]:
    # MQM sets keep every segment, so system-level accuracy uses complete test sets; DA sets are sampled as in
    # the meta-evaluation runner.
    for outputs in WmtMqmLoader(data_dir / "wmt-mqm" / "wmt-mqm-human-evaluation.csv").judged_outputs():
        yield outputs.segment_judgments()
    for judgments in WmtDaLoader(data_dir / "wmt-da" / "wmt-da-human-evaluation.csv").segment_judgments():
        yield judgments.sample_items(max_items)


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Fit the combined metric on human judgments")
    parser.add_argument("--scores", type=Path, default=Path("data/results/segment_scores.tsv"))
    parser.add_argument("--build", nargs="+", choices=registry.names(), help="Compute and save these features first")
    parser.add_argument(
        "--add", nargs="+", choices=registry.names(), help="Compute these features and add them to the existing table"
    )
    parser.add_argument("--features", nargs="+", help="Features to combine (default: all in the scores file)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--max-items", type=int, default=2000)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument(
        "--training-domain",
        choices=["news", "bible"],
        default="news",
        help="Text that aligners and language models train on; it must match the domain scored (news for WMT)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Folds for models trained on the judged sets' own text")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    judged = list(judged_sets(args.data_dir, args.max_items)) if args.build or args.add else []
    if args.training_domain == "news":
        registry = FeatureRegistry(NewsCorpus(in_domain_pool(judged)))
    table_file = SegmentScoreTable(args.scores, folds=args.folds if args.training_domain == "news" else 1)
    if args.build:
        table_file.build(judged, args.build, registry)
    if args.add:
        table_file.add_columns(judged, args.add, registry)
    table = table_file.load()
    features = args.features or [c for c in table.columns if c not in ("dataset", "item", "system", "human")]
    deltas, labels, datasets = within_item_pairs(table, features)
    values = table[features]
    scale = PerGroupScale().fit(values, table["dataset"].to_numpy(), features)
    scaled = scale.apply(deltas, datasets, features)
    collections = np.array([collection_of(dataset) for dataset in datasets])
    combined = leave_one_group_out(scaled, labels, collections, args.l2)

    rows = []
    for collection in sorted(set(collections)):
        held = collections == collection
        row = {
            "collection": collection,
            "pairs": int(held.sum()),
            "combined": float(np.mean(combined[held] * labels[held] > 0)),
        }
        for k, feature in enumerate(features):
            row[feature] = float(np.mean(scaled[held, k] * labels[held] > 0))
        rows.append(row)
    summary = pd.DataFrame(rows).round(3).to_string(index=False)
    LOGGER.info(f"Segment pairs ranked like the humans (combined: leave one collection out):\n{summary}")
    model = PairwiseLogLinear(args.l2).fit(scaled, labels)
    LOGGER.info(f"Weights fit on all collections: {dict(zip(features, model.weight_shares().round(3)))}")

    # System level, on MQM: a segment's combined score uses weights fit without its own collection.
    segment_values = scale.apply(values, table["dataset"].to_numpy(), features)
    segment_collections = table["dataset"].map(collection_of).to_numpy()
    combined_segments = np.full(len(table), np.nan)
    for collection in sorted(set(segment_collections)):
        held = segment_collections == collection
        if "-mqm" in collection:
            weights = PairwiseLogLinear(args.l2).fit(
                scaled[collections != collection], labels[collections != collection]
            )
            combined_segments[held] = segment_values[held] @ weights.weights
    mqm = table["dataset"].str.contains("-mqm-").to_numpy()
    lines = [f"combined: {system_level_accuracy(table[mqm], combined_segments[mqm]).bootstrap_accuracy()}"]
    for k, feature in enumerate(features):
        lines.append(f"{feature}: {system_level_accuracy(table[mqm], segment_values[mqm][:, k]).bootstrap_accuracy()}")
    LOGGER.info("System-level pairwise accuracy on WMT MQM (mean segment score per system):\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
