"""Agreement with silver labels: how often the metric's features rank translations as xCOMET-XL does (PLAN.md 3.3c).

xCOMET can only read languages its encoder (XLM-R) was trained on, so the data are past silnlp system outputs whose
source and target are both among those languages. Three steps, run separately so the GPU step stands alone:

    select   sample verses from those test sets, with every system's output for them, into one table
    score    add a column per feature or silver metric (xCOMET runs on the GPU; its scores are cached)
    agree    pairwise agreement of each metric with each silver label, within verses and at system level
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..evaluation.loaders import ExperimentOutputsLoader
from ..meta_eval.agreement import PairwiseAgreement
from ..meta_eval.runner import FeatureRegistry
from ..scripture.verses import expand_vref

LOGGER = logging.getLogger(__name__)


class XlmrLanguages:
    """The languages of XLM-R's training data (CC-100), which xCOMET's encoder can read, with the three-letter codes
    extract names use for them."""

    def __init__(self, table: pd.DataFrame) -> None:
        self._codes: Dict[str, str] = {}
        for row in table.itertuples(index=False):
            self._codes[row.code] = row.code
            for alias in str(row.aliases).split(","):
                self._codes[alias.strip()] = row.code

    @classmethod
    def load(cls) -> "XlmrLanguages":
        return cls(pd.read_csv(Path(__file__).with_name("xlmr_languages.tsv"), sep="\t"))

    def code(self, extract: str) -> Optional[str]:
        """The language of an extract name such as "en-NIV11R" (first of several), or None if XLM-R lacks it."""
        return self._codes.get(str(extract).split("+")[0].split("-")[0].lower())


class SilverTable:
    """One row per (test set, verse, system), with the texts and one column per scored feature."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def select(
        self, index: pd.DataFrame, loader: ExperimentOutputsLoader, verses_per_set: int, seed: int
    ) -> pd.DataFrame:
        languages = XlmrLanguages.load()
        index = index.dropna(subset=["src", "trg"])
        index = index.assign(source_code=index["src"].map(languages.code), target_code=index["trg"].map(languages.code))
        # Same-language pairs are script conversions or paraphrasing, not translation.
        usable = index[
            index["source_code"].notna() & index["target_code"].notna() & (index["source_code"] != index["target_code"])
        ]
        candidates = usable.groupby("test_hash").filter(lambda rows: rows["prediction_hash"].nunique() >= 2)
        rng = np.random.default_rng(seed)
        frames = []
        for test_hash, rows in candidates.groupby("test_hash"):
            try:
                outputs = loader.outputs_for(test_hash)
            except (OSError, ValueError, KeyError) as e:
                LOGGER.warning(f"Skipping {test_hash}: {e}")
                continue
            if len(outputs.systems()) < 2:
                continue
            evaluation_set = outputs.evaluation_set
            # Some test sets mix in non-Scripture segments (e.g. CCMatrix), which have no verse reference.
            indices = [
                i
                for i, (reference, vref) in enumerate(zip(evaluation_set.references, evaluation_set.segment_ids))
                if reference.strip() and vref
            ]
            if len(indices) > verses_per_set:
                indices = sorted(rng.choice(indices, size=verses_per_set, replace=False))
            first = rows.iloc[0]
            for system, hypotheses in outputs.hypotheses.items():
                frames.append(
                    pd.DataFrame(
                        {
                            "test_hash": test_hash,
                            "source_extract": first["src"].split("+")[0],
                            "target_extract": first["trg"].split("+")[0],
                            "language_pair": f"{first['source_code']}-{first['target_code']}",
                            "vref": [evaluation_set.segment_ids[i] for i in indices],
                            "system": system,
                            "source": [evaluation_set.sources[i] for i in indices],
                            "hypothesis": [hypotheses[i] for i in indices],
                            "reference": [evaluation_set.references[i] for i in indices],
                        }
                    )
                )
        table = pd.concat(frames, ignore_index=True)
        self._save(table)
        LOGGER.info(
            f"Selected {len(table):,} rows: {table['test_hash'].nunique()} test sets, "
            f"{table.groupby('test_hash')['system'].nunique().sum()} system outputs"
        )
        return table

    def score(self, feature_names: List[str], registry: FeatureRegistry) -> pd.DataFrame:
        table = self.load()
        for name in feature_names:
            feature = registry.get(name)
            scores = np.full(len(table), np.nan)
            try:
                groups = list(table.groupby("test_hash").groups.items())
                for k, (test_hash, rows) in enumerate(groups, start=1):
                    part = table.loc[rows]
                    LOGGER.info(f"{feature.name}: test set {k}/{len(groups)} ({test_hash}, {len(part):,} rows)")
                    first = part.iloc[0]
                    # Scripture-trained models leave out the verses about to be scored.
                    excluded = frozenset(v for vref in part["vref"] for v in expand_vref(vref))
                    feature.prepare(first["source_extract"], first["target_extract"], excluded)
                    scores[rows] = feature.verse_scores(
                        part["vref"].tolist(),
                        part["source"].tolist(),
                        part["hypothesis"].tolist(),
                        part["reference"].tolist(),
                    )
            finally:
                feature.close()
            table[feature.name] = scores
            # Saved after each feature, so a long run keeps every feature that finished.
            self._save(table)
        return table

    def load(self) -> pd.DataFrame:
        table = pd.read_csv(self._path, sep="\t", keep_default_na=False, na_values=[""], low_memory=False)
        # Empty cells are missing scores, but in the text columns they are empty outputs, which must stay strings.
        text_columns = ["source", "hypothesis", "reference"]
        table[text_columns] = table[text_columns].fillna("").astype(str)
        # Rows without a verse reference are non-Scripture segments that older selections let in.
        return table[table["vref"].notna()].reset_index(drop=True)

    def _save(self, table: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        table.to_csv(temporary, sep="\t", index=False)
        temporary.replace(self._path)


def silver_mean(table: pd.DataFrame, silver_columns: List[str]) -> pd.Series:
    """The silver labels averaged after z-scoring each within its test set, so their scales don't matter."""
    standardized = [
        table.groupby("test_hash")[column].transform(lambda values: (values - values.mean()) / (values.std() or 1.0))
        for column in silver_columns
    ]
    return pd.concat(standardized, axis=1).mean(axis=1)


def within_verse_agreement(table: pd.DataFrame, silver: str, metric: str) -> PairwiseAgreement:
    """Over every pair of systems translating the same verse: does the metric prefer the one the silver label does?"""
    gold_deltas, metric_deltas = [], []
    scored = table.dropna(subset=[silver, metric])
    for _, rows in scored.groupby(["test_hash", "vref"]):
        if len(rows) < 2:
            continue
        g, m = rows[silver].to_numpy(), rows[metric].to_numpy()
        i, j = np.triu_indices(len(rows), k=1)
        gold_deltas.append(g[i] - g[j])
        metric_deltas.append(m[i] - m[j])
    if not gold_deltas:
        return PairwiseAgreement([], [])
    return PairwiseAgreement(np.concatenate(gold_deltas), np.concatenate(metric_deltas))


def system_level_agreement(table: pd.DataFrame, silver: str, metric: str) -> PairwiseAgreement:
    """Systems ranked by their mean scores within each test set, pooled over test sets."""
    gold_deltas, metric_deltas = [], []
    for _, rows in table.dropna(subset=[silver, metric]).groupby("test_hash"):
        means = rows.groupby("system")[[silver, metric]].mean()
        if len(means) < 2:
            continue
        i, j = np.triu_indices(len(means), k=1)
        gold_deltas.append(means[silver].to_numpy()[i] - means[silver].to_numpy()[j])
        metric_deltas.append(means[metric].to_numpy()[i] - means[metric].to_numpy()[j])
    return PairwiseAgreement(np.concatenate(gold_deltas), np.concatenate(metric_deltas))


def agreement_report(table: pd.DataFrame, silver_columns: List[str], metrics: List[str]) -> pd.DataFrame:
    if len(silver_columns) > 1:
        table = table.assign(**{"silver-mean": silver_mean(table, silver_columns)})
        silver_columns = silver_columns + ["silver-mean"]
    rows = []
    for silver in silver_columns:
        for metric in metrics:
            if metric == silver or metric in silver_columns:
                continue
            within = within_verse_agreement(table, silver, metric)
            system = system_level_agreement(table, silver, metric).bootstrap_accuracy()
            rows.append(
                {
                    "silver": silver,
                    "metric": metric,
                    "within_verse": within.accuracy(),
                    "system_level": system.estimate,
                    "system_level_lower": system.lower,
                    "system_level_upper": system.upper,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Agreement of metric features with xCOMET-XL silver labels")
    parser.add_argument("step", choices=["select", "score", "agree"])
    parser.add_argument("--table", type=Path, default=Path("data/results/silver_rows.tsv"))
    parser.add_argument("--experiments-root", type=Path, default=Path("M:/MT/experiments"))
    parser.add_argument("--index", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--verses-per-set", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", nargs="+", choices=registry.names(), help="For score: features to add")
    parser.add_argument("--silver", nargs="+", default=["xCOMET-XL", "xCOMET-XL-QE"], help="For agree: label columns")
    parser.add_argument("--output", type=Path, default=Path("data/results/silver_agreement.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    table_file = SilverTable(args.table)
    if args.step == "select":
        index = pd.read_csv(args.index, sep="\t", dtype={"step": str})
        table_file.select(index, ExperimentOutputsLoader(args.experiments_root, index), args.verses_per_set, args.seed)
    elif args.step == "score":
        if not args.features:
            parser.error("score needs --features")
        table_file.score(args.features, registry)
    else:
        table = table_file.load()
        silver = [column for column in args.silver if column in table.columns]
        if not silver:
            parser.error(f"None of the silver columns {args.silver} has been scored yet")
        fixed = {"test_hash", "source_extract", "target_extract", "language_pair", "vref", "system", "source"}
        metrics = [c for c in table.columns if c not in fixed | {"hypothesis", "reference"} and c not in silver]
        report = agreement_report(table, silver, metrics)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, sep="\t", index=False)
        LOGGER.info(f"Agreement with the silver labels (pairwise accuracy):\n{report.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
