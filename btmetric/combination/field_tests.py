"""Fit the combined metric on the field-test draft comparisons and check the weights everywhere else (PLAN.md 4.1).

The field tests are the only human judgments of drafts in minority languages (PLAN.md 3.3g), so the weights are fit
there, leaving one team out at a time. The same weights are then applied, unchanged, to the Bible challenge set, the
WMT judgments and the xCOMET-XL silver labels, so one weighting can be judged against every test at once.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..analysis.silver_agreement import SilverTable, system_level_agreement, within_verse_agreement
from ..combination.challenge import ChallengeRowScores, accuracy_by_perturbation
from ..combination.human import system_level_accuracy, within_item_pairs
from ..combination.model import PairwiseLogLinear, PerGroupScale

LOGGER = logging.getLogger(__name__)

FEATURE_SETS: Dict[str, List[str]] = {
    "chrF3": ["chrF3"],
    "chrF3 + coverage-target": ["chrF3", "coverage-target"],
    "chrF3 + coverage": ["chrF3", "coverage-source", "coverage-target"],
    "chrF3 + detectors": ["chrF3", "coverage-source-detector", "coverage-target-detector"],
    "all": [
        "chrF3",
        "spBLEU",
        "perplexity",
        "coverage-source",
        "coverage-target",
        "coverage-reference",
        "coverage-source-detector",
        "coverage-target-detector",
        "coverage-reference-detector",
        "key-terms",
    ],
}


DETECTORS = [f"coverage-{side}-detector" for side in ["source", "target", "reference"]]


class FieldTestPairs:
    """The compared verses a team preferred one draft on: scaled feature differences A - B and +1 when A won."""

    def __init__(self, rows: pd.DataFrame, features: List[str]) -> None:
        compared = rows[(rows["kind"] == "comparison") & rows["label"].isin(["A", "B"])]
        wide = compared.pivot_table(index=["item", "label", "team", "independent"], columns="system", values=features)
        wide = wide.reset_index()
        self.labels = np.where(wide["label"] == "A", 1.0, -1.0)
        self.teams = wide["team"].to_numpy()
        self.independent = wide["independent"].to_numpy(dtype=bool)
        raw = pd.DataFrame({f: wide[(f, "A")] - wide[(f, "B")] for f in features})
        scored = rows[rows["kind"] == "comparison"]
        self.scale = PerGroupScale(DETECTORS).fit(scored, scored["team"].to_numpy(), features)
        self.deltas = self.scale.apply(raw, self.teams, features)

    def fit(self, l2: float, independent_only: bool) -> PairwiseLogLinear:
        """References written from a draft favour that draft, which teaches the fit that reference-based features
        disagree with the teams, so training can be limited to independent references."""
        train = self.independent if independent_only else np.ones(len(self.labels), dtype=bool)
        return PairwiseLogLinear(l2).fit(self.deltas[train], self.labels[train])

    def leave_one_team_out(self, l2: float, independent_only: bool = False) -> np.ndarray:
        """Each team's pairs scored by weights fit on the other teams; positive when the score sides with the team."""
        train = self.independent if independent_only else np.ones(len(self.labels), dtype=bool)
        predictions = np.full(len(self.labels), np.nan)
        for team in np.unique(self.teams):
            held = self.teams == team
            model = PairwiseLogLinear(l2).fit(self.deltas[~held & train], self.labels[~held & train])
            predictions[held] = model.score(self.deltas[held])
        return predictions * self.labels


def _accuracy(signed: np.ndarray) -> float:
    """Share of pairs on the side of the humans; a tie counts as wrong, as in PairwiseAgreement."""
    return float(np.mean(signed > 0)) if len(signed) else float("nan")


def _weighted(values: pd.DataFrame, groups: np.ndarray, weights: Dict[str, float]) -> Optional[np.ndarray]:
    """The combined score of every row in its own group's scale, or None if a weighted feature is missing."""
    features = [f for f, w in weights.items() if w > 0]
    if any(f not in values.columns for f in features):
        return None
    if not features:
        return np.zeros(len(values))
    scale = PerGroupScale(DETECTORS).fit(values, groups, features)
    return scale.apply(values, groups, features) @ np.array([weights[f] for f in features])


def check_challenge_set(path: Path, weights: Dict[str, float]) -> float:
    """Mean over perturbation types of the share of rows where the original outscores its perturbed copy."""
    rows = ChallengeRowScores.load(path)
    features = [f for f, w in weights.items() if w > 0]
    if any(f not in rows.features() for f in features):
        return float("nan")
    groups = rows.groups()
    scale = PerGroupScale(DETECTORS).fit(rows.values(features), np.concatenate([groups, groups]), features)
    signed = scale.apply(rows.deltas(features), groups, features) @ np.array([weights[f] for f in features])
    return float(accuracy_by_perturbation(signed, rows.perturbations()).mean())


def check_wmt(path: Path, weights: Dict[str, float], max_pairs: int) -> Dict[str, float]:
    table = pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""], low_memory=False)
    features = [f for f, w in weights.items() if w > 0]
    if any(f not in table.columns for f in features):
        return {"wmt_segment": float("nan"), "wmt_system": float("nan")}
    deltas, labels, datasets = within_item_pairs(table, features, max_pairs)
    scale = PerGroupScale(DETECTORS).fit(table, table["dataset"].to_numpy(), features)
    w = np.array([weights[f] for f in features])
    segment = _accuracy((scale.apply(deltas, datasets, features) @ w) * labels)
    mqm = table["dataset"].str.contains("-mqm-").to_numpy()
    combined = scale.apply(table, table["dataset"].to_numpy(), features) @ w
    system = system_level_accuracy(table[mqm], combined[mqm]).accuracy()
    return {"wmt_segment": segment, "wmt_system": system}


DYNAMIC_TRANSLATIONS = ["en-NLT07", "en-GNTD", "en-CEVUS06"]


def check_human_translations(paths: List[Path], weights: Dict[str, float]) -> Dict[str, float]:
    """The share of MT systems each published translation outscores on the NASB test (PLAN.md 3.3b), averaged over
    all of them and over the dynamic ones, which literal-leaning features punish.

    Only system-level scores were saved there, so each feature is scaled by its spread over the MT systems rather
    than over verses: an approximation of the weighting the fit learned."""
    scores = pd.concat([pd.read_csv(path, sep="\t") for path in paths if path.is_file()], ignore_index=True)
    wide = scores.pivot_table(index=["kind", "candidate"], columns="feature", values="score").reset_index()
    features = [f for f, w in weights.items() if w > 0]
    if any(f not in wide.columns for f in features):
        return {"human_all": float("nan"), "human_dynamic": float("nan")}
    mt = wide["kind"] == "mt"
    spread = wide.loc[mt, features].std().replace(0, 1.0)
    combined = (wide[features] / spread) @ np.array([weights[f] for f in features])
    mt_scores = combined[mt].to_numpy()
    shares = pd.Series({c: float(np.mean(v > mt_scores)) for c, v in zip(wide["candidate"][~mt], combined[~mt])})
    return {"human_all": shares.mean(), "human_dynamic": shares.reindex(DYNAMIC_TRANSLATIONS).mean()}


def check_silver(path: Path, weights: Dict[str, float], silver: str) -> Dict[str, float]:
    table = SilverTable(path).load()
    combined = _weighted(table, table["test_hash"].to_numpy(), weights)
    if combined is None or silver not in table.columns:
        return {"silver_verse": float("nan"), "silver_system": float("nan")}
    table = table.assign(combined=combined)
    return {
        "silver_verse": within_verse_agreement(table, silver, "combined").accuracy(),
        "silver_system": system_level_agreement(table, silver, "combined").accuracy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the combined metric on the field-test draft comparisons")
    parser.add_argument("--rows", type=Path, default=Path("data/field_tests/rows.tsv"))
    parser.add_argument("--challenge-rows", type=Path, default=Path("data/results/challenge_set_rows.tsv"))
    parser.add_argument("--wmt-scores", type=Path, default=Path("data/results/segment_scores.tsv"))
    parser.add_argument("--silver-rows", type=Path, default=Path("data/results/silver_rows.tsv"))
    parser.add_argument("--silver", default="xCOMET-XL")
    parser.add_argument(
        "--human-translations",
        nargs="+",
        type=Path,
        default=[
            Path(f"data/results/human_translations{suffix}.tsv")
            for suffix in ["", "_features", "_features2", "_detectors"]
        ],
    )
    parser.add_argument("--max-pairs-per-dataset", type=int, default=20000)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument(
        "--train-on",
        choices=["independent", "all"],
        default="independent",
        help="Comparisons to fit on: only those with references from before the trial, or all",
    )
    parser.add_argument("--output", type=Path, default=Path("data/results/combination_field_tests.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = SilverTable(args.rows).load()
    records = []
    for name, features in FEATURE_SETS.items():
        pairs = FieldTestPairs(rows, features)
        held_out = pairs.leave_one_team_out(args.l2, args.train_on == "independent")
        model = pairs.fit(args.l2, args.train_on == "independent")
        weights = dict(zip(features, model.weight_shares()))
        LOGGER.info(f"{name}: weights fit on all teams {({f: round(w, 3) for f, w in weights.items()})}")
        record = {
            "features": name,
            "field_independent": _accuracy(held_out[pairs.independent]),
            "field_all": _accuracy(held_out),
            "challenge": check_challenge_set(args.challenge_rows, weights),
            **check_wmt(args.wmt_scores, weights, args.max_pairs_per_dataset),
            **check_human_translations(args.human_translations, weights),
            **check_silver(args.silver_rows, weights, args.silver),
            "weights": ", ".join(f"{f} {w:.2f}" for f, w in weights.items() if w > 0.005),
        }
        records.append(record)
    report = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(
        "Field tests: leave one team out. Elsewhere: the weights fit on all teams, unchanged.\n"
        + report.drop(columns="weights").round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
