"""Agreement of metric features with the field-test judgments of Scripture Forge drafts (PLAN.md 3.3).

Three steps, like the silver-label agreement:

    build    one row per scored text (each draft of a compared pair, each usability draft) with its reference
    score    add a column per feature
    agree    how often each feature sides with the teams

Drafts are scored against the latest text of the team's project, where it has the verse. In the human-vs-AI sheets
the team's own translation of the verse is the reference instead.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..datasets.field_tests import ANALYSIS_LABELS
from ..meta_eval.agreement import PairwiseAgreement
from ..meta_eval.runner import FeatureRegistry
from .silver_agreement import SilverTable

LOGGER = logging.getLogger(__name__)

DETECTOR_FEATURES = [f"coverage-{side}-detector" for side in ["source", "target", "reference"]]

TEXT_COLUMNS = {"item", "comparison", "team", "kind", "label", "test_hash", "source_extract", "target_extract", "vref"}
TEXT_COLUMNS |= {"system", "source", "hypothesis", "reference", "independent"}


def scoring_rows(judgments: pd.DataFrame) -> pd.DataFrame:
    """The texts to score: two rows (systems A and B) per compared verse, one per usability or human-vs-AI draft."""
    judgments = judgments.assign(item=range(len(judgments)))
    judgments = judgments.assign(
        reference=np.where(judgments["kind"] == "human_ai", judgments["human_translation"], judgments["reference"])
    )
    judgments = judgments[(judgments["reference"] != "") & (judgments["kind"] != "correction")]
    frames = []
    for system, column in [("A", "draft_a"), ("B", "draft_b")]:
        part = judgments if system == "A" else judgments[judgments["kind"] == "comparison"]
        frames.append(
            pd.DataFrame(
                {
                    "item": part["item"],
                    "comparison": part["comparison"],
                    "team": part["team"],
                    "kind": part["kind"],
                    "label": part["label"],
                    # SilverTable groups by test_hash to prepare features once per team's language pair.
                    "test_hash": part["team"],
                    "source_extract": part["source_extract"],
                    "target_extract": part["target_extract"],
                    "vref": part["vref"],
                    "system": system if system == "B" else np.where(part["kind"] == "comparison", "A", "draft"),
                    "source": part["source"],
                    "hypothesis": part[column],
                    "reference": part["reference"],
                    "independent": part["independent_reference"] == "True",
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["item", "system"], ignore_index=True)


def verse_preferences(rows: pd.DataFrame, metric: str) -> PairwiseAgreement:
    """Each compared verse the team preferred one draft on: does the metric prefer the same one?"""
    compared = rows[(rows["kind"] == "comparison") & rows["label"].isin(["A", "B"])]
    wide = compared.pivot_table(index=["item", "label"], columns="system", values=metric).dropna().reset_index()
    gold = np.where(wide["label"] == "A", 1.0, -1.0)
    return PairwiseAgreement(gold, wide["A"] - wide["B"])


def comparison_preferences(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per comparison (two models over one passage): the team's margin for A over B, and the metric's."""
    compared = rows[rows["kind"] == "comparison"]
    records = []
    for comparison, part in compared.groupby("comparison"):
        labels = part.drop_duplicates("item")["label"]
        means = part.groupby("system")[metric].mean()
        records.append(
            {
                "comparison": comparison,
                "team": part["team"].iloc[0],
                "verses": len(labels),
                "human_margin": ((labels == "A").sum() - (labels == "B").sum()) / len(labels),
                "metric_margin": means.get("A", np.nan) - means.get("B", np.nan),
            }
        )
    return pd.DataFrame(records)


def usability_pairs(rows: pd.DataFrame, metric: str, kind: str) -> PairwiseAgreement:
    """Within each team, every usable draft against every draft the team would start over: is the usable one
    scored higher?"""
    scored = rows[(rows["kind"] == kind) & rows[metric].notna()]
    return PairwiseAgreement.pooled(
        [PairwiseAgreement.all_pairs(part["label"].astype(float), part[metric]) for _, part in scored.groupby("team")]
    )


def analysis_report(rows: pd.DataFrame, judgments: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """For each verse-analysis criterion: within each team, over every two analysed drafts the team rated
    differently, how often the metric scores the better-rated one higher. A marked error type and a larger edit
    proportion count as worse. The "pairs" rows give how many decided pairs each accuracy rests on."""
    analysed = _analysed_verses(rows, judgments)
    worse_when_higher = [label for label in ANALYSIS_LABELS if label.startswith("error_") or label == "edit_proportion"]
    analysed[worse_when_higher] = -analysed[worse_when_higher]
    records = []
    for references, subset in [("independent", analysed[analysed["independent"]]), ("all", analysed)]:
        pairs = {"metric": "pairs", "references": references, "verses": len(subset)}
        for metric in metrics:
            record = {"metric": metric, "references": references, "verses": int(subset[metric].notna().sum())}
            for label in ANALYSIS_LABELS:
                scored = subset[subset[metric].notna() & subset[label].notna()]
                agreement = PairwiseAgreement.pooled(
                    [PairwiseAgreement.all_pairs(p[label], p[metric]) for _, p in scored.groupby("team")]
                )
                record[label] = agreement.accuracy()
                pairs[label] = max(pairs.get(label, 0), agreement.decided_pairs())
            records.append(record)
        records.append(pairs)
    return pd.DataFrame(records)


def _analysed_verses(rows: pd.DataFrame, judgments: pd.DataFrame) -> pd.DataFrame:
    labels = judgments.loc[judgments["kind"] == "analysis", ["item"] + ANALYSIS_LABELS]
    return rows[rows["kind"] == "analysis"].merge(labels, on="item")


def flag_report(rows: pd.DataFrame, judgments: pd.DataFrame) -> pd.DataFrame:
    """Whether a feature's warning picks out the verses teams found a problem in: for each flag and problem, how
    often the problem was marked on flagged and on other verses, the share of marked verses flagged, and a one-sided
    Fisher test. Flags are a detector firing, a key-term mismatch, or a verse in a team's lowest quarter."""
    from scipy.stats import fisher_exact

    verses = _analysed_verses(rows, judgments)

    def lowest_quarter(feature: str) -> pd.Series:
        return verses.groupby("team")[feature].transform(lambda v: v <= v.quantile(0.25)) & verses[feature].notna()

    flags = {f"{f} fires": (verses[f] < 0, verses[f].notna()) for f in DETECTOR_FEATURES if f in verses}
    for feature in ["key-terms", "key-terms-pn"]:
        if feature in verses:
            flags[f"{feature} mismatch"] = (verses[feature] < 1, verses[feature].notna())
    for feature in ["chrF3", "perplexity", "coverage-target", "coverage-reference"]:
        if feature in verses:
            flags[f"{feature} lowest quarter"] = (lowest_quarter(feature), verses[feature].notna())
    problems = {error: verses[error] > 0 for error in ANALYSIS_LABELS if error.startswith("error_")}
    problems.update(
        {
            "meaning differs": verses["same_meaning"] == 0,
            "not clear": verses["clear"] == 0,
            "does not sound right": verses["sounds_right"] == 0,
            "score under 50": verses["score"] < 50,
        }
    )
    records = []
    for flag_name, (flag, applicable) in flags.items():
        for problem_name, problem in problems.items():
            f, p = flag[applicable].to_numpy(bool), problem[applicable].to_numpy(bool)
            if not f.any() or not p.any() or f.all():
                continue
            table = [[np.sum(f & p), np.sum(f & ~p)], [np.sum(~f & p), np.sum(~f & ~p)]]
            records.append(
                {
                    "flag": flag_name,
                    "problem": problem_name,
                    "flagged": int(f.sum()),
                    "verses": len(f),
                    "rate_if_flagged": float(np.mean(p[f])),
                    "rate_otherwise": float(np.mean(p[~f])),
                    "recall": float(np.mean(f[p])),
                    "p": float(fisher_exact(table, alternative="greater")[1]),
                }
            )
    return pd.DataFrame(records)


def agreement_report(rows: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Over all references, and over only those written before the trial (see field_tests.TRIAL_STARTS)."""
    return pd.concat(
        [
            _agreement(rows[rows["independent"]], metrics).assign(references="independent"),
            _agreement(rows, metrics).assign(references="all"),
        ],
        ignore_index=True,
    )


def _agreement(rows: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    records = []
    for metric in metrics:
        verses = verse_preferences(rows, metric)
        interval = verses.bootstrap_accuracy()
        comparisons = comparison_preferences(rows, metric)
        decided = comparisons[comparisons["human_margin"] != 0]
        usability = usability_pairs(rows, metric, "usability")
        human_ai = usability_pairs(rows, metric, "human_ai")
        records.append(
            {
                "metric": metric,
                "verse_preference": verses.accuracy(),
                "verse_preference_lower": interval.lower,
                "verse_preference_upper": interval.upper,
                "verse_pairs": len(verses),
                "comparison_preference": float(
                    np.mean(np.sign(decided["human_margin"]) == np.sign(decided["metric_margin"]))
                ),
                "comparisons": len(decided),
                "usability": usability.accuracy(),
                "human_ai": human_ai.accuracy(),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Agreement of metric features with the field-test judgments")
    parser.add_argument("step", choices=["build", "score", "agree"])
    parser.add_argument("--judgments", type=Path, default=Path("data/field_tests/judgments.tsv"))
    parser.add_argument("--table", type=Path, default=Path("data/field_tests/rows.tsv"))
    parser.add_argument("--features", nargs="+", choices=registry.names(), help="For score: features to add")
    parser.add_argument("--output", type=Path, default=Path("data/results/field_test_agreement.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    table_file = SilverTable(args.table)
    if args.step == "build":
        judgments = pd.read_csv(args.judgments, sep="\t", keep_default_na=False, dtype=str)
        rows = scoring_rows(judgments)
        table_file._save(rows)
        LOGGER.info(f"{len(rows):,} texts to score from {rows['item'].nunique():,} judgments")
    elif args.step == "score":
        if not args.features:
            parser.error("score needs --features")
        table_file.score(args.features, registry)
    else:
        rows = table_file.load()
        metrics = [c for c in rows.columns if c not in TEXT_COLUMNS]
        report = agreement_report(rows, metrics)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, sep="\t", index=False)
        LOGGER.info(f"Agreement with the field-test judgments:\n{report.round(3).to_string(index=False)}")
        judgments = pd.read_csv(args.judgments, sep="\t", low_memory=False).assign(item=lambda t: range(len(t)))
        analyses = analysis_report(rows, judgments, metrics)
        analyses.to_csv(args.output.with_name(args.output.stem + "_analysis.tsv"), sep="\t", index=False)
        LOGGER.info(f"Agreement with the verse analyses:\n{analyses.round(3).to_string(index=False)}")
        flags = flag_report(rows, judgments)
        flags.to_csv(args.output.with_name(args.output.stem + "_flags.tsv"), sep="\t", index=False)
        strong = flags[(flags["p"] < 0.05) & (flags["rate_if_flagged"] > flags["rate_otherwise"])]
        LOGGER.info(f"Flags that pick out marked problems (p < 0.05):\n{strong.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
