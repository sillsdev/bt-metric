"""The other-human-translations test (PLAN.md 3.3b): does a metric score valid human translations above MT?

Other published translations of the same source are scored against the experiment's reference, on the same
verses as the MT systems. A metric that rewards meaning rather than the reference's wording should place every
human translation above most MT outputs; chrF is expected to penalize the freer ones.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..evaluation.data import JudgedOutputs
from ..evaluation.loaders import ExperimentOutputsLoader
from ..meta_eval.runner import FeatureRegistry

LOGGER = logging.getLogger(__name__)


class VerseAlignedExtracts:
    """Scripture extracts with one line per verse of silnlp's vref.txt, looked up by verse reference."""

    def __init__(self, vref_path: Path) -> None:
        lines = vref_path.read_text(encoding="utf-8").splitlines()
        self._line_by_vref = {vref.strip(): i for i, vref in enumerate(lines)}

    def verses(self, extract_path: Path, vrefs: List[str]) -> List[str]:
        lines = extract_path.read_text(encoding="utf-8-sig").splitlines()
        verses = []
        for vref in vrefs:
            i = self._line_by_vref.get(vref.strip())
            text = lines[i].strip() if i is not None and i < len(lines) else ""
            verses.append("" if text == "<range>" else text)
        return verses


class HumanTranslationTest:
    def __init__(self, outputs: JudgedOutputs, alternatives: Dict[str, List[str]]) -> None:
        self._outputs = outputs
        # Only verses every human translation has, so all systems are scored on exactly the same verses.
        present = np.all([[bool(text) for text in verses] for verses in alternatives.values()], axis=0)
        present &= np.array([bool(text.strip()) for text in outputs.evaluation_set.references])
        self._verse_indices = np.flatnonzero(present)
        reference = outputs.evaluation_set.references
        # A human "alternative" identical to the reference would trivially score perfectly.
        self._alternatives = {
            name: verses
            for name, verses in alternatives.items()
            if not all(verses[i] == reference[i] for i in self._verse_indices[:200])
        }

    def sample(self, max_verses: int, seed: int = 0) -> "HumanTranslationTest":
        if len(self._verse_indices) > max_verses:
            chosen = np.random.default_rng(seed).choice(self._verse_indices, size=max_verses, replace=False)
            self._verse_indices = np.sort(chosen)
        return self

    def scores(self, feature_names: List[str], registry: FeatureRegistry) -> pd.DataFrame:
        evaluation_set = self._outputs.evaluation_set
        sources = [evaluation_set.sources[i] for i in self._verse_indices]
        references = [evaluation_set.references[i] for i in self._verse_indices]
        candidates = {("mt", system): hyps for system, hyps in self._outputs.hypotheses.items()}
        candidates.update({("human", name): verses for name, verses in self._alternatives.items()})
        LOGGER.info(f"Scoring {len(candidates)} candidates on {len(self._verse_indices)} verses")
        scored_vrefs = frozenset(evaluation_set.segment_ids[i] for i in self._verse_indices)
        rows = []
        for feature_name in feature_names:
            feature = registry.get(feature_name)
            try:
                # Experiment configs may list several source or target extracts; the first stands for the pair.
                feature.prepare(
                    evaluation_set.source_language.split("+")[0],
                    evaluation_set.target_language.split("+")[0],
                    scored_vrefs,
                )
                for k, ((kind, name), hypotheses) in enumerate(candidates.items(), start=1):
                    LOGGER.info(f"{feature.name}: candidate {k}/{len(candidates)} ({name})")
                    selected = [hypotheses[i] for i in self._verse_indices]
                    score = feature.system_score(sources, selected, references)
                    rows.append({"feature": feature.name, "kind": kind, "candidate": name, "score": score})
            finally:
                feature.close()
        return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """For each feature and human translation: the share of MT systems it outscores, and its rank among all."""
    rows = []
    for feature, group in scores.groupby("feature", sort=False):
        mt_scores = group[group["kind"] == "mt"]["score"].to_numpy()
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        for row in group[group["kind"] == "human"].itertuples():
            rows.append(
                {
                    "feature": feature,
                    "human_translation": row.candidate,
                    "score": row.score,
                    "beats_mt_share": float(np.mean(row.score > mt_scores)),
                    "rank": int(ranked.index[ranked["candidate"] == row.candidate][0]) + 1,
                    "candidates": len(group),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Score other human translations against an experiment reference")
    parser.add_argument("--experiments-root", type=Path, default=Path("M:/MT/experiments"))
    parser.add_argument("--index", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--test-hash", default="4b611a0f04920d6b", help="Default: Hebrew OT -> English NASB")
    parser.add_argument("--scripture", type=Path, default=Path("M:/MT/scripture"))
    parser.add_argument(
        "--alternatives",
        nargs="+",
        default=[
            "en-NASB1995", "en-BSB", "en-CSB", "en-ESVUS16", "en-NIV11R", "en-KJV", "en-NKJV", "en-NET08",
            "en-NRSV", "en-RSV", "en-LSV", "en-NLT07", "en-GNTD", "en-CEVUS06", "en-WEB", "en-YLT98",
        ],
    )  # fmt: skip
    parser.add_argument("--vref", type=Path, default=Path("../silnlp/silnlp/assets/vref.txt"))
    parser.add_argument("--features", nargs="+", default=registry.default_names(), choices=registry.names())
    parser.add_argument("--max-verses", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=Path("data/results/human_translations.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = pd.read_csv(args.index, sep="\t", dtype={"step": str})
    outputs = ExperimentOutputsLoader(args.experiments_root, index).outputs_for(args.test_hash)
    extracts = VerseAlignedExtracts(args.vref)
    vrefs = outputs.evaluation_set.segment_ids
    alternatives = {}
    for name in args.alternatives:
        path = args.scripture / f"{name}.txt"
        if path.is_file():
            alternatives[name] = extracts.verses(path, vrefs)
        else:
            LOGGER.warning(f"No extract {path}; skipping")
    test = HumanTranslationTest(outputs, alternatives).sample(args.max_verses)
    scores = test.scores(args.features, registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, sep="\t", index=False)
    summary = summarize(scores)
    summary.to_csv(args.output.with_name(args.output.stem + "_summary.tsv"), sep="\t", index=False)
    LOGGER.info(f"Share of MT systems each human translation outscores:\n{summary.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
