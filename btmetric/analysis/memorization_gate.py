"""The memorization gate for back-translation features (PLAN.md 2.3, 3.3a).

Back-translation models have seen Bible translations in many languages, so they may score a verse's source as
likely whatever the hypothesis says. A feature passes the gate only if errors injected into a human reference
survive: the perturbed reference must score below the clean one. Sources are English (BSB); references are other
Bibles; key terms and negation words are inferred by alignment, as for minority languages.
"""

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..features.base import Feature
from ..features.corpora import BibleCorpus
from ..meta_eval.runner import FeatureRegistry
from ..perturbations.perturbations import VerseContext, all_perturbations
from ..terms.major_terms import MajorTermList
from ..terms.project_alignment import ProjectAlignment

LOGGER = logging.getLogger(__name__)


class PerturbedReferences:
    """A sample of verses with their English source, clean reference, and one perturbed copy per error type."""

    def __init__(self, corpus: BibleCorpus, assets_dir: Path, verses: int, seed: int) -> None:
        self._corpus = corpus
        self._assets_dir = assets_dir
        self._verses = verses
        self._rng = np.random.default_rng(seed)

    def build(self, english: str, target: str) -> pd.DataFrame:
        vrefs = self._corpus.vref_path.read_text(encoding="utf-8").splitlines()
        english_lines = self._read(english)
        target_lines = self._read(target)
        candidates = [
            i
            for i, (e, t) in enumerate(zip(english_lines, target_lines))
            if e.strip() and t.strip() and "<range>" not in (e.strip(), t.strip())
        ]
        chosen = sorted(self._rng.choice(candidates, size=min(self._verses, len(candidates)), replace=False))
        excluded = {vrefs[i] for i in chosen}
        target_path = self._corpus.extract_path(target)
        alignment = ProjectAlignment.build(
            MajorTermList.load(self._assets_dir), vrefs, english_lines, target_path, excluded
        )
        key_terms = alignment.key_terms(0.6) if alignment else None
        negation = tuple(alignment.negation_markers()) if alignment else ()
        LOGGER.info(f"{target}: negation words {list(negation)}")
        rows = []
        perturbations = [p for p in all_perturbations() if p.severity == 1]
        for i in chosen:
            reference = target_lines[i].strip()
            neighbours = tuple(target_lines[j].strip() for j in range(max(0, i - 10), i + 11) if j != i)
            context = VerseContext(
                source=english_lines[i].strip(),
                other_verses=tuple(v for v in neighbours if v and v != "<range>"),
                key_terms=key_terms.occurrences(vrefs[i], reference, self._rng) if key_terms else (),
                negation_markers=negation,
            )
            for perturbation in perturbations:
                perturbed = perturbation.apply(reference, context, self._rng)
                if perturbed and perturbed.strip() and perturbed != reference:
                    rows.append(
                        {
                            "target": target,
                            "vref": vrefs[i],
                            "perturbation": perturbation.name,
                            "source": english_lines[i].strip(),
                            "reference": reference,
                            "perturbed": perturbed,
                        }
                    )
        return pd.DataFrame(rows)

    def _read(self, language: str) -> List[str]:
        path = self._corpus.extract_path(language)
        if path is None:
            raise FileNotFoundError(f"No extract for {language}")
        return path.read_text(encoding="utf-8-sig").splitlines()


def survival(table: pd.DataFrame, feature: Feature, english: str) -> pd.DataFrame:
    """Per target and perturbation: the share of scored verses where the perturbed reference scores lower."""
    results = []
    for target, rows in table.groupby("target"):
        # Trained reverse models must not see the verses being scored.
        feature.prepare(english, target, frozenset(rows["vref"]))
        scores = feature.segment_scores(rows["source"].tolist(), rows["perturbed"].tolist(), rows["reference"].tolist())
        scored = rows.assign(score=scores)
        for perturbation, group in scored.groupby("perturbation"):
            valid = group["score"].notna()
            results.append(
                {
                    "feature": feature.name,
                    "target": target,
                    "perturbation": perturbation,
                    "survives": float((group.loc[valid, "score"] < 0).mean()),
                    "flagged_memorized": float(1 - valid.mean()),
                    "verses": int(valid.sum()),
                }
            )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memorization gate for back-translation scoring")
    parser.add_argument("--english", default="en-BSB")
    parser.add_argument("--targets", nargs="+", default=["de-GCLNR00", "fr-TOB", "ru-Synod"])
    parser.add_argument("--verses", type=int, default=400)
    parser.add_argument("--assets-dir", type=Path, default=Path("../silnlp/silnlp/assets"))
    parser.add_argument(
        "--features", nargs="+", default=["bt-nllb"], choices=["bt-nllb", "bt-nllb-finetuned", "bt-scratch"]
    )
    parser.add_argument("--output", type=Path, default=Path("data/results/memorization_gate.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    builder = PerturbedReferences(BibleCorpus(), args.assets_dir, args.verses, seed=0)
    table = pd.concat([builder.build(args.english, target) for target in args.targets], ignore_index=True)
    LOGGER.info(f"{len(table)} perturbed references:\n{table.groupby(['target', 'perturbation']).size().unstack()}")
    registry = FeatureRegistry()
    results = pd.concat(
        [survival(table, registry.get(name), args.english) for name in args.features], ignore_index=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, sep="\t", index=False)
    pivot = results.pivot_table(index="perturbation", columns=["feature", "target"], values="survives")
    LOGGER.info(f"Share of verses where the injected error lowers the score (survives):\n{pivot.round(3).to_string()}")
    flagged = results.groupby(["feature", "target"])["flagged_memorized"].mean()
    LOGGER.info(f"Share of verses flagged as memorized (info(ref) near zero):\n{flagged.round(3).to_string()}")


if __name__ == "__main__":
    main()
