"""Validates inferred key term renderings against the curated renderings some projects have (PLAN.md 2.2.1).

For each sampled project with Major renderings in MT/terms and a matching extract in MT/scripture, renderings are
inferred from the extract and an English Bible, then compared term by term with the curated ones.
"""

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .inference import AlignedVerses, InferredRendering, RenderingInferrer, specific_renderings
from .major_terms import MajorTermList

LOGGER = logging.getLogger(__name__)


class CuratedRenderings:
    """A project's Major renderings, lowercased, with Paratext's "*" wildcards removed."""

    def __init__(self, renderings: List[Set[str]]) -> None:
        self._renderings = renderings

    @classmethod
    def load(cls, path: Path) -> "CuratedRenderings":
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        return cls([{r.replace("*", "").strip().lower() for r in line.split("\t") if r.strip()} for line in lines])

    def has(self, term: int) -> bool:
        return term < len(self._renderings) and bool(self._renderings[term])

    def matches(self, term: int, rendering: str) -> bool:
        """Equal to a curated rendering, or sharing a stem with one (inflected forms, Paratext's wildcards)."""
        for curated in self._renderings[term] if term < len(self._renderings) else ():
            if rendering == curated:
                return True
            shorter, longer = sorted((rendering, curated), key=len)
            if len(shorter) >= 3 and longer.startswith(shorter) and len(longer) - len(shorter) <= 4:
                return True
        return False


@dataclass(frozen=True)
class TermComparison:
    project: str
    term: int
    category: str
    curated: bool
    inferred: bool
    correct: bool
    confidence: float
    occurrences: int


def compare(
    project: str, terms: MajorTermList, curated: CuratedRenderings, inferred: List[InferredRendering]
) -> List[TermComparison]:
    by_term: Dict[int, InferredRendering] = {rendering.term: rendering for rendering in inferred}
    rows = []
    for term in sorted(set(by_term) | {t for t in range(len(terms)) if curated.has(t) and terms.glosses(t)}):
        rendering = by_term.get(term)
        rows.append(
            TermComparison(
                project=project,
                term=term,
                category=terms.category(term),
                curated=curated.has(term),
                inferred=rendering is not None and bool(rendering.rendering),
                correct=rendering is not None
                and bool(rendering.rendering)
                and curated.matches(term, rendering.rendering),
                confidence=rendering.confidence if rendering else float("nan"),
                occurrences=rendering.occurrences if rendering else 0,
            )
        )
    return rows


def summarize(table: pd.DataFrame) -> str:
    """Precision over terms with both renderings; recall over curated terms whose gloss occurs in the corpus."""
    both = table[table["curated"] & table["inferred"]]
    findable = table[table["curated"] & (table["occurrences"] > 0)]
    lines = [
        f"Projects: {table['project'].nunique()}; curated terms found in English verses: {len(findable)}",
        f"Precision {both['correct'].mean():.3f} on {len(both)} terms; recall {findable['correct'].mean():.3f}",
    ]
    by_category = both.groupby("category").agg(terms=("correct", "size"), precision=("correct", "mean"))
    by_category = by_category.sort_values("terms", ascending=False).head(10)
    lines.append(f"Precision by category (largest 10):\n{by_category.round(3)}")
    bins = pd.cut(both["confidence"], [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
    by_confidence = both.groupby(bins, observed=True).agg(terms=("correct", "size"), precision=("correct", "mean"))
    lines.append(f"Precision by confidence:\n{by_confidence.round(3)}")
    per_project = both.groupby("project")["correct"].mean()
    lines.append(
        f"Precision per project: median {per_project.median():.3f}, "
        f"range {per_project.min():.3f}-{per_project.max():.3f}"
    )
    return "\n\n".join(lines)


class ValidationProjects:
    """Projects with curated Major renderings and a same-named extract covering most of the NT."""

    def __init__(self, terms_dir: Path, scripture_dir: Path) -> None:
        self._terms_dir = terms_dir
        self._scripture_dir = scripture_dir

    def sample(self, count: int, seed: int, exclude_isos: Set[str]) -> List[str]:
        candidates = []
        for path in sorted(self._terms_dir.glob("*-Major-renderings.txt")):
            extract = path.name[: -len("-Major-renderings.txt")]
            if extract.split("-")[0] not in exclude_isos and (self._scripture_dir / f"{extract}.txt").is_file():
                candidates.append(extract)
        rng = np.random.default_rng(seed)
        chosen, isos = [], set()
        for extract in rng.permutation(candidates):
            iso = extract.split("-")[0]
            if iso in isos or not self._covers_nt(extract):
                continue
            if not self._same_characters(extract):
                LOGGER.warning(f"Skipping {extract}: its extract and renderings share few characters (mis-encoded?)")
                continue
            chosen.append(str(extract))
            isos.add(iso)
            if len(chosen) == count:
                break
        return chosen

    def _covers_nt(self, extract: str) -> bool:
        lines = (self._scripture_dir / f"{extract}.txt").read_text(encoding="utf-8-sig").splitlines()
        return sum(1 for line in lines[23213:31170] if line.strip()) >= 5000

    def _same_characters(self, extract: str) -> bool:
        """Whether the renderings' letters occur in the extract. A mis-encoded extract (UTF-8 read as cp1252, say)
        shares almost none, and no inference could match its renderings."""
        renderings = (self._terms_dir / f"{extract}-Major-renderings.txt").read_text(encoding="utf-8-sig")
        text = (self._scripture_dir / f"{extract}.txt").read_text(encoding="utf-8-sig")
        letters = {char for char in renderings if char.isalpha()}
        return not letters or len(letters & set(text)) / len(letters) >= 0.8


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate alignment-inferred key term renderings")
    parser.add_argument("--terms-dir", type=Path, default=Path("M:/MT/terms"))
    parser.add_argument("--scripture-dir", type=Path, default=Path("M:/MT/scripture"))
    parser.add_argument("--assets-dir", type=Path, default=Path("../silnlp/silnlp/assets"))
    parser.add_argument("--english", default="en-BSB")
    parser.add_argument("--projects", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/results/rendering_inference.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    terms = MajorTermList.load(args.assets_dir)
    vref_lines = (args.assets_dir / "vref.txt").read_text(encoding="utf-8").splitlines()
    english_lines = (args.scripture_dir / f"{args.english}.txt").read_text(encoding="utf-8-sig").splitlines()
    projects = ValidationProjects(args.terms_dir, args.scripture_dir).sample(args.projects, args.seed, {"en", "eng"})
    LOGGER.info(f"Validating on {len(projects)} projects: {', '.join(projects)}")
    rows: List[TermComparison] = []
    for project in projects:
        target_lines = (args.scripture_dir / f"{project}.txt").read_text(encoding="utf-8-sig").splitlines()
        verses = AlignedVerses.from_extracts(vref_lines, english_lines, target_lines, excluded=set()).align()
        inferred = specific_renderings(RenderingInferrer(terms).infer(verses), verses, terms)
        curated = CuratedRenderings.load(args.terms_dir / f"{project}-Major-renderings.txt")
        project_rows = compare(project, terms, curated, inferred)
        rows.extend(project_rows)
        both = [row for row in project_rows if row.curated and row.inferred]
        precision = np.mean([row.correct for row in both]) if both else float("nan")
        LOGGER.info(f"{project}: {len(verses)} verses, {len(inferred)} renderings inferred, precision {precision:.3f}")
    table = pd.DataFrame([asdict(row) for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(summarize(table))


if __name__ == "__main__":
    main()
