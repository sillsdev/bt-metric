"""Biblical key terms per verse, from Paratext's Major Biblical Terms list and a project's renderings of it."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..perturbations.perturbations import KeyTermOccurrence


class MajorTermList:
    """silnlp's Major-metadata.txt (term, Paratext category, semantic domain), Major-vrefs.txt and the English
    glosses in en-Major-glosses.txt, all line-aligned."""

    def __init__(self, kinds: List[str], vrefs: List[List[str]], glosses: Optional[List[List[str]]] = None) -> None:
        self._kinds = kinds
        self._vrefs = vrefs
        self._glosses = glosses or [[] for _ in kinds]
        self._terms_by_vref: Dict[str, List[int]] = defaultdict(list)
        for term, term_vrefs in enumerate(vrefs):
            for vref in term_vrefs:
                self._terms_by_vref[vref].append(term)

    @classmethod
    def load(cls, assets_dir: Path, gloss_language: str = "en") -> "MajorTermList":
        metadata = (assets_dir / "Major-metadata.txt").read_text(encoding="utf-8").splitlines()
        vrefs = (assets_dir / "Major-vrefs.txt").read_text(encoding="utf-8").splitlines()
        # A term's kind for substitution: its Paratext category plus first semantic domain, e.g. "PN|person".
        kinds = []
        for line in metadata:
            fields = line.split("\t")
            category = fields[1] if len(fields) > 1 else ""
            domain = fields[2].split(";")[0].strip() if len(fields) > 2 else ""
            kinds.append(f"{category}|{domain}")
        glosses_path = assets_dir / f"{gloss_language}-Major-glosses.txt"
        glosses = None
        if glosses_path.is_file():
            lines = glosses_path.read_text(encoding="utf-8").splitlines()
            glosses = [[gloss.strip() for gloss in line.split("\t") if gloss.strip()] for line in lines]
            glosses += [[] for _ in range(len(kinds) - len(glosses))]
        return cls(kinds, [line.split("\t") if line else [] for line in vrefs], glosses)

    def terms_in(self, vref: str) -> List[int]:
        return self._terms_by_vref.get(vref, [])

    def vrefs(self, term: int) -> List[str]:
        return self._vrefs[term]

    def glosses(self, term: int) -> List[str]:
        return self._glosses[term]

    def kind(self, term: int) -> str:
        return self._kinds[term]

    def category(self, term: int) -> str:
        return self._kinds[term].split("|")[0]

    def __len__(self) -> int:
        return len(self._kinds)


class ProjectKeyTerms:
    """A project's renderings of the Major terms, used to find terms in a verse and plausible substitutes."""

    def __init__(self, terms: MajorTermList, renderings: List[List[str]], max_alternatives: int = 20) -> None:
        self._terms = terms
        self._renderings = renderings
        self._max_alternatives = max_alternatives
        self._renderings_by_kind: Dict[str, Set[str]] = defaultdict(set)
        for term, term_renderings in enumerate(renderings):
            self._renderings_by_kind[terms.kind(term)].update(term_renderings)

    @classmethod
    def load(cls, terms: MajorTermList, renderings_path: Path) -> "ProjectKeyTerms":
        lines = renderings_path.read_text(encoding="utf-8-sig").splitlines()
        renderings = []
        for line in lines[: len(terms)]:
            # Paratext marks morphological variation with "*"; the fixed part is what can be found in a verse.
            cleaned = {rendering.replace("*", "").strip() for rendering in line.split("\t")}
            renderings.append(sorted(rendering for rendering in cleaned if len(rendering) >= 2))
        renderings += [[] for _ in range(len(terms) - len(renderings))]
        return cls(terms, renderings)

    def occurrences(self, vref: str, text: str, rng: np.random.Generator) -> Tuple[KeyTermOccurrence, ...]:
        found = []
        lowered = text.lower()
        for term in self._terms.terms_in(vref):
            surface = next((s for s in (self._surface(text, lowered, r) for r in self._renderings[term]) if s), None)
            if surface is None:
                continue
            own = {r.lower() for r in self._renderings[term]}
            alternatives = sorted(
                r
                for r in self._renderings_by_kind[self._terms.kind(term)]
                if r.lower() not in own and r.lower() not in lowered
            )
            if len(alternatives) > self._max_alternatives:
                alternatives = list(rng.choice(alternatives, size=self._max_alternatives, replace=False))
            found.append(KeyTermOccurrence(surface, tuple(alternatives)))
        return tuple(found)

    def _surface(self, text: str, lowered: str, rendering: str) -> Optional[str]:
        """The rendering as written in the text, found ignoring case (inferred renderings are lowercase)."""
        if rendering in text:
            return rendering
        # Lowercasing can change a string's length in a few scripts, and then positions don't carry over.
        if len(lowered) != len(text):
            return None
        start = lowered.find(rendering.lower())
        return text[start : start + len(rendering)] if start >= 0 else None


def find_renderings_file(terms_dir: Path, target_extract: str) -> Optional[Path]:
    """The Major renderings for an extract such as "qaa-ABC_2023_11_30", or the latest for the same project."""
    exact = terms_dir / f"{target_extract}-Major-renderings.txt"
    if exact.is_file():
        return exact
    # Only dated copies of the same project: "qaa-ABC_2024_04_22", not another project such as "qaa-ABC_BTE".
    project = target_extract.split("_20")[0]
    candidates = sorted(terms_dir.glob(f"{project}_20*-Major-renderings.txt"))
    undated = terms_dir / f"{project}-Major-renderings.txt"
    if not candidates and undated.is_file():
        return undated
    return candidates[-1] if candidates else None
