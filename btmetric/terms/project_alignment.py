"""One English-target alignment per project, giving its inferred key term renderings and negation words."""

from pathlib import Path
from typing import List, Optional, Sequence, Set

from .inference import AlignedVerses, RenderingInferrer, specific_renderings
from .major_terms import MajorTermList, ProjectKeyTerms
from .negation import NegationMarkers


class ProjectAlignment:
    def __init__(self, terms: MajorTermList, verses: AlignedVerses) -> None:
        self._terms = terms
        self._verses = verses

    @classmethod
    def build(
        cls,
        terms: MajorTermList,
        vref_lines: Sequence[str],
        english_lines: Sequence[str],
        target_path: Path,
        excluded_vrefs: Set[str],
    ) -> Optional["ProjectAlignment"]:
        if not target_path.is_file():
            return None
        target_lines = target_path.read_text(encoding="utf-8-sig").splitlines()
        verses = AlignedVerses.from_extracts(vref_lines, english_lines, target_lines, excluded_vrefs)
        # Too few verses to train an aligner that says anything useful.
        if len(verses) < 1000:
            return None
        return cls(terms, verses.align())

    def key_terms(self, min_confidence: float) -> ProjectKeyTerms:
        renderings: List[List[str]] = [[] for _ in range(len(self._terms))]
        for inferred in specific_renderings(
            RenderingInferrer(self._terms).infer(self._verses), self._verses, self._terms
        ):
            if inferred.rendering and inferred.confidence >= min_confidence:
                renderings[inferred.term] = [inferred.rendering]
        return ProjectKeyTerms(self._terms, renderings)

    def negation_markers(self) -> List[str]:
        return NegationMarkers().infer(self._verses)
