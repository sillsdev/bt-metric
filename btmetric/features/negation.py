"""Negation consistency: whether the hypothesis has as many negation words as the reference.

A dropped or added negation reverses a verse's meaning in a word or two, which chrF3 and coverage barely weigh: removed
negations are where chrF-B falls furthest behind chrF3 (PLAN.md 4.1.1). Negation words are inferred by aligning the
English Bible with the target's (btmetric/terms/negation.py), so languages that negate with affixes, where none are
found, get no score (NaN).
"""

from typing import FrozenSet, Optional, Sequence

import numpy as np
from machine.translation import SymmetrizationHeuristic

from .alignment import Tokenizer
from .base import Feature
from .corpora import BibleCorpus
from .key_terms import EnglishBible, InferredRenderings


class NegationConsistency(Feature):
    """0 when the hypothesis has as many negation words as the reference, -1 when it has more or fewer.

    missing_only penalizes only fewer. MT output often has a negation word more than the reference already (a
    different phrasing, or an inferred word that isn't only a negator), and counting that as an error rewards deleting
    a negation that brings the counts level.
    """

    criterion = "accuracy"

    def __init__(
        self,
        resources: Optional[BibleCorpus] = None,
        english: EnglishBible = EnglishBible(),
        renderings: Optional[InferredRenderings] = None,
        heuristic: SymmetrizationHeuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND,
        missing_only: bool = False,
    ) -> None:
        self.name = "negation-missing" if missing_only else "negation"
        self._missing_only = missing_only
        self._renderings = renderings or InferredRenderings(resources, english)
        self._heuristic = heuristic
        self._tokenizer = Tokenizer(by_character=False)
        self._markers: Optional[FrozenSet[str]] = None

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        markers = self._renderings.negation_markers(target_language, excluded_segments, self._heuristic)
        self._markers = frozenset(markers) if markers else None

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._markers is None:
            return np.full(len(hypotheses), np.nan)
        return np.array([-float(self._mismatch(h, r)) for h, r in zip(hypotheses, references)], dtype=float)

    def _mismatch(self, hypothesis: str, reference: str) -> bool:
        hypothesis_count, reference_count = self._count(hypothesis), self._count(reference)
        return hypothesis_count < reference_count if self._missing_only else hypothesis_count != reference_count

    def _count(self, text: str) -> int:
        return sum(token in self._markers for token in self._tokenizer.tokenize(text))
