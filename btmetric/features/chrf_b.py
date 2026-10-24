"""chrF-B ("chirf-bee", B for Bible): the recommended metric for comparing systems on held-out Bible books (PLAN.md
4.1). chrF3, plus alignment coverage of the hypothesis and key-term consistency, in chrF3 points.

The weights were fit on the field-test draft comparisons (key terms' 20% share is an expert prior) on scores scaled
within each language. Each feature's median spread across the silver test sets and field-test teams turns them into
fixed chrF3 points (coverage 137, key terms 22), which keeps the scaled version's agreement while making scores
comparable between experiments.

Coverage of the hypothesis rises when words with nothing to align to are dropped, so a large coefficient lets it
excuse omissions. Between 30 and 135 the field tests and silver labels can't tell coefficients apart; on the
challenge set, 60 catches 92% of omissions (85% at 135) while still catching 89% of additions and 93% of loops
(chrF3: 49% and 62%), the best mean over error types.
"""

from typing import FrozenSet, Sequence

import numpy as np

from .base import Feature

COVERAGE_TARGET_COEFFICIENT = 60.0
KEY_TERMS_COEFFICIENT = 20.0


class ChrFB(Feature):
    name = "chrF-B"
    criterion = "accuracy"

    def __init__(self, chrf: Feature, coverage_target: Feature, key_terms: Feature) -> None:
        self._chrf = chrf
        self._coverage = coverage_target
        self._key_terms = key_terms

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        self._coverage.prepare(source_language, target_language, excluded_segments)
        self._key_terms.prepare(source_language, target_language, excluded_segments)

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        # A missing coverage or key-term score (no Bible text to align, no key terms in the reference) is the same
        # for every system scored on the verse, so it counts as 0 and leaves chrF3 to decide.
        coverage = np.nan_to_num(self._coverage.segment_scores(sources, hypotheses, references))
        key_terms = np.nan_to_num(self._key_terms.segment_scores(sources, hypotheses, references))
        chrf = self._chrf.segment_scores(sources, hypotheses, references)
        return chrf + COVERAGE_TARGET_COEFFICIENT * coverage + KEY_TERMS_COEFFICIENT * key_terms

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        coverage = self._coverage.segment_scores(sources, hypotheses, references)
        key_terms = self._key_terms.segment_scores(sources, hypotheses, references)
        return (
            self._chrf.system_score(sources, hypotheses, references)
            + COVERAGE_TARGET_COEFFICIENT * _mean_or_zero(coverage)
            + KEY_TERMS_COEFFICIENT * _mean_or_zero(key_terms)
        )

    def close(self) -> None:
        for feature in (self._chrf, self._coverage, self._key_terms):
            feature.close()


def _mean_or_zero(scores: np.ndarray) -> float:
    """The mean over the verses that have a score: key terms as the share of all expected renderings found."""
    return float(np.nanmean(scores)) if not np.isnan(scores).all() else 0.0
