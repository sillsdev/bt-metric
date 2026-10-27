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

With normalize, chrF3 is rescaled between a floor and 100 (PLAN.md 2.4) so scores read alike across languages: the
floor is chrF3 of the references shuffled among themselves, what a verse in the right language but with the wrong
meaning gets for shared names, punctuation and common words (4 to 26 points across the silver test sets). Within one
test set the floor is a constant, so it only matters when comparing scores across languages or projects.

chrF-B is only validated on scripts that put spaces between words. Thai, Lao, Khmer, Burmese, Tibetan, Shan and
Chinese Bibles space phrases, not words (19 to 44 characters between spaces, against at most 10 elsewhere), which leaves
key terms almost never matched and coverage aligning whole phrases, so scoring them logs a warning.
"""

import logging
from typing import FrozenSet, Sequence

import numpy as np

from .base import Feature

LOGGER = logging.getLogger(__name__)

COVERAGE_TARGET_COEFFICIENT = 60.0
KEY_TERMS_COEFFICIENT = 20.0


FLOOR_SHUFFLES = 3

# Every Bible and test set checked is at most 10 characters per space-separated word or at least 19.
MAX_CHARACTERS_PER_WORD = 15.0


def lacks_word_spaces(texts: Sequence[str]) -> bool:
    """Whether the texts' script doesn't put spaces between words, judged by the characters between spaces."""
    words = sum(len(text.split()) for text in texts)
    characters = sum(len("".join(text.split())) for text in texts)
    return words > 0 and characters / words > MAX_CHARACTERS_PER_WORD


class ChrFB(Feature):
    criterion = "accuracy"

    def __init__(self, chrf: Feature, coverage_target: Feature, key_terms: Feature, normalize: bool = False) -> None:
        self.name = "chrF-B-norm" if normalize else "chrF-B"
        self._chrf = chrf
        self._coverage = coverage_target
        self._key_terms = key_terms
        self._normalize = normalize
        self._target_language = ""
        self._checked_spacing = False

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        self._target_language = target_language
        self._checked_spacing = False
        self._coverage.prepare(source_language, target_language, excluded_segments)
        self._key_terms.prepare(source_language, target_language, excluded_segments)

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        # A missing coverage or key-term score (no Bible text to align, no key terms in the reference) is the same
        # for every system scored on the verse, so it counts as 0 and leaves chrF3 to decide.
        self._check_spacing(references)
        coverage = np.nan_to_num(self._coverage.segment_scores(sources, hypotheses, references))
        key_terms = np.nan_to_num(self._key_terms.segment_scores(sources, hypotheses, references))
        chrf = self._scaled(self._chrf.segment_scores(sources, hypotheses, references), sources, references)
        return chrf + COVERAGE_TARGET_COEFFICIENT * coverage + KEY_TERMS_COEFFICIENT * key_terms

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        self._check_spacing(references)
        coverage = self._coverage.segment_scores(sources, hypotheses, references)
        key_terms = self._key_terms.segment_scores(sources, hypotheses, references)
        return (
            self._scaled(self._chrf.system_score(sources, hypotheses, references), sources, references)
            + COVERAGE_TARGET_COEFFICIENT * _mean_or_zero(coverage)
            + KEY_TERMS_COEFFICIENT * _mean_or_zero(key_terms)
        )

    def floor(self, sources: Sequence[str], references: Sequence[str]) -> float:
        """chrF3 of the references shuffled among themselves (rotated, so none stays in place), averaged over a few
        shuffles. Shuffling within each book instead gives practically the same floor (r = 0.994 on silver)."""
        references = list(references)
        if len(references) < 2:
            return 0.0
        rng = np.random.default_rng(0)
        scores = []
        for _ in range(FLOOR_SHUFFLES):
            shift = 1 + int(rng.integers(len(references) - 1))
            scores.append(self._chrf.system_score(sources, references[shift:] + references[:shift], references))
        return float(np.mean(scores))

    def _check_spacing(self, references: Sequence[str]) -> None:
        # Once per language pair, so scoring many systems doesn't repeat it.
        if self._checked_spacing:
            return
        self._checked_spacing = True
        if lacks_word_spaces(references):
            language = f" for {self._target_language}" if self._target_language else ""
            LOGGER.warning(
                f"{self.name}{language} is unreliable: the references have no word breaks, and chrF-B is only "
                "validated on scripts that have them. chrF3 is the score to use."
            )

    def _scaled(self, chrf, sources: Sequence[str], references: Sequence[str]):
        if not self._normalize:
            return chrf
        floor = self.floor(sources, references)
        return 100 * (chrf - floor) / (100 - floor)

    def close(self) -> None:
        for feature in (self._chrf, self._coverage, self._key_terms):
            feature.close()


def _mean_or_zero(scores: np.ndarray) -> float:
    """The mean over the verses that have a score: key terms as the share of all expected renderings found."""
    return float(np.nanmean(scores)) if not np.isnan(scores).all() else 0.0
