"""The interface every metric feature implements."""

from abc import ABC, abstractmethod
from typing import FrozenSet, Sequence

import numpy as np


class Feature(ABC):
    """A score for hypotheses against sources and references, higher meaning better.

    `criterion` names which of accuracy, clarity and naturalness the feature measures, so sub-scores can be
    reported per criterion.
    """

    name: str = ""
    criterion: str = ""

    @abstractmethod
    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        """One score per hypothesis."""

    def verse_scores(
        self, vrefs: Sequence[str], sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        """One score per hypothesis, for Scripture whose verse references are known. Features that find other text by
        verse reference override this."""
        return self.segment_scores(sources, hypotheses, references)

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        """The score for a whole system output; features with a corpus-level form (e.g. chrF) override this."""
        return float(np.mean(self.segment_scores(sources, hypotheses, references)))

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        """Called before scoring each evaluation set. Features that need language-specific models (aligners,
        language models) load them here; languages are ISO codes or Scripture extract names such as "qaa-ABC".
        Models trained on Scripture must leave out the excluded segments (verse references), which are the verses
        about to be scored."""

    def close(self) -> None:
        """Releases models or worker processes the feature holds."""
