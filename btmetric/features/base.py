"""The interface every metric feature implements."""

from abc import ABC, abstractmethod
from typing import Sequence

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

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        """The score for a whole system output; features with a corpus-level form (e.g. chrF) override this."""
        return float(np.mean(self.segment_scores(sources, hypotheses, references)))

    def close(self) -> None:
        """Releases models or worker processes the feature holds."""
