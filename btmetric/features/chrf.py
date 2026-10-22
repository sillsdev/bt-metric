from typing import Sequence

import numpy as np
from sacrebleu.metrics import CHRF

from .base import Feature


class ChrF(Feature):
    """chrF with silnlp's settings: beta = 3 (recall-weighted), character 6-grams, word n-grams for chrF3++."""

    criterion = "accuracy"

    def __init__(self, word_order: int = 0) -> None:
        self.name = "chrF3" + "+" * word_order
        self._metric = CHRF(char_order=6, word_order=word_order, beta=3)

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        return np.array(
            [
                self._metric.sentence_score(hypothesis, [reference]).score
                for hypothesis, reference in zip(hypotheses, references)
            ]
        )

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        return float(self._metric.corpus_score(list(hypotheses), [list(references)]).score)
