"""Surface-overlap baselines with the settings silnlp's scorers use."""

from typing import Sequence

import numpy as np
from sacrebleu.metrics import BLEU, TER

from .base import Feature
from .chrf import ChrF


class Bleu(Feature):
    """Lowercased BLEU; spBLEU uses the FLORES-200 SentencePiece tokenizer instead of 13a."""

    criterion = "accuracy"

    def __init__(self, tokenize: str = "13a") -> None:
        self.name = "spBLEU" if tokenize == "flores200" else "BLEU"
        self._metric = BLEU(lowercase=True, tokenize=tokenize, effective_order=True)
        self._corpus_metric = BLEU(lowercase=True, tokenize=tokenize)

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
        return float(self._corpus_metric.corpus_score(list(hypotheses), [list(references)]).score)


class NegativeTer(Feature):
    """TER negated, so that higher is better like every other feature."""

    name = "TER"
    criterion = "accuracy"

    def __init__(self) -> None:
        self._metric = TER()

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        return -np.array(
            [
                self._metric.sentence_score(hypothesis, [reference]).score
                for hypothesis, reference in zip(hypotheses, references)
            ]
        )

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        return -float(self._metric.corpus_score(list(hypotheses), [list(references)]).score)


class MeanChrF(ChrF):
    """m-chrF3 (arXiv 2407.12832): verse-level chrF3 averaged over the test set, instead of corpus chrF3."""

    def __init__(self, word_order: int = 0) -> None:
        super().__init__(word_order)
        self.name = "m-" + self.name

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        return float(np.mean(self.segment_scores(sources, hypotheses, references)))
