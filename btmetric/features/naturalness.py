"""Naturalness: how far the hypothesis's character-level perplexity is from the reference's, under a language model
trained on text in the domain being scored (PLAN.md 2.2).

The penalty is two-sided: degenerate output (repetitive, generic) can be more predictable than natural text, so
being less perplexing than the reference earns nothing.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, FrozenSet, Optional, Sequence

import numpy as np

from .base import Feature
from .corpora import BibleCorpus, TrainingCorpus

LOGGER = logging.getLogger(__name__)


class CharNgramModel:
    """An interpolated Witten-Bell character n-gram model, small and fast enough for one Bible."""

    def __init__(self, order: int = 5) -> None:
        self._order = order
        self._counts: Dict[str, int] = defaultdict(int)
        self._context_totals: Dict[str, int] = defaultdict(int)
        self._context_types: Dict[str, int] = defaultdict(int)
        self._vocabulary_size = 1
        self._memo: Dict[str, float] = {}

    def train(self, texts: Sequence[str]) -> "CharNgramModel":
        vocabulary = set()
        for text in texts:
            padded = self._pad(text)
            vocabulary.update(padded)
            for i in range(self._order - 1, len(padded)):
                for n in range(1, self._order + 1):
                    context, char = padded[i - n + 1 : i], padded[i]
                    key = context + "\x00" + char
                    if self._counts[key] == 0:
                        self._context_types[context] += 1
                    self._counts[key] += 1
                    self._context_totals[context] += 1
        # One extra symbol stands for characters never seen in training.
        self._vocabulary_size = len(vocabulary) + 1
        return self

    def log_perplexity(self, text: str) -> float:
        """Mean negative log probability per character (natural log). Memoized: a reference is scored once for every
        system compared against it."""
        if text not in self._memo:
            padded = self._pad(text)
            positions = range(self._order - 1, len(padded))
            log_probabilities = (
                math.log(self._probability(padded[i - self._order + 1 : i], padded[i])) for i in positions
            )
            self._memo[text] = -sum(log_probabilities) / max(1, len(positions))
        return self._memo[text]

    def _probability(self, context: str, char: str) -> float:
        probability = 1.0 / self._vocabulary_size
        for n in range(len(context) + 1):
            history = context[len(context) - n :]
            total = self._context_totals.get(history, 0)
            if total == 0:
                continue
            types = self._context_types[history]
            count = self._counts.get(history + "\x00" + char, 0)
            probability = (count + types * probability) / (total + types)
        return probability

    def _pad(self, text: str) -> str:
        # \x02 and \x03 mark the start and end of a verse, so verse-initial and verse-final characters are modelled.
        return "\x02" * (self._order - 1) + text.strip() + "\x03"


class PerplexityRatio(Feature):
    criterion = "naturalness"

    def __init__(
        self,
        order: int = 5,
        corpus: Optional[TrainingCorpus] = None,
        one_sided: bool = False,
        models: Optional[Dict[str, CharNgramModel]] = None,
    ) -> None:
        # One-sided, only being less predictable than the reference is penalized: degenerate output that is more
        # predictable (loops) is left to coverage-target, which catches it directly.
        self.name = "perplexity-one-sided" if one_sided else "perplexity"
        self._one_sided = one_sided
        self._order = order
        self._corpus = corpus or BibleCorpus()
        # Models by the text they were trained on, so calls that would train on the same text share one; the
        # dictionary may be shared between the two variants.
        self._models: Dict[str, CharNgramModel] = models if models is not None else {}
        self._model: Optional[CharNgramModel] = None

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        text = self._corpus.monolingual(target_language, excluded_segments)
        if text is None:
            LOGGER.warning(f"No training text for {target_language}; perplexity scores are NaN")
            self._model = None
            return
        if text.key not in self._models:
            LOGGER.info(f"Training a character {self._order}-gram model on {len(text.lines)} segments ({text.key})")
            self._models[text.key] = CharNgramModel(self._order).train(text.lines)
        self._model = self._models[text.key]

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._model is None:
            return np.full(len(hypotheses), np.nan)
        differences = np.array(
            [
                self._model.log_perplexity(hypothesis) - self._model.log_perplexity(reference)
                for hypothesis, reference in zip(hypotheses, references)
            ]
        )
        return -np.maximum(differences, 0.0) if self._one_sided else -np.abs(differences)
