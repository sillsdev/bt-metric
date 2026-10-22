"""Naturalness: how far the hypothesis's character-level perplexity is from the reference's, under a language model
trained only on the target language's Bible (PLAN.md 2.2).

The penalty is two-sided: degenerate output (repetitive, generic) can be more predictable than natural text, so
being less perplexing than the reference earns nothing.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from .alignment import BibleResources
from .base import Feature

LOGGER = logging.getLogger(__name__)


class CharNgramModel:
    """An interpolated Witten-Bell character n-gram model, small and fast enough for one Bible."""

    def __init__(self, order: int = 5) -> None:
        self._order = order
        self._counts: Dict[str, int] = defaultdict(int)
        self._context_totals: Dict[str, int] = defaultdict(int)
        self._context_types: Dict[str, int] = defaultdict(int)
        self._vocabulary_size = 1

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
        """Mean negative log probability per character (natural log)."""
        padded = self._pad(text)
        positions = range(self._order - 1, len(padded))
        return -sum(math.log(self._probability(padded[i - self._order + 1 : i], padded[i])) for i in positions) / max(
            1, len(positions)
        )

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
    name = "perplexity"
    criterion = "naturalness"

    def __init__(self, order: int = 5, resources: Optional[BibleResources] = None) -> None:
        self._order = order
        self._resources = resources or BibleResources()
        self._models: Dict[Tuple[str, FrozenSet[str]], Optional[CharNgramModel]] = {}
        self._model: Optional[CharNgramModel] = None

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        key = (target_language, excluded_segments)
        if key not in self._models:
            self._models[key] = self._train(target_language, excluded_segments)
        self._model = self._models[key]

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._model is None:
            return np.full(len(hypotheses), np.nan)
        return np.array(
            [
                -abs(self._model.log_perplexity(hypothesis) - self._model.log_perplexity(reference))
                for hypothesis, reference in zip(hypotheses, references)
            ]
        )

    def _train(self, target_language: str, excluded: FrozenSet[str]) -> Optional[CharNgramModel]:
        path = self._resources.extract_path(target_language)
        if path is None:
            LOGGER.warning(f"No Bible text for {target_language}; perplexity scores are NaN")
            return None
        vrefs = self._resources.vref_path.read_text(encoding="utf-8").splitlines()
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        verses = [
            line.strip()
            for vref, line in zip(vrefs, lines)
            if line.strip() and line.strip() != "<range>" and vref not in excluded
        ]
        LOGGER.info(f"Training a character {self._order}-gram model on {len(verses)} verses of {path.stem}")
        return CharNgramModel(self._order).train(verses)
