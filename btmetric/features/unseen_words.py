"""Made-up words: the share of the hypothesis's words never seen in the target language's Bible, minus the reference's.

Low-resource MT invents word forms, and the teams' verse analyses mark non-words where chrF3 is at chance (PLAN.md
3.3g). Real words the Bible lacks (a rare inflection, a modern term) count too; the reference's own share cancels most
of that. With implausible_only, an unseen word counts only when a SentencePiece model of the Bible breaks it into short
fragments, as made-up strings break, while an unseen inflection of a known stem keeps long pieces.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Set

import numpy as np

from .alignment import SubwordTokenizer, Tokenizer
from .base import Feature
from .corpora import BibleCorpus, TrainingCorpus

LOGGER = logging.getLogger(__name__)


@dataclass
class BibleVocabulary:
    tokenizer: Tokenizer
    words: Set[str]
    lines: List[str]
    pieces: Optional[SubwordTokenizer] = None

    def subwords(self) -> SubwordTokenizer:
        if self.pieces is None:
            self.pieces = SubwordTokenizer.train(self.lines, vocab_size=8000)
        return self.pieces


class VocabularyCache:
    """Vocabularies by the text they were built from, shared by the two variants. Only the most recent are kept, as
    each Bible test set leaves out its own verses."""

    def __init__(self, corpus: Optional[TrainingCorpus] = None, capacity: int = 4) -> None:
        self._corpus = corpus or BibleCorpus()
        self._capacity = capacity
        self._vocabularies: "OrderedDict[str, Optional[BibleVocabulary]]" = OrderedDict()

    def get(self, language: str, excluded: FrozenSet[str]) -> Optional[BibleVocabulary]:
        text = self._corpus.monolingual(language, excluded)
        if text is None:
            return None
        if text.key in self._vocabularies:
            self._vocabularies.move_to_end(text.key)
        else:
            tokenizer = Tokenizer.for_corpus(text.lines)
            # Without spaces between words there are no word forms to check.
            self._vocabularies[text.key] = (
                None
                if tokenizer.by_character
                else BibleVocabulary(
                    tokenizer, {token for line in text.lines for token in tokenizer.tokenize(line)}, text.lines
                )
            )
            while len(self._vocabularies) > self._capacity:
                self._vocabularies.popitem(last=False)
        return self._vocabularies[text.key]


class UnseenWords(Feature):
    criterion = "naturalness"

    def __init__(
        self,
        corpus: Optional[TrainingCorpus] = None,
        implausible_only: bool = False,
        min_piece_length: float = 2.5,
        vocabularies: Optional[VocabularyCache] = None,
    ) -> None:
        self.name = "unseen-words-implausible" if implausible_only else "unseen-words"
        self._implausible_only = implausible_only
        self._min_piece_length = min_piece_length
        self._vocabularies = vocabularies or VocabularyCache(corpus)
        self._vocabulary: Optional[BibleVocabulary] = None

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        self._vocabulary = self._vocabularies.get(target_language, excluded_segments)

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._vocabulary is None:
            return np.full(len(hypotheses), np.nan)
        return np.array([self._share(r) - self._share(h) for h, r in zip(hypotheses, references)], dtype=float)

    def _share(self, text: str) -> float:
        words = [token for token in self._vocabulary.tokenizer.tokenize(text) if any(c.isalpha() for c in token)]
        if not words:
            return 0.0
        return sum(self._counts(word) for word in words) / len(words)

    def _counts(self, word: str) -> bool:
        if word in self._vocabulary.words:
            return False
        if not self._implausible_only:
            return True
        pieces = self._vocabulary.subwords().tokenize(word)
        return len(word) / len(pieces) < self._min_piece_length
