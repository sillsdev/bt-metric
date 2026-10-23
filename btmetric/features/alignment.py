"""Alignment coverage: how much of the source the hypothesis accounts for, and how much of the hypothesis the
source accounts for.

Unaligned source words point to omissions; unaligned hypothesis words to additions, repetition or untranslated
text. Aligners are symmetrized Thot eflomal models trained on a small amount of text in the domain being scored
(corpora.py). Each score is relative to the reference's coverage under the same aligner, which cancels out a weak
aligner and how differently the two languages express things.
"""

import logging
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from machine.tokenization import LatinWordTokenizer
from machine.translation.thot import ThotSymmetrizedWordAlignmentModel

from ..terms.inference import AlignedVerses
from .base import Feature
from .corpora import BibleCorpus, ParallelText, TrainingCorpus

LOGGER = logging.getLogger(__name__)


class Tokenizer:
    """Lowercased word tokens, or characters for scripts written without spaces between words (zh, ja, km, ...)."""

    def __init__(self, by_character: bool) -> None:
        self._by_character = by_character
        self._words = LatinWordTokenizer()

    @classmethod
    def for_corpus(cls, verses: Sequence[str]) -> "Tokenizer":
        sample = [verse for verse in verses if verse.strip()][:2000]
        with_spaces = sum(1 for verse in sample if " " in verse.strip())
        return cls(by_character=bool(sample) and with_spaces / len(sample) < 0.5)

    def tokenize(self, text: str) -> List[str]:
        if self._by_character:
            return [char for char in text.lower() if not char.isspace()]
        return [token.lower() for token in self._words.tokenize(text)]


class WordAligner:
    """A symmetrized eflomal model trained on parallel text, for aligning new sentence pairs."""

    def __init__(self, model: ThotSymmetrizedWordAlignmentModel, source: Tokenizer, target: Tokenizer) -> None:
        self._model = model
        self._source = source
        self._target = target
        self._memo: Dict[Tuple[str, str], Tuple[List[Tuple[int, int, float]], int, int]] = {}

    @classmethod
    def train(cls, text: ParallelText, min_pairs: int = 500) -> Optional["WordAligner"]:
        source_tokenizer, target_tokenizer = Tokenizer.for_corpus(text.source), Tokenizer.for_corpus(text.target)
        verses = AlignedVerses([], [], [])
        for i, (source, target) in enumerate(zip(text.source, text.target)):
            verses.vrefs.append(str(i))
            verses.english.append(source_tokenizer.tokenize(source))
            verses.target.append(target_tokenizer.tokenize(target))
        if len(verses.vrefs) < min_pairs:
            LOGGER.warning(f"Only {len(verses.vrefs)} segment pairs for {text.key}")
            return None
        LOGGER.info(f"Training eflomal on {len(verses.vrefs)} segment pairs ({text.key})")
        return cls(verses.trained_model(), source_tokenizer, target_tokenizer)

    def coverage(
        self, sources: Sequence[str], targets: Sequence[str], weighted: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per pair: how well source tokens are covered by target tokens, and target tokens by source tokens.

        Weighted, each token counts by the best translation probability among its alignments, so a word aligned
        only because the aligner must align something (an untranslated copy of the source, say) counts for
        little. Unweighted, a token counts fully if it is aligned at all.
        """
        source_coverage, target_coverage = [], []
        for source_best, target_best, source_length, target_length in self._best_scores(sources, targets, weighted):
            source_coverage.append(sum(source_best.values()) / source_length if source_length else 0.0)
            target_coverage.append(sum(target_best.values()) / target_length if target_length else 0.0)
        return np.array(source_coverage), np.array(target_coverage)

    def source_profiles(self, sources: Sequence[str], targets: Sequence[str]) -> List[Dict[int, float]]:
        """Per pair: each aligned source token's best translation probability."""
        return [source_best for source_best, _, _, _ in self._best_scores(sources, targets, weighted=True)]

    def _best_scores(
        self, sources: Sequence[str], targets: Sequence[str], weighted: bool
    ) -> List[Tuple[Dict[int, float], Dict[int, float], int, int]]:
        alignments = self._aligned(sources, targets)
        results = []
        for word_pairs, source_length, target_length in alignments:
            source_best: Dict[int, float] = {}
            target_best: Dict[int, float] = {}
            for source_index, target_index, translation_score in word_pairs:
                score = translation_score if weighted else 1.0
                source_best[source_index] = max(source_best.get(source_index, 0.0), score)
                target_best[target_index] = max(target_best.get(target_index, 0.0), score)
            results.append((source_best, target_best, source_length, target_length))
        return results

    def _aligned(
        self, sources: Sequence[str], targets: Sequence[str]
    ) -> List[Tuple[List[Tuple[int, int, float]], int, int]]:
        """Aligned word pairs with translation scores, memoized: the same source-reference pair is aligned for every
        system scored against it and for every coverage feature."""
        missing = list(dict.fromkeys(pair for pair in zip(sources, targets) if pair not in self._memo))
        if missing:
            tokenized = [(self._source.tokenize(s), self._target.tokenize(t)) for s, t in missing]
            for pair, (source_tokens, target_tokens), word_pairs in zip(
                missing, tokenized, self._model.get_best_aligned_word_pairs_batch(tokenized)
            ):
                kept = [
                    (p.source_index, p.target_index, p.translation_score)
                    for p in word_pairs
                    if p.source_index >= 0 and p.target_index >= 0
                ]
                self._memo[pair] = (kept, len(source_tokens), len(target_tokens))
        return [self._memo[pair] for pair in zip(sources, targets)]


class AlignerCache:
    """Aligners by the training text they were trained on, shared by the coverage features so each is trained once.

    Keyed by the corpus's description of its text, so two calls that would train on the same text (e.g. a news
    sample whatever segments are excluded) share one aligner.
    """

    def __init__(self, corpus: Optional[TrainingCorpus] = None) -> None:
        self._corpus = corpus or BibleCorpus()
        self._aligners: Dict[str, Optional[WordAligner]] = {}
        self._keys: Dict[Tuple[str, str, FrozenSet[str]], Optional[str]] = {}

    def get(self, source_language: str, target_language: str, excluded: FrozenSet[str]) -> Optional[WordAligner]:
        request = (source_language, target_language, excluded)
        if request not in self._keys:
            text = self._corpus.parallel(source_language, target_language, excluded)
            self._keys[request] = text.key if text else None
            if text and text.key not in self._aligners:
                self._aligners[text.key] = WordAligner.train(text)
        key = self._keys[request]
        return self._aligners[key] if key else None


def deficit_penalty(ratio: np.ndarray, tolerance: float) -> np.ndarray:
    """0 while the hypothesis keeps at least (1 - tolerance) of the reference's coverage, then the shortfall beyond
    that as a penalty.

    Used as an error detector: within the normal range, more coverage only means following the source more
    literally, which human judgments don't reward (PLAN.md 2.2.2), so only gross shortfalls (dropped clauses,
    inserted passages, loops) count.
    """
    return -np.maximum(0.0, (1.0 - ratio) - tolerance)


class AlignmentCoverage(Feature):
    """Source coverage ("source", omissions) or hypothesis coverage ("target", additions), minus the reference's.

    With a detector tolerance, the score is instead deficit_penalty of the hypothesis's coverage as a ratio of the
    reference's.
    """

    criterion = "accuracy"

    def __init__(
        self,
        side: str,
        corpus: Optional[TrainingCorpus] = None,
        aligners: Optional[AlignerCache] = None,
        detector_tolerance: Optional[float] = None,
    ) -> None:
        if side not in ("source", "target"):
            raise ValueError(f"Unknown coverage side {side!r}.")
        self.name = f"coverage-{side}" + ("-detector" if detector_tolerance is not None else "")
        self._side = side
        self._detector_tolerance = detector_tolerance
        self._aligners = aligners or AlignerCache(corpus)
        self._aligner: Optional[WordAligner] = None

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        self._aligner = self._aligners.get(source_language, target_language, excluded_segments)

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._aligner is None:
            return np.full(len(hypotheses), np.nan)
        side = 0 if self._side == "source" else 1
        hypothesis_coverage = self._aligner.coverage(sources, hypotheses)[side]
        reference_coverage = self._aligner.coverage(sources, references)[side]
        if self._detector_tolerance is None:
            return hypothesis_coverage - reference_coverage
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(reference_coverage > 0, hypothesis_coverage / reference_coverage, np.nan)
        return deficit_penalty(ratio, self._detector_tolerance)


class ReferenceAnchoredCoverage(Feature):
    """Omissions measured only against what the reference expresses.

    Plain source coverage rewards literal translation: a meaning-based translation restructures the source and
    leaves source words unaligned, so it scores low even when nothing is missing. Here a source word counts only
    as far as the reference expresses it, and the score is the share of that the hypothesis also expresses. The
    hypothesis is never asked to be more literal than the reference.
    """

    criterion = "accuracy"

    def __init__(
        self,
        corpus: Optional[TrainingCorpus] = None,
        aligners: Optional[AlignerCache] = None,
        detector_tolerance: Optional[float] = None,
    ) -> None:
        self.name = "coverage-reference" + ("-detector" if detector_tolerance is not None else "")
        self._detector_tolerance = detector_tolerance
        self._aligners = aligners or AlignerCache(corpus)
        self._aligner: Optional[WordAligner] = None

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        self._aligner = self._aligners.get(source_language, target_language, excluded_segments)

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._aligner is None:
            return np.full(len(hypotheses), np.nan)
        scores = []
        reference_profiles = self._aligner.source_profiles(sources, references)
        hypothesis_profiles = self._aligner.source_profiles(sources, hypotheses)
        for expressed, covered in zip(reference_profiles, hypothesis_profiles):
            total = sum(expressed.values())
            kept = sum(min(covered.get(i, 0.0), weight) for i, weight in expressed.items())
            scores.append(kept / total if total > 0 else np.nan)
        kept_share = np.array(scores)
        if self._detector_tolerance is None:
            return kept_share
        return deficit_penalty(kept_share, self._detector_tolerance)
