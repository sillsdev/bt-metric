"""Alignment coverage: how much of the source the hypothesis accounts for, and how much of the hypothesis the
source accounts for.

Unaligned source words point to omissions; unaligned hypothesis words to additions, repetition or untranslated
text. Aligners are symmetrized Thot eflomal models trained on a small amount of text in the domain being scored
(corpora.py). Each score is relative to the reference's coverage under the same aligner, which cancels out a weak
aligner and how differently the two languages express things.
"""

import io
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
from machine.tokenization import LatinWordTokenizer
from machine.translation import SymmetrizationHeuristic
from machine.translation.thot import ThotSymmetrizedWordAlignmentModel

from ..scripture.verses import expand_vref
from ..terms.inference import AlignedVerses
from .base import Feature
from .corpora import BibleCorpus, ParallelText, TrainingCorpus

if TYPE_CHECKING:
    import sentencepiece as spm

LOGGER = logging.getLogger(__name__)


class Tokenizer:
    """Lowercased word tokens, or characters for scripts written without spaces between words (zh, ja, km, ...)."""

    def __init__(self, by_character: bool) -> None:
        self._by_character = by_character
        self._words = LatinWordTokenizer()

    @property
    def by_character(self) -> bool:
        return self._by_character

    @classmethod
    def for_corpus(cls, verses: Sequence[str]) -> "Tokenizer":
        sample = [verse for verse in verses if verse.strip()][:2000]
        with_spaces = sum(1 for verse in sample if " " in verse.strip())
        return cls(by_character=bool(sample) and with_spaces / len(sample) < 0.5)

    def tokenize(self, text: str) -> List[str]:
        if self._by_character:
            return [char for char in text.lower() if not char.isspace()]
        return [token.lower() for token in self._words.tokenize(text)]


class SubwordTokenizer:
    """SentencePiece pieces of lowercased text, from a unigram model trained on the aligner's own training text, so
    inflected forms the aligner never saw share pieces with forms it did."""

    def __init__(self, processor: "spm.SentencePieceProcessor") -> None:
        self._processor = processor

    @classmethod
    def train(cls, lines: Sequence[str], vocab_size: int) -> "SubwordTokenizer":
        import sentencepiece as spm

        model = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter([line.lower() for line in lines if line.strip()]),
            model_writer=model,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            # A small Bible may not support the full vocabulary.
            hard_vocab_limit=False,
            minloglevel=2,
        )
        return cls(spm.SentencePieceProcessor(model_proto=model.getvalue()))

    def tokenize(self, text: str) -> List[str]:
        # A lone word-boundary marker carries no content to align; counting it would dilute coverage.
        return [piece for piece in self._processor.encode(text.lower(), out_type=str) if piece != "▁"]


class WordAligner:
    """A symmetrized eflomal model trained on parallel text, for aligning new sentence pairs.

    The symmetrization heuristic only combines the two directions' alignments, so one trained model serves them all.
    """

    def __init__(
        self,
        model: ThotSymmetrizedWordAlignmentModel,
        source: Union[Tokenizer, SubwordTokenizer],
        target: Union[Tokenizer, SubwordTokenizer],
    ) -> None:
        self._model = model
        self._source = source
        self._target = target
        self._memo: Dict[Tuple[SymmetrizationHeuristic, str, str], Tuple[List[Tuple[int, int, float]], int, int]] = {}

    @classmethod
    def train(
        cls, text: ParallelText, min_pairs: int = 500, subword_vocab: Optional[int] = None
    ) -> Optional["WordAligner"]:
        """subword_vocab: align SentencePiece pieces of this vocabulary size, trained on each side, instead of words."""
        if subword_vocab:
            source_tokenizer: Union[Tokenizer, SubwordTokenizer] = SubwordTokenizer.train(text.source, subword_vocab)
            target_tokenizer: Union[Tokenizer, SubwordTokenizer] = SubwordTokenizer.train(text.target, subword_vocab)
        else:
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
        self,
        sources: Sequence[str],
        targets: Sequence[str],
        weighted: bool = True,
        heuristic: SymmetrizationHeuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per pair: how well source tokens are covered by target tokens, and target tokens by source tokens.

        Weighted, each token counts by the best translation probability among its alignments, so a word aligned
        only because the aligner must align something (an untranslated copy of the source, say) counts for
        little. Unweighted, a token counts fully if it is aligned at all.
        """
        source_coverage, target_coverage = [], []
        for source_best, target_best, source_length, target_length in self._best_scores(
            sources, targets, weighted, heuristic
        ):
            source_coverage.append(sum(source_best.values()) / source_length if source_length else 0.0)
            target_coverage.append(sum(target_best.values()) / target_length if target_length else 0.0)
        return np.array(source_coverage), np.array(target_coverage)

    def source_profiles(self, sources: Sequence[str], targets: Sequence[str]) -> List[Dict[int, float]]:
        """Per pair: each aligned source token's best translation probability."""
        return [
            source_best
            for source_best, _, _, _ in self._best_scores(
                sources, targets, True, SymmetrizationHeuristic.GROW_DIAG_FINAL_AND
            )
        ]

    def crossing_rates(self, sources: Sequence[str], targets: Sequence[str]) -> np.ndarray:
        """Per pair: of every two alignment links that differ on both sides, the share whose order differs between
        source and target. 0 when the target follows the source word order; NaN with fewer than two such links."""
        rates = []
        for word_pairs, _, _ in self._aligned(sources, targets, SymmetrizationHeuristic.GROW_DIAG_FINAL_AND):
            crossed = compared = 0
            for i, (source_i, target_i, _) in enumerate(word_pairs):
                for source_j, target_j, _ in word_pairs[i + 1 :]:
                    if source_i != source_j and target_i != target_j:
                        compared += 1
                        crossed += (source_i < source_j) != (target_i < target_j)
            rates.append(crossed / compared if compared else np.nan)
        return np.array(rates)

    def _best_scores(
        self, sources: Sequence[str], targets: Sequence[str], weighted: bool, heuristic: SymmetrizationHeuristic
    ) -> List[Tuple[Dict[int, float], Dict[int, float], int, int]]:
        alignments = self._aligned(sources, targets, heuristic)
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
        self, sources: Sequence[str], targets: Sequence[str], heuristic: SymmetrizationHeuristic
    ) -> List[Tuple[List[Tuple[int, int, float]], int, int]]:
        """Aligned word pairs with translation scores, memoized: the same source-reference pair is aligned for every
        system scored against it and for every coverage feature."""
        keys = [(heuristic, s, t) for s, t in zip(sources, targets)]
        missing = list(dict.fromkeys(key for key in keys if key not in self._memo))
        if missing:
            self._model.heuristic = heuristic
            tokenized = [(self._source.tokenize(s), self._target.tokenize(t)) for _, s, t in missing]
            for pair, (source_tokens, target_tokens), word_pairs in zip(
                missing, tokenized, self._model.get_best_aligned_word_pairs_batch(tokenized)
            ):
                kept = [
                    (p.source_index, p.target_index, p.translation_score)
                    for p in word_pairs
                    if p.source_index >= 0 and p.target_index >= 0
                ]
                self._memo[pair] = (kept, len(source_tokens), len(target_tokens))
        return [self._memo[key] for key in keys]


class AlignerCache:
    """Aligners by the training text they were trained on, shared by the coverage features so each is trained once.

    Keyed by the corpus's description of its text, so two calls that would train on the same text (e.g. a news
    sample whatever segments are excluded) share one aligner.

    Only the most recent aligners are kept: each Bible test set excludes its own verses, so it trains its own aligner,
    and keeping them all ran a 162-test-set silver run out of memory. Eflomal is deterministic, so retraining an
    evicted one gives the same alignments.
    """

    def __init__(
        self, corpus: Optional[TrainingCorpus] = None, capacity: int = 4, subword_vocab: Optional[int] = None
    ) -> None:
        self._corpus = corpus or BibleCorpus()
        self._capacity = capacity
        self._subword_vocab = subword_vocab
        self._aligners: "OrderedDict[str, Optional[WordAligner]]" = OrderedDict()

    def get(self, source_language: str, target_language: str, excluded: FrozenSet[str]) -> Optional[WordAligner]:
        text = self._corpus.parallel(source_language, target_language, excluded)
        if text is None:
            return None
        if text.key in self._aligners:
            self._aligners.move_to_end(text.key)
        else:
            self._aligners[text.key] = WordAligner.train(text, subword_vocab=self._subword_vocab)
            while len(self._aligners) > self._capacity:
                self._aligners.popitem(last=False)
        return self._aligners[text.key]


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
        heuristic: SymmetrizationHeuristic = SymmetrizationHeuristic.GROW_DIAG_FINAL_AND,
    ) -> None:
        if side not in ("source", "target"):
            raise ValueError(f"Unknown coverage side {side!r}.")
        self.name = f"coverage-{side}" + ("-detector" if detector_tolerance is not None else "")
        self._side = side
        self._detector_tolerance = detector_tolerance
        self._heuristic = heuristic
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
        hypothesis_coverage = self._aligner.coverage(sources, hypotheses, heuristic=self._heuristic)[side]
        reference_coverage = self._aligner.coverage(sources, references, heuristic=self._heuristic)[side]
        if self._detector_tolerance is None:
            return hypothesis_coverage - reference_coverage
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(reference_coverage > 0, hypothesis_coverage / reference_coverage, np.nan)
        return deficit_penalty(ratio, self._detector_tolerance)


class PivotCoverage(Feature):
    """Hypothesis coverage against another Bible's text of each verse (English, or the original languages) instead of
    its source, minus the reference's, so one pivot-target aligner serves every source language.

    The pivot text is found by verse reference, so only verse_scores scores; segment_scores, without references,
    gives NaN.
    """

    criterion = "accuracy"

    def __init__(
        self,
        pivot: str,
        name: str,
        corpus: Optional[BibleCorpus] = None,
        aligners: Optional[AlignerCache] = None,
    ) -> None:
        self.name = name
        self._pivot = pivot
        self._corpus = corpus or BibleCorpus()
        self._aligners = aligners or AlignerCache(self._corpus)
        self._aligner: Optional[WordAligner] = None
        self._pivot_by_verse: Dict[str, str] = {}

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        self._aligner = self._aligners.get(self._pivot, target_language, excluded_segments)
        if self._pivot_by_verse:
            return
        pivot_path = self._corpus.extract_path(self._pivot)
        if pivot_path is None:
            return
        vrefs = self._corpus.vref_path.read_text(encoding="utf-8").splitlines()
        lines = pivot_path.read_text(encoding="utf-8-sig").splitlines()
        self._pivot_by_verse = {
            vref: line.strip() for vref, line in zip(vrefs, lines) if line.strip() not in ("", "<range>", "...")
        }

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        return np.full(len(hypotheses), np.nan)

    def verse_scores(
        self, vrefs: Sequence[str], sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        scores = np.full(len(hypotheses), np.nan)
        if self._aligner is None:
            return scores
        # A verse range takes the pivot text of each of its verses.
        pivot = [
            " ".join(self._pivot_by_verse[v] for v in expand_vref(vref) if v in self._pivot_by_verse) for vref in vrefs
        ]
        found = [i for i, verse in enumerate(pivot) if verse]
        if len(found) < len(vrefs):
            LOGGER.info(f"{self.name}: pivot text found for {len(found)} of {len(vrefs)} verses")
        if found:
            verses = [pivot[i] for i in found]
            hypothesis_coverage = self._aligner.coverage(verses, [hypotheses[i] for i in found])[1]
            reference_coverage = self._aligner.coverage(verses, [references[i] for i in found])[1]
            scores[found] = hypothesis_coverage - reference_coverage
        return scores


class WordOrder(Feature):
    """Naturalness of word order: how far the hypothesis's reordering of the source differs from the reference's.

    MT that copies the source word order (a calque) crosses fewer alignment links than a natural translation, so
    its crossing rate falls below the reference's. With literal_only, only that shortfall counts; otherwise any
    difference from the reference's reordering does, which also catches scrambled order.
    """

    criterion = "naturalness"

    def __init__(
        self,
        literal_only: bool = False,
        corpus: Optional[TrainingCorpus] = None,
        aligners: Optional[AlignerCache] = None,
    ) -> None:
        self.name = "word-order-literal" if literal_only else "word-order"
        self._literal_only = literal_only
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
        difference = self._aligner.crossing_rates(sources, hypotheses) - self._aligner.crossing_rates(
            sources, references
        )
        return np.minimum(difference, 0.0) if self._literal_only else -np.abs(difference)


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
