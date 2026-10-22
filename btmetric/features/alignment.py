"""Alignment coverage: how much of the source the hypothesis accounts for, and how much of the hypothesis the
source accounts for.

Unaligned source words point to omissions; unaligned hypothesis words to additions, repetition or untranslated
text. Aligners are trained only on Bible text (PLAN.md 3.2), with symmetrized Thot eflomal. Each score is relative
to the reference's coverage under the same aligner, which cancels out a weak aligner and how differently the two
languages express things.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from machine.tokenization import LatinWordTokenizer
from machine.translation.thot import ThotSymmetrizedWordAlignmentModel

from ..scripture.corpora import EvaluationLanguages
from ..terms.inference import AlignedVerses
from .base import Feature

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


@dataclass(frozen=True)
class BibleResources:
    """Where Bible-only training text comes from. A language is either an extract name ("qaa-ABC_2024_04_22") or an
    ISO code, which resolves to the extract btmetric-find-bible-corpora selected for it."""

    scripture_dirs: Tuple[Path, ...] = (Path("M:/MT/scripture"), Path("M:/eBible/MT/scripture"))
    corpora_table: Path = Path("data/bible_corpora.tsv")
    vref_path: Path = Path("../silnlp/silnlp/assets/vref.txt")

    def extract_path(self, language: str) -> Optional[Path]:
        # Experiment configs don't always name an extract exactly as it is stored: some add a suffix
        # ("qac-ABCD-QAC" for qac-ABCD.txt) and many extracts carry a date ("qab-XYZ_2023_08_22.txt").
        parts = language.split("-")
        names = ["-".join(parts[:end]) for end in range(len(parts), 1, -1)]
        for name in names:
            for directory in self.scripture_dirs:
                if (directory / f"{name}.txt").is_file():
                    return directory / f"{name}.txt"
                dated = sorted(directory.glob(f"{name}_20*.txt"))
                if dated:
                    return dated[-1]
        code = language.split("-")[0]
        evaluation_language = EvaluationLanguages.load().language_for_code(code)
        if evaluation_language is None or not self.corpora_table.is_file():
            return None
        table = pd.read_csv(self.corpora_table, sep="\t")
        selected = table[(table["language"] == evaluation_language) & table["selected"]]
        return Path(selected["path"].iloc[0]) if not selected.empty else None


class BibleAligner:
    """A symmetrized eflomal model trained on two verse-aligned Bible extracts, for aligning new sentence pairs."""

    def __init__(self, model: ThotSymmetrizedWordAlignmentModel, source: Tokenizer, target: Tokenizer) -> None:
        self._model = model
        self._source = source
        self._target = target

    @classmethod
    def train(
        cls, source_path: Path, target_path: Path, resources: BibleResources, excluded_vrefs: FrozenSet[str]
    ) -> Optional["BibleAligner"]:
        vrefs = resources.vref_path.read_text(encoding="utf-8").splitlines()
        source_lines = source_path.read_text(encoding="utf-8-sig").splitlines()
        target_lines = target_path.read_text(encoding="utf-8-sig").splitlines()
        source_tokenizer, target_tokenizer = Tokenizer.for_corpus(source_lines), Tokenizer.for_corpus(target_lines)
        verses = AlignedVerses([], [], [])
        for vref, source, target in zip(vrefs, source_lines, target_lines):
            source, target = source.strip(), target.strip()
            if source and target and "<range>" not in (source, target) and vref not in excluded_vrefs:
                verses.vrefs.append(vref)
                verses.english.append(source_tokenizer.tokenize(source))
                verses.target.append(target_tokenizer.tokenize(target))
        if len(verses.vrefs) < 1000:
            LOGGER.warning(f"Only {len(verses.vrefs)} shared verses in {source_path.name}, {target_path.name}")
            return None
        LOGGER.info(f"Training eflomal on {len(verses.vrefs)} verses of {source_path.stem} / {target_path.stem}")
        return cls(verses.trained_model(), source_tokenizer, target_tokenizer)

    def coverage(
        self, sources: Sequence[str], targets: Sequence[str], weighted: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per pair: how well source tokens are covered by target tokens, and target tokens by source tokens.

        Weighted, each token counts by the best translation probability among its alignments, so a word aligned
        only because the aligner must align something (an untranslated copy of the source, say) counts for
        little. Unweighted, a token counts fully if it is aligned at all.
        """
        pairs = [(self._source.tokenize(s), self._target.tokenize(t)) for s, t in zip(sources, targets)]
        word_pairs_batch = self._model.get_best_aligned_word_pairs_batch(pairs)
        source_coverage, target_coverage = [], []
        for (source_tokens, target_tokens), word_pairs in zip(pairs, word_pairs_batch):
            source_best: Dict[int, float] = {}
            target_best: Dict[int, float] = {}
            for pair in word_pairs:
                if pair.source_index < 0 or pair.target_index < 0:
                    continue
                score = pair.translation_score if weighted else 1.0
                source_best[pair.source_index] = max(source_best.get(pair.source_index, 0.0), score)
                target_best[pair.target_index] = max(target_best.get(pair.target_index, 0.0), score)
            source_coverage.append(sum(source_best.values()) / len(source_tokens) if source_tokens else 0.0)
            target_coverage.append(sum(target_best.values()) / len(target_tokens) if target_tokens else 0.0)
        return np.array(source_coverage), np.array(target_coverage)


class AlignerCache:
    """Bible aligners by language pair and excluded verses, shared by the source- and target-side coverage
    features so each pair is trained once."""

    def __init__(self, resources: Optional[BibleResources] = None) -> None:
        self._resources = resources or BibleResources()
        self._aligners: Dict[Tuple[str, str, FrozenSet[str]], Optional[BibleAligner]] = {}

    def get(self, source_language: str, target_language: str, excluded: FrozenSet[str]) -> Optional[BibleAligner]:
        key = (source_language, target_language, excluded)
        if key not in self._aligners:
            source_path = self._resources.extract_path(source_language)
            target_path = self._resources.extract_path(target_language)
            if source_path is None or target_path is None:
                LOGGER.warning(f"No Bible text for {source_language} or {target_language}; coverage scores are NaN")
                self._aligners[key] = None
            else:
                self._aligners[key] = BibleAligner.train(source_path, target_path, self._resources, excluded)
        return self._aligners[key]


class AlignmentCoverage(Feature):
    """Source coverage ("source", omissions) or hypothesis coverage ("target", additions), minus the reference's."""

    criterion = "accuracy"

    def __init__(
        self, side: str, resources: Optional[BibleResources] = None, aligners: Optional[AlignerCache] = None
    ) -> None:
        if side not in ("source", "target"):
            raise ValueError(f"Unknown coverage side {side!r}.")
        self.name = f"coverage-{side}"
        self._side = side
        self._aligners = aligners or AlignerCache(resources)
        self._aligner: Optional[BibleAligner] = None

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
        return hypothesis_coverage - reference_coverage
