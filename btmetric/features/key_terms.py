"""Key term accuracy: the share of the reference's key term renderings that the hypothesis also uses.

Renderings are inferred by aligning an English Bible with the target language's Bible (PLAN.md 2.2.1), since most
projects have no curated renderings. Segments whose reference contains no key term get no score (NaN).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..terms.inference import AlignedVerses, InferredRendering, RenderingInferrer, specific_renderings
from ..terms.major_terms import MajorTermList
from .alignment import BibleResources, Tokenizer
from .base import Feature

LOGGER = logging.getLogger(__name__)


class RenderingMatcher:
    """Finds renderings in tokenized text, tolerating suffixes on each token (inflected forms)."""

    def __init__(self, renderings: Set[Tuple[str, ...]]) -> None:
        # Indexed by a fixed-length prefix, which an inflected form shares with its rendering whatever the suffix.
        self._by_prefix: Dict[str, List[Tuple[str, ...]]] = {}
        for rendering in renderings:
            self._by_prefix.setdefault(rendering[0][:3], []).append(rendering)

    def found(self, tokens: List[str]) -> Set[Tuple[str, ...]]:
        found = set()
        for start, token in enumerate(tokens):
            for rendering in self._by_prefix.get(token[:3], ()):
                window = tokens[start : start + len(rendering)]
                if len(window) == len(rendering) and all(self._matches(t, r) for t, r in zip(window, rendering)):
                    found.add(rendering)
        return found

    def contains(self, tokens: List[str], rendering: Tuple[str, ...]) -> bool:
        return any(
            all(self._matches(t, r) for t, r in zip(tokens[start : start + len(rendering)], rendering))
            for start in range(len(tokens) - len(rendering) + 1)
        )

    def _matches(self, token: str, rendering_token: str) -> bool:
        # Short renderings must match whole and take few extra letters: their letters are often an inflectional
        # prefix shared by many words (Swahili "aka-"). Longer ones may lose two final letters and add three.
        length = len(rendering_token)
        stem = rendering_token if length <= 5 else rendering_token[:-2]
        extra = 0 if length <= 3 else 2 if length <= 5 else 3
        return token.startswith(stem) and len(token) - length <= extra


@dataclass(frozen=True)
class EnglishBible:
    extract: str = "en-BSB"
    assets_dir: Path = Path("../silnlp/silnlp/assets")


class InferredRenderings:
    """Inferred, specificity-filtered renderings by target language and excluded verses, shared by the key term
    features so each language is aligned once whatever categories a feature keeps."""

    def __init__(self, resources: Optional[BibleResources] = None, english: EnglishBible = EnglishBible()) -> None:
        self._resources = resources or BibleResources()
        self._english = english
        self._terms: Optional[MajorTermList] = None
        self._inferred: Dict[Tuple[str, FrozenSet[str]], Optional[List[InferredRendering]]] = {}

    def get(self, target_language: str, excluded: FrozenSet[str]) -> Optional[List[InferredRendering]]:
        key = (target_language, excluded)
        if key not in self._inferred:
            self._inferred[key] = self._infer(target_language, excluded)
        return self._inferred[key]

    def _infer(self, target_language: str, excluded: FrozenSet[str]) -> Optional[List[InferredRendering]]:
        english_path = self._resources.extract_path(self._english.extract)
        target_path = self._resources.extract_path(target_language)
        if english_path is None or target_path is None:
            LOGGER.warning(f"No Bible text for {target_language}; key term scores are NaN")
            return None
        if self._terms is None:
            self._terms = MajorTermList.load(self._english.assets_dir)
        vrefs = self._resources.vref_path.read_text(encoding="utf-8").splitlines()
        target_lines = target_path.read_text(encoding="utf-8-sig").splitlines()
        english_lines = english_path.read_text(encoding="utf-8-sig").splitlines()
        verses = AlignedVerses.from_extracts(vrefs, english_lines, target_lines, set(excluded))
        if len(verses) < 1000:
            return None
        LOGGER.info(f"Inferring key term renderings for {target_path.stem} from {len(verses)} verses")
        return specific_renderings(RenderingInferrer(self._terms).infer(verses.align()), verses, self._terms)


class KeyTermAccuracy(Feature):
    criterion = "accuracy"

    def __init__(
        self,
        categories: Optional[FrozenSet[str]] = None,
        min_confidence: float = 0.6,
        resources: Optional[BibleResources] = None,
        english: EnglishBible = EnglishBible(),
        renderings: Optional[InferredRenderings] = None,
    ) -> None:
        self.name = "key-terms" if categories is None else f"key-terms-{'-'.join(sorted(categories)).lower()}"
        self._categories = categories
        self._min_confidence = min_confidence
        self._renderings = renderings or InferredRenderings(resources, english)
        self._matcher: Optional[Tuple[Tokenizer, RenderingMatcher]] = None

    def prepare(
        self, source_language: str, target_language: str, excluded_segments: FrozenSet[str] = frozenset()
    ) -> None:
        inferred = self._renderings.get(target_language, excluded_segments)
        if inferred is None:
            self._matcher = None
            return
        kept = {
            tuple(rendering.rendering.split())
            for rendering in inferred
            if rendering.rendering
            and rendering.confidence >= self._min_confidence
            and (self._categories is None or rendering.category in self._categories)
        }
        # Renderings are matched in the same tokenization the aligner saw.
        self._matcher = (Tokenizer(by_character=False), RenderingMatcher(kept))

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        if self._matcher is None:
            return np.full(len(hypotheses), np.nan)
        tokenizer, matcher = self._matcher
        scores = []
        for hypothesis, reference in zip(hypotheses, references):
            expected = matcher.found(tokenizer.tokenize(reference))
            hypothesis_tokens = tokenizer.tokenize(hypothesis)
            scores.append(
                np.mean([matcher.contains(hypothesis_tokens, rendering) for rendering in expected])
                if expected
                else np.nan
            )
        return np.array(scores, dtype=float)

    def system_score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        scores = self.segment_scores(sources, hypotheses, references)
        return float(np.nanmean(scores)) if not np.isnan(scores).all() else float("nan")
