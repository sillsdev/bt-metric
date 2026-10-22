"""Controlled, meaning-changing errors injected into a translation, for testing what each metric notices.

Every perturbation works on any language: text is handled as words where the script separates them with spaces and
as characters where it doesn't, and clause boundaries come from Unicode punctuation.
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class KeyTermOccurrence:
    """A key term's rendering found in a verse, with renderings of other terms of the same kind to swap in."""

    rendering: str
    alternatives: Tuple[str, ...]


@dataclass(frozen=True)
class VerseContext:
    """What a perturbation may draw on besides the text itself."""

    source: str = ""
    other_verses: Tuple[str, ...] = ()
    key_terms: Tuple[KeyTermOccurrence, ...] = field(default_factory=tuple)
    negation_markers: Tuple[str, ...] = ()


class Units:
    """A text split into words, or into characters for scripts written without spaces between words."""

    def __init__(self, text: str) -> None:
        words = text.split()
        self._by_word = len(words) >= 2
        self.items: List[str] = words if self._by_word else list(text)

    def separator(self) -> str:
        return " " if self._by_word else ""

    def join(self, items: Sequence[str]) -> str:
        return self.separator().join(items)


class Perturbation(ABC):
    """An error type. apply() returns None when the error can't be made in this verse."""

    name: str = ""

    def __init__(self, severity: int = 1) -> None:
        # How much of the error to inject, for graded perturbations: 1 is mild, 2 is severe.
        self.severity = severity

    @abstractmethod
    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        """The text with this error injected."""


class ClauseOmission(Perturbation):
    """Drops one clause (two when severe), or a contiguous quarter of the verse (half when severe) when it has no
    internal punctuation."""

    name = "omission"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        clauses = _split_clauses(text)
        if len(clauses) > self.severity:
            dropped = set(rng.choice(len(clauses), size=self.severity, replace=False).tolist())
            return "".join(clause for i, clause in enumerate(clauses) if i not in dropped).strip()
        units = Units(text)
        span = max(1, len(units.items) * self.severity // 4)
        if len(units.items) <= span:
            return None
        start = int(rng.integers(len(units.items) - span + 1))
        return units.join(units.items[:start] + units.items[start + span :])


class Addition(Perturbation):
    """Inserts a clause from another verse (two when severe): fluent text with content the source doesn't have."""

    name = "addition"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        donors = [verse for verse in context.other_verses if verse.strip() and verse != text]
        if not donors:
            return None
        clauses = _split_clauses(text)
        for _ in range(self.severity):
            donor_clauses = _split_clauses(donors[int(rng.integers(len(donors)))])
            inserted = donor_clauses[int(rng.integers(len(donor_clauses)))].strip()
            position = int(rng.integers(len(clauses) + 1))
            clauses = [*clauses[:position], inserted, *clauses[position:]]
        return Units(text).separator().join(clauses).strip()


class KeyTermSubstitution(Perturbation):
    """Replaces a key term's rendering with the rendering of a different term of the same kind (two terms when
    severe)."""

    name = "key_term"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        candidates = [term for term in context.key_terms if term.alternatives and term.rendering in text]
        if len(candidates) < self.severity:
            return None
        for k in rng.choice(len(candidates), size=self.severity, replace=False):
            term = candidates[int(k)]
            if term.rendering in text:
                text = text.replace(term.rendering, term.alternatives[int(rng.integers(len(term.alternatives)))], 1)
        return text


class NegationRemoval(Perturbation):
    """Deletes a negation word, so a negative statement becomes positive (or a command its opposite)."""

    name = "negation"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        words = text.split()
        markers = {marker.lower() for marker in context.negation_markers}
        # Words are compared without surrounding punctuation, which stays in the text if the word is removed.
        positions = [i for i, word in enumerate(words) if _strip_punctuation(word).lower() in markers]
        if not positions:
            return None
        i = positions[int(rng.integers(len(positions)))]
        before = words[:i]
        punctuation = "".join(char for char in words[i] if unicodedata.category(char).startswith("P"))
        if punctuation and before:
            before[-1] += punctuation
        return " ".join(before + words[i + 1 :]) or None


class LocalShuffle(Perturbation):
    """Reorders words (or characters) within small windows, of 2 or (severe) 4 units: the content stays, the grammar
    breaks."""

    name = "word_order"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        units = Units(text)
        if len(units.items) < 2:
            return None
        window = 2 * self.severity
        items = list(units.items)
        for start in range(0, len(items), window):
            chunk = items[start : start + window]
            items[start : start + window] = [chunk[i] for i in rng.permutation(len(chunk))]
        shuffled = units.join(items)
        return shuffled if shuffled != text else None


class WrongVerse(Perturbation):
    """Replaces the verse with another verse: fluent and on-topic, but the wrong meaning."""

    name = "wrong_verse"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        others = [verse for verse in context.other_verses if verse.strip() and verse != text]
        return others[int(rng.integers(len(others)))] if others else None


class SourceCopy(Perturbation):
    """Returns the source untranslated, a common failure of low-resource MT."""

    name = "source_copy"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        return context.source if context.source.strip() and context.source != text else None


class Repetition(Perturbation):
    """Repeats a span back to back, once more (mild) or four more times (severe), like a decoder loop."""

    name = "repetition"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        units = Units(text)
        if len(units.items) < 2:
            return None
        span = max(1, min(4, len(units.items) // 4))
        start = int(rng.integers(len(units.items) - span + 1))
        repeated = units.items[start : start + span] * (1 if self.severity == 1 else 4)
        return units.join(units.items[: start + span] + repeated + units.items[start + span :])


class Truncation(Perturbation):
    """Cuts off the last quarter of the verse, or (severe) the second half."""

    name = "truncation"

    def apply(self, text: str, context: VerseContext, rng: np.random.Generator) -> Optional[str]:
        units = Units(text)
        if len(units.items) < 2:
            return None
        kept = len(units.items) * (3 if self.severity == 1 else 2) // 4
        return units.join(units.items[: max(1, kept)]) if kept < len(units.items) else None


def all_perturbations() -> List[Perturbation]:
    """Every perturbation type; graded types at both severities, so metrics can be checked for grading errors."""
    graded = [ClauseOmission, Addition, KeyTermSubstitution, LocalShuffle, Repetition, Truncation]
    ungraded = [NegationRemoval(), WrongVerse(), SourceCopy()]
    return [cls(severity) for cls in graded for severity in (1, 2)] + ungraded


def _strip_punctuation(word: str) -> str:
    return "".join(char for char in word if not unicodedata.category(char).startswith("P"))


def _split_clauses(text: str) -> List[str]:
    """Clauses with their trailing punctuation kept, split after clause-ending Unicode punctuation."""
    pieces, start = [], 0
    for match in re.finditer(r"\S", text):
        char = match.group()
        if _ends_clause(char) and match.end() < len(text.rstrip()):
            pieces.append(text[start : match.end()])
            start = match.end()
    pieces.append(text[start:])
    return [piece for piece in pieces if piece.strip()]


def _ends_clause(char: str) -> bool:
    # Other punctuation (Po) covers , ; : . ! ? and their equivalents in other scripts (، ؛ 。 ， ። ...), but also
    # apostrophes and similar marks used inside words, which must not split.
    return unicodedata.category(char) == "Po" and char not in "'’ʼ·*#%&@\\/"
