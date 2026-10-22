"""Infers a language's negation words from their alignment with English negators.

A target token counts as a negation marker when it is aligned to an English negator in a large share of the verses
it occurs in. Languages that negate with affixes rather than separate words won't yield markers this way.
"""

from collections import Counter
from typing import List

from .inference import AlignedVerses


class NegationMarkers:
    def __init__(
        self,
        english_negators: frozenset = frozenset({"not", "no", "never", "nothing", "none", "nor", "neither", "nobody"}),
        min_count: int = 10,
        min_share: float = 0.5,
        max_markers: int = 5,
    ) -> None:
        self._english_negators = english_negators
        self._min_count = min_count
        self._min_share = min_share
        self._max_markers = max_markers

    def infer(self, verses: AlignedVerses) -> List[str]:
        aligned_to_negator: Counter = Counter()
        occurrences: Counter = Counter()
        for english, target, alignment in zip(verses.english, verses.target, verses.alignments):
            negator_positions = {i for i, token in enumerate(english) if token in self._english_negators}
            negated_targets = {target[j] for i, j in alignment if i in negator_positions}
            aligned_to_negator.update(negated_targets)
            occurrences.update(set(target))
        markers = [
            (count / occurrences[token], token)
            for token, count in aligned_to_negator.items()
            if count >= self._min_count and count / occurrences[token] >= self._min_share and token.isalpha()
        ]
        return [token for _, token in sorted(markers, reverse=True)[: self._max_markers]]
