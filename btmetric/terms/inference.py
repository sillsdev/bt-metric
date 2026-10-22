"""Infers a project's renderings of the Major Biblical Terms by word-aligning an English Bible with its text.

Each term is located in English through its glosses, in the verses the term list says it occurs in. The target
words eflomal aligns to the gloss are counted across those verses, and the most frequent is the inferred rendering.
"""

import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from machine.corpora import TextFileTextCorpus
from machine.tokenization import LatinWordTokenizer, WhitespaceTokenizer
from machine.translation.thot import ThotEflomalWordAlignmentModel, ThotSymmetrizedWordAlignmentModel

from .major_terms import MajorTermList


@dataclass(frozen=True)
class InferredRendering:
    term: int
    category: str
    rendering: str
    confidence: float
    occurrences: int


class AlignedVerses:
    """Verses both texts have, tokenized and word-aligned with a symmetrized eflomal model trained on them."""

    def __init__(self, vrefs: List[str], english: List[List[str]], target: List[List[str]]) -> None:
        self.vrefs = vrefs
        self.english = english
        self.target = target
        self._row_by_vref = {vref: i for i, vref in enumerate(vrefs)}
        self.alignments: List[List[Tuple[int, int]]] = []

    @classmethod
    def from_extracts(
        cls, vref_lines: Sequence[str], english_lines: Sequence[str], target_lines: Sequence[str], excluded: Set[str]
    ) -> "AlignedVerses":
        tokenizer = LatinWordTokenizer()
        vrefs, english, target = [], [], []
        for vref, english_verse, target_verse in zip(vref_lines, english_lines, target_lines):
            english_verse, target_verse = english_verse.strip(), target_verse.strip()
            if not english_verse or not target_verse or "<range>" in (english_verse, target_verse) or vref in excluded:
                continue
            vrefs.append(vref)
            english.append([token.lower() for token in tokenizer.tokenize(english_verse)])
            target.append([token.lower() for token in tokenizer.tokenize(target_verse)])
        return cls(vrefs, english, target)

    def align(self) -> "AlignedVerses":
        model = self.trained_model()
        matrices = model.align_batch(list(zip(self.english, self.target)))
        self.alignments = [
            sorted((pair.source_index, pair.target_index) for pair in matrix.to_aligned_word_pairs())
            for matrix in matrices
        ]
        return self

    def trained_model(self) -> ThotSymmetrizedWordAlignmentModel:
        """A symmetrized eflomal model trained on these verses; it can also align sentence pairs it wasn't
        trained on."""
        with tempfile.TemporaryDirectory() as work:
            src_path, trg_path = Path(work) / "src.txt", Path(work) / "trg.txt"
            src_path.write_text("\n".join(" ".join(tokens) for tokens in self.english) + "\n", encoding="utf-8")
            trg_path.write_text("\n".join(" ".join(tokens) for tokens in self.target) + "\n", encoding="utf-8")
            corpus = TextFileTextCorpus(src_path).align_rows(TextFileTextCorpus(trg_path))
            model = ThotSymmetrizedWordAlignmentModel(ThotEflomalWordAlignmentModel(), ThotEflomalWordAlignmentModel())
            with model.create_trainer(corpus.tokenize(WhitespaceTokenizer())) as trainer:
                trainer.train()
                trainer.save()
        return model

    def row(self, vref: str) -> Optional[int]:
        return self._row_by_vref.get(vref)

    def __len__(self) -> int:
        return len(self.vrefs)


class GlossMatcher:
    """Finds a gloss in tokenized English, allowing inflectional endings ("apostle" matches "apostles")."""

    def positions(self, tokens: List[str], glosses: List[str]) -> List[int]:
        for gloss in glosses:
            gloss_tokens = gloss.lower().split()
            for start in range(len(tokens) - len(gloss_tokens) + 1):
                if all(self._matches(tokens[start + i], g) for i, g in enumerate(gloss_tokens)):
                    return list(range(start, start + len(gloss_tokens)))
        return []

    def _matches(self, token: str, gloss_token: str) -> bool:
        if token == gloss_token:
            return True
        return len(gloss_token) >= 3 and token.startswith(gloss_token) and len(token) - len(gloss_token) <= 3


class RenderingInferrer:
    def __init__(self, terms: MajorTermList, max_span: int = 4) -> None:
        self._terms = terms
        self._max_span = max_span
        self._matcher = GlossMatcher()

    def infer(self, verses: AlignedVerses) -> List[InferredRendering]:
        inferred = []
        for term in range(len(self._terms)):
            glosses = self._terms.glosses(term)
            if not glosses:
                continue
            counts: Counter = Counter()
            occurrences = 0
            for vref in self._terms.vrefs(term):
                row = verses.row(vref)
                if row is None:
                    continue
                positions = self._matcher.positions(verses.english[row], glosses)
                if not positions:
                    continue
                occurrences += 1
                phrase = self._aligned_phrase(verses, row, positions)
                if phrase:
                    counts[phrase] += 1
            if occurrences == 0:
                continue
            # Terms found in English but never aligned get an empty rendering, so recall can count them as misses.
            rendering, count = counts.most_common(1)[0] if counts else ("", 0)
            inferred.append(
                InferredRendering(term, self._terms.category(term), rendering, count / occurrences, occurrences)
            )
        return inferred

    def _aligned_phrase(self, verses: AlignedVerses, row: int, positions: List[int]) -> Optional[str]:
        wanted = set(positions)
        target_indices = sorted({j for i, j in verses.alignments[row] if i in wanted})
        # A contiguous target span; a long one means the alignment wandered, so it says little about the rendering.
        if not target_indices or target_indices[-1] - target_indices[0] >= self._max_span:
            return None
        return " ".join(verses.target[row][target_indices[0] : target_indices[-1] + 1])


def renderings_by_term(inferred: List[InferredRendering]) -> Dict[int, InferredRendering]:
    return {rendering.term: rendering for rendering in inferred}


def specific_renderings(
    inferred: List[InferredRendering], verses: AlignedVerses, terms: MajorTermList, min_share: float = 0.2
) -> List[InferredRendering]:
    """Drops renderings that mostly occur in verses where their term doesn't.

    Alignment sometimes attaches a gloss to a frequent neighbour, such as a possessive or conjunction, with high
    confidence. Such a word appears mostly outside the term's verses; a real rendering, even a frequent one like
    "God", mostly inside them. (Comparing raw frequencies instead drops frequent real renderings too.)
    """
    verses_with_token: Dict[str, Set[int]] = {}
    for row, tokens in enumerate(verses.target):
        for token in set(tokens):
            verses_with_token.setdefault(token, set()).add(row)
    # Namesakes are separate terms (John the Baptist, John the apostle) usually sharing one rendering, so a
    # rendering's verses are compared with those of every term it was inferred for.
    term_rows_by_rendering: Dict[str, Set[int]] = {}
    for rendering in inferred:
        rows = {verses.row(vref) for vref in terms.vrefs(rendering.term)} - {None}
        term_rows_by_rendering.setdefault(rendering.rendering, set()).update(rows)
    kept = []
    for rendering in inferred:
        token_rows = [verses_with_token.get(token, set()) for token in rendering.rendering.split()]
        rendering_rows = set.intersection(*token_rows) if token_rows else set()
        if not rendering_rows:
            continue
        term_rows = term_rows_by_rendering[rendering.rendering]
        if len(rendering_rows & term_rows) / len(rendering_rows) >= min_share:
            kept.append(rendering)
    return kept
