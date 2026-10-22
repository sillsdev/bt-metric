from pathlib import Path

import pandas as pd

from btmetric.terms.inference import AlignedVerses, GlossMatcher, InferredRendering, RenderingInferrer
from btmetric.terms.major_terms import MajorTermList
from btmetric.terms.validate import CuratedRenderings, compare, summarize


def test_gloss_matcher_allows_inflection_and_multiword_glosses():
    matcher = GlossMatcher()
    tokens = "the apostles went to the high priest".split()
    assert matcher.positions(tokens, ["apostle"]) == [1]
    assert matcher.positions(tokens, ["high priest"]) == [5, 6]
    assert matcher.positions(tokens, ["at"]) == []


def test_curated_renderings_match_exactly_or_by_stem():
    curated = CuratedRenderings([{"yesu"}, {"abraham"}, set()])
    assert curated.matches(0, "yesu") and curated.matches(1, "abrahamu")
    assert not curated.matches(0, "musa") and not curated.has(2)


def test_inference_counts_the_aligned_target_words_of_each_gloss():
    terms = MajorTermList(["PN|person", "PN|person"], [["MAT 1:1", "MAT 1:2"], ["MAT 1:3"]], [["Jesus"], ["Moses"]])
    verses = AlignedVerses(
        ["MAT 1:1", "MAT 1:2", "MAT 1:3"],
        [["jesus", "spoke"], ["then", "jesus", "left"], ["moses", "said"]],
        [["yesu", "alisema"], ["kisha", "yesu", "akaenda"], ["musa", "alisema"]],
    )
    verses.alignments = [[(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1)]]
    inferred = {r.term: r for r in RenderingInferrer(terms).infer(verses)}
    assert (inferred[0].rendering, inferred[0].confidence, inferred[0].occurrences) == ("yesu", 1.0, 2)
    assert inferred[1].rendering == "musa"


def test_alignment_trains_and_aligns_real_text():
    english = [["jesus", "spoke"], ["jesus", "left"], ["moses", "spoke"], ["moses", "left"]] * 5
    target = [["yesu", "alisema"], ["yesu", "akaenda"], ["musa", "alisema"], ["musa", "akaenda"]] * 5
    verses = AlignedVerses([str(i) for i in range(20)], english, target).align()
    assert (0, 0) in verses.alignments[0]


def test_validation_reports_precision_and_recall(tmp_path: Path):
    terms = MajorTermList(["PN|person"] * 3, [["A"], ["B"], ["C"]], [["x"], ["y"], ["z"]])
    curated = CuratedRenderings([{"yesu"}, {"musa"}, {"petro"}])
    inferred = [
        InferredRendering(0, "PN", "yesu", 1.0, 2),
        InferredRendering(1, "PN", "daudi", 0.5, 2),
        InferredRendering(2, "PN", "", 0.0, 1),
    ]
    table = pd.DataFrame([vars(row) for row in compare("p", terms, curated, inferred)])
    assert table["correct"].tolist() == [True, False, False]
    assert "Precision 0.500 on 2 terms; recall 0.333" in summarize(table)
