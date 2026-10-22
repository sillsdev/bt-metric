from pathlib import Path

import numpy as np

from btmetric.features.alignment import BibleResources
from btmetric.features.key_terms import EnglishBible, KeyTermAccuracy, RenderingMatcher


def test_rendering_matcher_finds_inflected_and_multiword_renderings():
    matcher = RenderingMatcher({("yesu",), ("mlima", "sayuni")})
    tokens = "kisha yesuni akapanda mlimani sayuni".split()
    assert matcher.found(tokens) == {("yesu",), ("mlima", "sayuni")}
    assert not matcher.contains("musa akapanda".split(), ("yesu",))


def write_toy_bibles(tmp_path: Path) -> KeyTermAccuracy:
    vrefs = [f"GEN 1:{i + 1}" for i in range(2400)]
    (tmp_path / "vref.txt").write_text("\n".join(vrefs) + "\n", encoding="utf-8")
    english = ["jesus spoke to them", "then moses left the house", "the people waited", "jesus and moses met"]
    target = ["yesu aliongea nao", "kisha musa akaondoka nyumbani", "watu walisubiri", "yesu na musa walikutana"]
    (tmp_path / "en-BSB.txt").write_text("\n".join(english * 600) + "\n", encoding="utf-8")
    (tmp_path / "sw-TEST.txt").write_text("\n".join(target * 600) + "\n", encoding="utf-8")
    jesus = "\t".join(vrefs[0::4] + vrefs[3::4])
    moses = "\t".join(vrefs[1::4] + vrefs[3::4])
    (tmp_path / "Major-metadata.txt").write_text("Ἰησοῦς\tPN\tperson\nΜωϋσῆς\tPN\tperson\n", encoding="utf-8")
    (tmp_path / "Major-vrefs.txt").write_text(f"{jesus}\n{moses}\n", encoding="utf-8")
    (tmp_path / "en-Major-glosses.txt").write_text("Jesus\nMoses\n", encoding="utf-8")
    resources = BibleResources(
        scripture_dirs=(tmp_path,), corpora_table=tmp_path / "none.tsv", vref_path=tmp_path / "vref.txt"
    )
    return KeyTermAccuracy(resources=resources, english=EnglishBible("en-BSB", tmp_path))


def test_key_term_accuracy_rewards_keeping_the_references_terms(tmp_path: Path):
    feature = write_toy_bibles(tmp_path)
    feature.prepare("en-X", "sw-TEST")
    sources = ["s"] * 3
    references = ["yesu na musa walikutana"] * 2 + ["watu walisubiri"]
    scores = feature.segment_scores(
        sources, ["yesu na musa walikutana", "yesu na petro walikutana", "watu"], references
    )
    assert scores[0] == 1.0 and scores[1] == 0.5 and np.isnan(scores[2])


def test_short_renderings_do_not_match_words_that_merely_share_a_prefix():
    matcher = RenderingMatcher({("akani",), ("yese",), ("yerusalemu",)})
    assert matcher.found("akasema yesu".split()) == set()
    assert matcher.found("akanini yerusalem".split()) == {("akani",), ("yerusalemu",)}


def test_renderings_mostly_found_outside_their_terms_verses_are_dropped():
    from btmetric.terms.inference import AlignedVerses, InferredRendering, specific_renderings
    from btmetric.terms.major_terms import MajorTermList

    vrefs = [str(i) for i in range(60)]
    # "musa" occurs only in its term's 10 verses; "yake" ("his") in all 60, only 10 of them its term's.
    verses = AlignedVerses(vrefs, [["x"]] * 60, [["yake", "musa"]] * 10 + [["yake"]] * 50)
    terms = MajorTermList(["PN|person", "PN|person"], [vrefs[:10], vrefs[:10]])
    inferred = [InferredRendering(0, "PN", "musa", 1.0, 10), InferredRendering(1, "PN", "yake", 1.0, 10)]
    assert [r.rendering for r in specific_renderings(inferred, verses, terms)] == ["musa"]


def test_three_letter_renderings_only_match_whole_words():
    matcher = RenderingMatcher({("aka",)})
    assert matcher.found("akaja aka".split()) == {("aka",)}
    assert matcher.found("akaja".split()) == set()


def test_namesakes_sharing_a_rendering_are_not_dropped():
    from btmetric.terms.inference import AlignedVerses, InferredRendering, specific_renderings
    from btmetric.terms.major_terms import MajorTermList

    vrefs = [str(i) for i in range(40)]
    # Two Johns, each a separate term in 20 verses, both rendered "yohana".
    verses = AlignedVerses(vrefs, [["x"]] * 40, [["yohana"]] * 40)
    terms = MajorTermList(["PN|person", "PN|person"], [vrefs[:20], vrefs[20:]])
    inferred = [InferredRendering(0, "PN", "yohana", 1.0, 20), InferredRendering(1, "PN", "yohana", 1.0, 20)]
    assert len(specific_renderings(inferred, verses, terms, min_share=0.6)) == 2
