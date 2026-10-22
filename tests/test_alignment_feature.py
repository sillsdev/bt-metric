from pathlib import Path

import numpy as np

from btmetric.features.alignment import AlignmentCoverage, BibleResources, Tokenizer

ENGLISH = ["jesus spoke to them", "then jesus left the house", "moses spoke to the people", "moses left the city"]
TARGET = ["yesu aliongea nao", "kisha yesu akaondoka nyumbani", "musa aliongea na watu", "musa akaondoka mjini"]


def write_bibles(tmp_path: Path) -> BibleResources:
    vrefs = [f"GEN 1:{i + 1}" for i in range(2000)]
    (tmp_path / "vref.txt").write_text("\n".join(vrefs) + "\n", encoding="utf-8")
    (tmp_path / "en-TEST.txt").write_text("\n".join(ENGLISH * 500) + "\n", encoding="utf-8")
    (tmp_path / "sw-TEST.txt").write_text("\n".join(TARGET * 500) + "\n", encoding="utf-8")
    return BibleResources(
        scripture_dirs=(tmp_path,), corpora_table=tmp_path / "none.tsv", vref_path=tmp_path / "vref.txt"
    )


def test_tokenizer_uses_characters_for_scripts_without_spaces():
    assert Tokenizer.for_corpus(["天地万物都是神创造的"] * 10).tokenize("天地 万物") == ["天", "地", "万", "物"]
    assert Tokenizer.for_corpus(["Jesus wept."] * 10).tokenize("Jesus wept.") == ["jesus", "wept", "."]


def test_bible_resources_find_extracts_by_name(tmp_path: Path):
    resources = write_bibles(tmp_path)
    assert resources.extract_path("en-TEST") == tmp_path / "en-TEST.txt"
    assert resources.extract_path("xx-MISSING") is None


def test_coverage_penalizes_additions_and_omissions_relative_to_the_reference(tmp_path: Path):
    resources = write_bibles(tmp_path)
    sources, references = ["jesus spoke to them"], ["yesu aliongea nao"]
    target_side = AlignmentCoverage("target", resources)
    target_side.prepare("en-TEST", "sw-TEST")
    added = target_side.segment_scores(sources, ["yesu aliongea nao xyzzy plugh qwerty"], references)[0]
    assert added < -0.2
    assert target_side.segment_scores(sources, references, references)[0] == 0.0
    source_side = AlignmentCoverage("source", resources)
    source_side.prepare("en-TEST", "sw-TEST")
    assert source_side.segment_scores(sources, ["yesu"], references)[0] < 0.0


def test_coverage_is_nan_without_bible_text(tmp_path: Path):
    feature = AlignmentCoverage("source", write_bibles(tmp_path))
    feature.prepare("xx-MISSING", "sw-TEST")
    assert np.isnan(feature.segment_scores(["a"], ["b"], ["c"])).all()


def test_weighted_coverage_does_not_credit_an_untranslated_copy_of_the_source(tmp_path: Path):
    resources = write_bibles(tmp_path)
    feature = AlignmentCoverage("target", resources)
    feature.prepare("en-TEST", "sw-TEST")
    sources, references = ["jesus spoke to them"], ["yesu aliongea nao"]
    assert feature.segment_scores(sources, sources, references)[0] < -0.5


def test_extracts_are_found_without_a_config_suffix_or_with_a_date(tmp_path: Path):
    resources = write_bibles(tmp_path)
    (tmp_path / "qac-ABCD.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "qab-XYZ_2023_08_22.txt").write_text("x\n", encoding="utf-8")
    assert resources.extract_path("qac-ABCD-QAC") == tmp_path / "qac-ABCD.txt"
    assert resources.extract_path("qab-XYZ") == tmp_path / "qab-XYZ_2023_08_22.txt"
