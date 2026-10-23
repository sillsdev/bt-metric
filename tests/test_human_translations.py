from pathlib import Path

from btmetric.analysis.human_translations import HumanTranslationTest, VerseAlignedExtracts, summarize
from btmetric.evaluation.data import EvaluationSet, JudgedOutputs
from btmetric.meta_eval.runner import FeatureRegistry


def test_extract_verses_are_looked_up_by_vref(tmp_path: Path):
    (tmp_path / "vref.txt").write_text("GEN 1:1\nGEN 1:2\nGEN 1:3\n", encoding="utf-8")
    (tmp_path / "en-X.txt").write_text("one\n<range>\nthree\n", encoding="utf-8")
    verses = VerseAlignedExtracts(tmp_path / "vref.txt").verses(
        tmp_path / "en-X.txt", ["GEN 1:3", "GEN 1:2", "EXO 1:1"]
    )
    assert verses == ["three", "", ""]


def test_human_translations_are_scored_on_verses_all_of_them_have_and_ranked_against_mt():
    references = ["in the beginning god created the heavens", "and the earth was formless", "and god said"]
    evaluation_set = EvaluationSet(
        "set", "hbo", "en", ["s1", "s2", "s3"], references, ["GEN 1:1", "GEN 1:2", "GEN 1:3"]
    )
    outputs = JudgedOutputs(evaluation_set)
    outputs.add_system("weak", ["god made", "earth", "said"])
    outputs.add_system("strong", ["in the beginning god created the heaven", "the earth was formless", "god said"])
    alternatives = {
        "close": ["in the beginning god created the heavens", "and the earth was formless.", ""],
        "same-as-reference": references,
    }
    scores = HumanTranslationTest(outputs, alternatives).scores(["chrf3"], FeatureRegistry())
    assert sorted(scores["candidate"]) == ["close", "strong", "weak"]
    summary = summarize(scores)
    assert summary["human_translation"].tolist() == ["close"]
    assert summary["beats_mt_share"].tolist() == [1.0]
    assert summary["rank"].tolist() == [1]


def test_strip_marks_removes_hebrew_points_and_cantillation():
    from btmetric.analysis.human_translations import strip_marks

    assert strip_marks("\u05d5\u05b0\u05d0\u05b5\u0597\u05dc\u05bc\u05b6\u05d4") == "\u05d5\u05d0\u05dc\u05d4"
    assert strip_marks("Jesus wept.") == "Jesus wept."
