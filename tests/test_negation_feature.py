from pathlib import Path

import numpy as np

from btmetric.features.corpora import BibleCorpus
from btmetric.features.key_terms import EnglishBible
from btmetric.features.negation import NegationConsistency

SUBJECTS = [("he", "yeye"), ("they", "wao"), ("the people", "watu"), ("moses", "musa")]
VERBS = [("go", "enda"), ("eat", "kula"), ("sleep", "lala"), ("pray", "omba"), ("wait", "subiri")]


def toy_feature(tmp_path: Path, negator: str, missing_only: bool = False) -> NegationConsistency:
    """Toy Bibles where English "not" is rendered by the negator word (by nothing when it is empty)."""
    english, target = [], []
    for i in range(2400):
        (subject, subject_t), (verb, verb_t) = SUBJECTS[i % 4], VERBS[(i // 4) % 5]
        negated = (i // 20) % 2 == 0
        english.append(f"{subject} {'not' if negated else 'will'} {verb}")
        target.append(" ".join(filter(None, [subject_t, negator if negated else "ata", verb_t])))
    vrefs = [f"GEN 1:{i + 1}" for i in range(2400)]
    (tmp_path / "vref.txt").write_text("\n".join(vrefs) + "\n", encoding="utf-8")
    (tmp_path / "en-BSB.txt").write_text("\n".join(english) + "\n", encoding="utf-8")
    (tmp_path / "sw-TEST.txt").write_text("\n".join(target) + "\n", encoding="utf-8")
    (tmp_path / "Major-metadata.txt").write_text("", encoding="utf-8")
    (tmp_path / "Major-vrefs.txt").write_text("", encoding="utf-8")
    (tmp_path / "en-Major-glosses.txt").write_text("", encoding="utf-8")
    resources = BibleCorpus(
        scripture_dirs=(tmp_path,), corpora_table=tmp_path / "none.tsv", vref_path=tmp_path / "vref.txt"
    )
    return NegationConsistency(resources=resources, english=EnglishBible("en-BSB", tmp_path), missing_only=missing_only)


def test_a_dropped_or_added_negation_is_penalized(tmp_path: Path):
    feature = toy_feature(tmp_path, "si")
    feature.prepare("en-X", "sw-TEST")
    references = ["yeye si enda", "yeye si enda", "watu ata subiri"]
    scores = feature.segment_scores(["s"] * 3, ["yeye si enda", "yeye enda", "watu si subiri"], references)
    assert list(scores) == [0.0, -1.0, -1.0]


def test_no_score_without_negation_words(tmp_path: Path):
    feature = toy_feature(tmp_path, "")
    feature.prepare("en-X", "sw-TEST")
    assert np.isnan(feature.segment_scores(["s"], ["yeye enda"], ["yeye enda"])).all()


def test_missing_only_ignores_an_extra_negation(tmp_path: Path):
    feature = toy_feature(tmp_path, "si", missing_only=True)
    feature.prepare("en-X", "sw-TEST")
    references = ["yeye si enda", "watu ata subiri"]
    scores = feature.segment_scores(["s"] * 2, ["yeye enda", "watu si subiri"], references)
    assert list(scores) == [-1.0, 0.0]
