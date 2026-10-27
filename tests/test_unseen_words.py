from pathlib import Path

import numpy as np

from btmetric.features.corpora import BibleCorpus
from btmetric.features.unseen_words import UnseenWords, VocabularyCache

TARGET = ["yesu aliongea nao", "kisha yesu akaondoka nyumbani", "musa aliongea na watu", "musa akaondoka mjini"]


def bible(tmp_path: Path, lines: list) -> BibleCorpus:
    vrefs = [f"GEN 1:{i + 1}" for i in range(len(lines))]
    (tmp_path / "vref.txt").write_text("\n".join(vrefs) + "\n", encoding="utf-8")
    (tmp_path / "sw-TEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BibleCorpus(scripture_dirs=(tmp_path,), corpora_table=tmp_path / "none.tsv", vref_path=tmp_path / "vref.txt")


def test_words_missing_from_the_bible_count_against_the_hypothesis(tmp_path: Path):
    feature = UnseenWords(bible(tmp_path, TARGET * 100))
    feature.prepare("en", "sw-TEST")
    scores = feature.segment_scores(
        ["s"] * 3, ["yesu aliongea nao", "yesu xqzv nao", "yesu, 12."], ["yesu aliongea nao"] * 3
    )
    assert scores[0] == 0.0 and np.isclose(scores[1], -1 / 3) and scores[2] == 0.0


def test_implausible_only_spares_unseen_inflections_of_known_stems(tmp_path: Path):
    vocabularies = VocabularyCache(bible(tmp_path, TARGET * 100))
    feature = UnseenWords(implausible_only=True, vocabularies=vocabularies)
    feature.prepare("en", "sw-TEST")
    references = ["musa akaondoka mjini"] * 2
    scores = feature.segment_scores(["s"] * 2, ["musa akaondokani mjini", "musa qxjzvw mjini"], references)
    assert scores[0] == 0.0 and scores[1] < 0.0


def test_no_score_for_scripts_without_spaces(tmp_path: Path):
    feature = UnseenWords(bible(tmp_path, ["天地万物都是神创造的"] * 100))
    feature.prepare("en", "sw-TEST")
    assert np.isnan(feature.segment_scores(["s"], ["天地"], ["天地"])).all()
