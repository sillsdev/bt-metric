from pathlib import Path

from btmetric.scripture.corpora import CorpusFinder, EvaluationLanguages, VerseCounter, VerseCounts


def write_extract(path: Path, ot: int, nt: int, dt: int = 0) -> None:
    lines = ["v"] * ot + [""] * (23213 - ot) + ["v"] * nt + [""] * (31170 - 23213 - nt) + ["v"] * dt
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verse_counter_splits_counts_by_testament(tmp_path: Path):
    path = tmp_path / "de-LU12.txt"
    write_extract(path, ot=10, nt=5, dt=2)
    assert VerseCounter({}).count(path) == VerseCounts(17, 10, 5)


def test_verse_counter_prefers_known_counts(tmp_path: Path):
    path = tmp_path / "de-LU12.txt"
    write_extract(path, ot=1, nt=1)
    assert VerseCounter({"de-LU12": VerseCounts(9, 4, 5)}).count(path) == VerseCounts(9, 4, 5)


def test_extracts_are_matched_to_test_set_languages_by_any_of_their_codes():
    languages = EvaluationLanguages.load()
    assert languages.language_of(Path("swh-ONEN.txt")) == "swh"
    assert languages.language_of(Path("sw-SUV.txt")) == "swh"
    assert languages.language_of(Path("cmn-CUV.txt")) == "zh"
    assert languages.language_of(Path("qaa-ABC.txt")) is None


def test_finder_selects_the_most_complete_extract_per_language(tmp_path: Path):
    write_extract(tmp_path / "de-NT.txt", ot=0, nt=50)
    write_extract(tmp_path / "deu-FULL.txt", ot=100, nt=50)
    write_extract(tmp_path / "qaa-ABC.txt", ot=100, nt=50)
    write_extract(tmp_path / "de-TEST.txt", ot=0, nt=0)
    counter = VerseCounter({"de-TEST": VerseCounts(94907, 23205, 7952)})
    finder = CorpusFinder(EvaluationLanguages.load(), counter, workers=2)
    table = finder.candidates([tmp_path])
    assert table[table["selected"]]["extract"].tolist() == ["deu-FULL"]
    assert len(table) == 3
    assert "zu" in finder.missing_languages(table)


def test_finder_selects_a_preferred_extract_over_a_more_complete_one(tmp_path: Path):
    write_extract(tmp_path / "en-BSB.txt", ot=100, nt=50)
    write_extract(tmp_path / "en-QaaBT.txt", ot=200, nt=50, dt=10)
    table = CorpusFinder(EvaluationLanguages.load(), VerseCounter({}), workers=2).candidates([tmp_path])
    assert table[table["selected"]]["extract"].tolist() == ["en-BSB"]
