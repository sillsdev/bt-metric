from pathlib import Path

import yaml

from btmetric.experiments.index import ExperimentScanner
from btmetric.experiments.pairs import SystemPairFinder


def write_experiment(directory: Path, config: dict, predictions: dict, vrefs: str = "MAT 1:1\nMAT 1:2\n") -> None:
    directory.mkdir(parents=True)
    (directory / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (directory / "test.vref.txt").write_text(vrefs, encoding="utf-8")
    (directory / "test.trg.detok.txt").write_text("ref one\nref two\n", encoding="utf-8")
    for step, text in predictions.items():
        (directory / f"test.trg-predictions.detok.txt.{step}").write_text(text, encoding="utf-8")


def config(src: str, test_books: bool = False) -> dict:
    pair = {"src": src, "trg": "qaa-ABC", "type": "train,test,val"}
    if test_books:
        pair["test_books"] = "MAT"
    return {"model": "facebook/nllb-200-distilled-1.3B", "data": {"corpus_pairs": [pair]}}


def scan(root: Path):
    return ExperimentScanner(root, max_depth=4, workers=2).scan()


def test_scanner_indexes_each_checkpoint_with_its_test_set(tmp_path: Path):
    write_experiment(tmp_path / "Series" / "a", config("en-NIV"), {"1000": "x\ny\n", "2000": "x\nz\n"})
    index = scan(tmp_path)
    assert sorted(index["step"]) == ["1000", "2000"]
    assert index["experiment"].unique().tolist() == ["Series/a"]
    assert index["test_hash"].nunique() == 1
    assert index["verses"].tolist() == [2, 2]
    assert index["src"].unique().tolist() == ["en-NIV"]
    assert index["test_design"].unique().tolist() == ["random"]


def test_scanner_identifies_held_out_books(tmp_path: Path):
    write_experiment(tmp_path / "a", config("en-NIV", test_books=True), {"1000": "x\ny\n"})
    assert scan(tmp_path)["test_design"].tolist() == ["books"]


def test_line_endings_do_not_change_the_test_hash(tmp_path: Path):
    write_experiment(tmp_path / "a", config("en-NIV"), {"1000": "x\ny\n"})
    write_experiment(tmp_path / "b", config("en-NLT"), {"1000": "x\ny\n"})
    (tmp_path / "b" / "test.vref.txt").write_bytes(b"MAT 1:1\r\nMAT 1:2\r\n")
    assert scan(tmp_path)["test_hash"].nunique() == 1


def test_pairs_are_only_formed_within_a_test_set(tmp_path: Path):
    write_experiment(tmp_path / "a", config("en-NIV"), {"1000": "a\n"})
    write_experiment(tmp_path / "b", config("en-NLT"), {"1000": "b\n"})
    write_experiment(tmp_path / "c", config("en-NLT"), {"1000": "c\n"}, vrefs="MAT 2:1\nMAT 2:2\n")
    pairs = SystemPairFinder(scan(tmp_path)).pairs()
    assert len(pairs) == 1
    assert pairs.iloc[0]["relation"] == "config"
    assert pairs.iloc[0]["differences"] == "data.corpus_pairs.0.src"


def test_pairs_classify_checkpoints_and_reruns_and_drop_identical_outputs(tmp_path: Path):
    write_experiment(tmp_path / "a", config("en-NIV"), {"1000": "a1\n", "2000": "a2\n"})
    write_experiment(tmp_path / "rerun", config("en-NIV"), {"1000": "r\n", "2000": "a2\n"})
    pairs = SystemPairFinder(scan(tmp_path)).pairs()
    relations = sorted(pairs["relation"])
    assert len(pairs) == 3
    assert relations == ["checkpoint", "rerun", "rerun"]
