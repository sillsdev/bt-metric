from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btmetric.evaluation.data import EvaluationSet, JudgedOutputs, SegmentJudgments
from btmetric.evaluation.loaders import AfriMteLoader, ExperimentOutputsLoader, WmtDaLoader, WmtMqmLoader
from btmetric.features.chrf import ChrF
from btmetric.meta_eval.runner import SegmentLevelEvaluator, SystemLevelEvaluator


def evaluation_set(n: int) -> EvaluationSet:
    return EvaluationSet(
        "set", "en", "de", [f"s{i}" for i in range(n)], [f"r{i}" for i in range(n)], [str(i) for i in range(n)]
    )


def test_judged_outputs_reject_hypotheses_of_the_wrong_length():
    with pytest.raises(ValueError):
        JudgedOutputs(evaluation_set(2)).add_system("a", ["only one"])


def test_judged_outputs_flatten_to_segment_judgments_with_segments_as_items():
    outputs = JudgedOutputs(evaluation_set(2))
    outputs.add_system("a", ["a0", "a1"], np.array([1.0, np.nan]))
    outputs.add_system("b", ["b0", "b1"], np.array([2.0, 3.0]))
    judgments = outputs.segment_judgments()
    assert judgments.column("hypothesis") == ["a0", "b0", "b1"]
    assert judgments.items().tolist() == [0, 0, 1]


def test_sampling_keeps_every_hypothesis_of_the_chosen_items():
    table = pd.DataFrame(
        {"item": [0, 0, 1, 1, 2, 2], "source": "s", "hypothesis": "h", "reference": "r", "human": np.arange(6.0)}
    )
    sampled = SegmentJudgments("set", table).sample_items(2)
    assert len(sampled) == 4
    assert all(count == 2 for count in pd.Series(sampled.items()).value_counts())


def write_mqm(path: Path, rows: list) -> Path:
    columns = ["lp", "domain", "src", "mt", "ref", "score", "system"]
    pd.DataFrame(rows, columns=columns).assign(year=2022).to_csv(path, index=False)
    return path


def test_mqm_loader_averages_repeated_sentences_and_uses_the_majority_reference(tmp_path: Path):
    rows = [
        ("en-de", "news", "s1", "a1", "r1", -1.0, "A"),
        ("en-de", "news", "s1", "a1", "r1", -3.0, "A"),
        ("en-de", "news", "s1", "b1", "r1-alt", 0.0, "B"),
        ("en-de", "news", "s1", "b1", "r1", 0.0, "B"),
        ("en-de", "social", "s2", "a2", "r2", -5.0, "A"),
        ("en-de", "social", "s2", "b2", "r2", -1.0, "B"),
    ]
    [outputs] = list(WmtMqmLoader(write_mqm(tmp_path / "mqm.csv", rows)).judged_outputs())
    assert outputs.evaluation_set.name == "wmt22-mqm-en-de"
    assert outputs.evaluation_set.references == ["r1", "r2"]
    assert outputs.hypotheses == {"A": ["a1", "a2"], "B": ["b1", "b2"]}
    assert outputs.human_system_score("A") == -3.5


def test_mqm_loader_splits_domains_judged_on_different_systems(tmp_path: Path):
    rows = [
        ("en-de", "news", "s1", "a1", "r1", -1.0, "A"),
        ("en-de", "news", "s1", "b1", "r1", -2.0, "B"),
        ("en-de", "ted", "s2", "a2", "r2", -5.0, "A"),
    ]
    outputs = list(WmtMqmLoader(write_mqm(tmp_path / "mqm.csv", rows)).judged_outputs())
    assert [(o.evaluation_set.name, o.systems()) for o in outputs] == [
        ("wmt22-mqm-en-de-news", ["A", "B"]),
        ("wmt22-mqm-en-de-ted", ["A"]),
    ]


def test_da_loader_groups_hypotheses_of_one_source_into_an_item(tmp_path: Path):
    rows = [("en-gu", "s1", "h1", "r1", 0.5), ("en-gu", "s1", "h2", "r1", -0.5), ("en-gu", "s2", "h3", "r2", 0.1)]
    table = pd.DataFrame(rows, columns=["lp", "src", "mt", "ref", "score"]).assign(year=2019)
    table.to_csv(tmp_path / "da.csv", index=False)
    [judgments] = list(WmtDaLoader(tmp_path / "da.csv").segment_judgments())
    assert judgments.name == "wmt19-da-en-gu"
    assert judgments.items().tolist() == [0, 0, 1]


def test_afrimte_loader_reads_parallel_files(tmp_path: Path):
    for kind, text in [("src", "a\nb\n"), ("hyp-1", "x\ny\n"), ("ref", "p\nq\n"), ("score", "0.5\n-1\n")]:
        (tmp_path / f"AfriMTE-ade-devtest-v2.eng-hau.{kind}.txt").write_text(text, encoding="utf-8")
    [judgments] = list(AfriMteLoader(tmp_path).segment_judgments())
    assert judgments.name == "afrimte-eng-hau"
    assert judgments.column("hypothesis") == ["x", "y"]
    assert judgments.human().tolist() == [0.5, -1.0]


def test_experiment_outputs_loader_reads_each_distinct_prediction(tmp_path: Path):
    directory = tmp_path / "Series" / "exp"
    directory.mkdir(parents=True)
    for name, text in [
        ("test.vref.txt", "MAT 1:1\nMAT 1:2\n"),
        ("test.src.detok.txt", "s1\ns2\n"),
        ("test.trg.detok.txt", "r1\nr2\n"),
        ("test.trg-predictions.detok.txt.1000", "p1\np2\n"),
    ]:
        (directory / name).write_text(text, encoding="utf-8")
    index = pd.DataFrame(
        [
            {
                "experiment": "Series/exp",
                "prefix": "test",
                "step": "1000",
                "test_hash": "h",
                "prediction_hash": "p",
                "src": "en-NIV",
                "trg": "qaa-ABC",
            }
        ]
    )
    outputs = ExperimentOutputsLoader(tmp_path, index).outputs_for("h")
    assert outputs.evaluation_set.segment_ids == ["MAT 1:1", "MAT 1:2"]
    assert outputs.hypotheses == {"Series/exp|test|1000": ["p1", "p2"]}


def test_chrf_scores_identical_text_as_perfect_and_has_a_corpus_form():
    feature = ChrF(word_order=2)
    assert feature.name == "chrF3++"
    assert feature.segment_scores(["s"], ["the same text"], ["the same text"]).tolist() == [100.0]
    assert feature.system_score(["s", "s"], ["abc def", "ghi"], ["abc def", "xyz"]) < 100.0


def test_system_level_ranks_systems_by_human_mean_and_excludes_human_references():
    outputs = JudgedOutputs(evaluation_set(2))
    outputs.add_system("good", ["r0", "r1"], np.array([0.0, 0.0]))
    outputs.add_system("bad", ["xx", "yy"], np.array([-5.0, -5.0]))
    outputs.add_system("refB", ["zz", "ww"], np.array([1.0, 1.0]))
    results = SystemLevelEvaluator(r"^ref").evaluate([outputs], ChrF(), "toy")
    assert [(result.dataset, result.value, result.comparisons) for result in results] == [
        ("set", 1.0, 1),
        ("toy (pooled)", 1.0, 1),
    ]


def test_segment_level_uses_acc_eq_within_items_and_kendall_without_them():
    grouped = pd.DataFrame(
        {
            "item": [0, 0],
            "source": "s",
            "hypothesis": ["the reference", "zzz"],
            "reference": "the reference",
            "human": [1.0, -1.0],
        }
    )
    evaluator = SegmentLevelEvaluator(max_items=10, tie_epsilons=[0.0])
    result = evaluator.evaluate(SegmentJudgments("grouped", grouped), ChrF())
    assert (result.statistic, result.value) == ("acc_eq", 1.0)
    single = grouped.assign(item=[0, 1])
    result = evaluator.evaluate(SegmentJudgments("single", single), ChrF())
    assert result.statistic == "kendall_tau_b"


def test_results_file_replaces_rows_of_rerun_features(tmp_path: Path):
    from btmetric.meta_eval.runner import ResultsFile

    path = tmp_path / "results.tsv"
    ResultsFile(path).merge(pd.DataFrame({"feature": ["chrF3", "BLEU"], "value": [0.1, 0.2]}))
    table = ResultsFile(path).merge(pd.DataFrame({"feature": ["chrF3"], "value": [0.9]}))
    assert sorted(zip(table["feature"], table["value"])) == [("BLEU", 0.2), ("chrF3", 0.9)]
