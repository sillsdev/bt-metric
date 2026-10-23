import numpy as np
import pandas as pd

from btmetric.combination.model import PairwiseLogLinear, PerGroupScale, leave_one_group_out


def test_pairwise_model_weights_the_informative_feature_and_ignores_noise():
    rng = np.random.default_rng(0)
    informative = rng.normal(size=500)
    noise = rng.normal(size=500)
    labels = np.sign(informative + 0.1 * rng.normal(size=500))
    model = PairwiseLogLinear().fit(np.column_stack([informative, noise]), labels)
    shares = model.weight_shares()
    assert shares[0] > 0.8 and abs(shares.sum() - 1) < 1e-9
    assert np.mean(np.sign(model.score(np.column_stack([informative, noise]))) == labels) > 0.9


def test_weights_are_never_negative_even_for_a_misleading_feature():
    rng = np.random.default_rng(1)
    good = rng.normal(size=300)
    labels = np.sign(good)
    model = PairwiseLogLinear().fit(np.column_stack([good, -good]), labels)
    assert (model.weights >= 0).all() and model.weights[1] < 1e-6


def test_per_group_scale_divides_by_each_groups_spread_and_zeroes_missing():
    values = pd.DataFrame({"f": [0.0, 2.0, 0.0, 20.0]})
    scale = PerGroupScale().fit(values, ["a", "a", "b", "b"], ["f"])
    deltas = pd.DataFrame({"f": [1.0, 10.0, np.nan]})
    scaled = scale.apply(deltas, ["a", "b", "b"], ["f"])
    assert np.allclose(scaled[:, 0], [1 / np.std([0, 2], ddof=1), 10 / np.std([0, 20], ddof=1), 0.0])


def test_leave_one_group_out_predicts_each_group_from_the_others():
    rng = np.random.default_rng(2)
    deltas = rng.normal(size=(200, 1))
    labels = np.sign(deltas[:, 0])
    groups = np.repeat(["a", "b", "c", "d"], 50)
    predictions = leave_one_group_out(deltas, labels, groups)
    assert not np.isnan(predictions).any()
    assert np.mean(np.sign(predictions) == labels) > 0.95


def test_within_item_pairs_label_by_human_preference_and_skip_ties():
    from btmetric.combination.human import collection_of, system_level_accuracy, within_item_pairs

    table = pd.DataFrame(
        {
            "dataset": "wmt22-mqm-en-de",
            "item": [0, 0, 0, 1, 1],
            "system": ["A", "B", "C", "A", "B"],
            "human": [-1.0, -3.0, -1.0, 0.0, -2.0],
            "chrF3": [60.0, 40.0, 50.0, 70.0, 30.0],
        }
    )
    deltas, labels, datasets = within_item_pairs(table, ["chrF3"])
    assert labels.tolist() == [1.0, -1.0, 1.0] and deltas["chrF3"].tolist() == [20.0, -10.0, 40.0]
    assert collection_of("wmt21-mqm-en-de-news") == "wmt21-mqm"
    assert system_level_accuracy(table, table["chrF3"].to_numpy()).accuracy() == 1.0


def test_score_table_adds_feature_columns_to_matching_rows(tmp_path):
    from btmetric.combination.human import SegmentScoreTable
    from btmetric.evaluation.data import SegmentJudgments
    from btmetric.meta_eval.runner import FeatureRegistry

    rows = pd.DataFrame(
        {"item": [0, 0], "source": "s", "hypothesis": ["the cat", "a dog"], "reference": "the cat", "human": [1.0, 0.0]}
    )
    judged = [SegmentJudgments("wmt22-da-en-de", rows, "en", "de")]
    table = SegmentScoreTable(tmp_path / "scores.tsv")
    table.build(judged, ["chrf3"], FeatureRegistry())
    table.add_columns(judged, ["bleu"], FeatureRegistry())
    loaded = table.load()
    assert list(loaded.columns) == ["dataset", "item", "system", "human", "chrF3", "BLEU"]
    assert loaded["chrF3"].iloc[0] == 100.0
