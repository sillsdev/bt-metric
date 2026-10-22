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
