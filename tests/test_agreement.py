import math

import numpy as np

from btmetric.meta_eval.agreement import PairwiseAgreement
from btmetric.meta_eval.bootstrap import PairedBootstrap


def test_accuracy_skips_gold_ties_and_counts_metric_ties_as_wrong():
    agreement = PairwiseAgreement(gold_deltas=[1, -2, 0, 3], metric_deltas=[0.5, 1.0, 9.0, 0.0])
    assert agreement.accuracy() == 1 / 3


def test_accuracy_is_nan_without_decided_pairs():
    assert math.isnan(PairwiseAgreement([0, 0], [1, -1]).accuracy())


def test_all_pairs_compares_every_item_with_every_other():
    agreement = PairwiseAgreement.all_pairs(gold_scores=[3, 2, 1], metric_scores=[30, 10, 20])
    assert agreement.accuracy() == 2 / 3


def test_accuracy_with_ties_rewards_metric_ties_on_gold_ties():
    agreement = PairwiseAgreement(gold_deltas=[0, 1, -1], metric_deltas=[0.05, 0.5, 0.05])
    assert agreement.accuracy_with_ties(epsilon=0.0) == 1 / 3
    assert agreement.accuracy_with_ties(epsilon=0.1) == 2 / 3


def test_best_tie_epsilon_picks_the_most_accurate_candidate():
    agreement = PairwiseAgreement(gold_deltas=[0, 1, -1], metric_deltas=[0.05, 0.5, -0.5])
    epsilon, accuracy = agreement.best_tie_epsilon([0.0, 0.1, 1.0])
    assert epsilon == 0.1
    assert accuracy == 1.0


def test_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(1)
    gold = rng.normal(size=200)
    metric = gold + rng.normal(scale=0.5, size=200)
    interval = PairwiseAgreement(gold, metric).bootstrap_accuracy(samples=200)
    assert interval.lower <= interval.estimate <= interval.upper


def test_paired_bootstrap_detects_a_consistent_improvement():
    rng = np.random.default_rng(2)
    stats_b = rng.uniform(40, 60, size=250)
    stats_a = stats_b + 2 + rng.normal(scale=1, size=250)
    result = PairedBootstrap(np.mean, samples=500).compare(stats_a, stats_b)
    assert result.delta > 0
    assert result.lower > 0
    assert result.p_value < 0.01


def test_paired_bootstrap_does_not_find_a_difference_between_identical_systems():
    stats = np.random.default_rng(3).uniform(40, 60, size=250)
    result = PairedBootstrap(np.mean, samples=200).compare(stats, stats.copy())
    assert result.delta == 0
    assert result.p_value == 1.0
