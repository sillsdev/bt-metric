import numpy as np
import pandas as pd

from btmetric.analysis.significance import AgreementCurve, threshold_table


def test_agreement_curve_recovers_a_rising_probability():
    rng = np.random.default_rng(0)
    differences = rng.uniform(0, 6, size=4000)
    true_probability = 1 / (1 + np.exp(-(0.2 + 0.6 * differences)))
    agrees = rng.random(4000) < true_probability
    curve = AgreementCurve().fit(differences, agrees)
    assert abs(curve.slope - 0.6) < 0.1 and abs(curve.intercept - 0.2) < 0.15
    assert 0.5 < curve.probability(0.0) < 0.6 and curve.probability(5.0) > 0.9


def test_difference_needed_rises_with_the_confidence_asked_for():
    curve = AgreementCurve()
    curve.intercept, curve.slope = 0.0, 1.0
    assert curve.difference_for(0.5) == 0.0
    assert curve.difference_for(0.9) > curve.difference_for(0.8) > 0
    curve.slope = -0.1
    assert curve.difference_for(0.8) == float("inf")


def test_threshold_table_reports_pairs_agreement_and_thresholds():
    pairs = pd.DataFrame(
        {"difference": [0.1, 0.2, 3.0, 4.0, 5.0, 6.0], "agrees": [False, True, True, True, True, True]}
    )
    row = threshold_table(pairs, [0.8])
    assert row["pairs"] == 6 and abs(row["agreement"] - 5 / 6) < 1e-9 and "diff_for_80%" in row


def test_bible_pairs_compare_metric_and_silver_preferences_within_each_test_set():
    from btmetric.analysis.significance import score_pairs

    scores = pd.DataFrame(
        {
            "dataset": ["t1", "t1", "t1", "t2", "t2"],
            "silver": [0.9, 0.5, 0.5, 0.2, 0.3],
            "chrF-B": [60.0, 50.0, 55.0, 40.0, 30.0],
        }
    )
    pairs = score_pairs(scores, "chrF-B")
    # t1 has one silver tie (systems 2 and 3), which is skipped; t2's metric disagrees with the silver label.
    assert sorted(pairs["difference"].tolist()) == [5.0, 10.0, 10.0]
    assert pairs.groupby("dataset")["agrees"].all().to_dict() == {"t1": True, "t2": False}


def test_bands_cap_each_test_set_and_interval_the_agreement():
    from btmetric.analysis.significance import band_table

    # t1 has 20 small pairs that all disagree; capped at 10 they weigh as much as t2's 10 agreeing pairs.
    pairs = pd.DataFrame(
        {
            "dataset": ["t1"] * 20 + ["t2"] * 10 + ["t3"] * 4,
            "difference": [0.2] * 30 + [-5.0] * 4,
            "agrees": [False] * 20 + [True] * 10 + [True, True, True, False],
        }
    )
    capped = band_table(pairs, cap=10, resamples=200).set_index("band")
    assert capped.loc["0-0.5", "pairs"] == 30
    assert abs(capped.loc["0-0.5", "agreement"] - 0.5) < 1e-9
    assert abs(capped.loc["0-0.5", "agreement_uncapped"] - 1 / 3) < 1e-9
    assert capped.loc["0-0.5", "lower"] <= 0.5 <= capped.loc["0-0.5", "upper"]
    assert abs(capped.loc["over 4", "agreement"] - 0.75) < 1e-9
    assert capped.loc["1-2", "pairs"] == 0 and np.isnan(capped.loc["1-2", "agreement"])
    uncapped = band_table(pairs, cap=None, resamples=10).set_index("band")
    assert abs(uncapped.loc["0-0.5", "agreement"] - 1 / 3) < 1e-9
