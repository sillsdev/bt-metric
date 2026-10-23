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
