from pathlib import Path

import numpy as np
import pandas as pd

from btmetric.analysis.silver_agreement import (
    SilverTable,
    XlmrLanguages,
    agreement_report,
    silver_mean,
    system_level_agreement,
    within_verse_agreement,
)
from btmetric.meta_eval.runner import FeatureRegistry


def test_xlmr_languages_recognize_two_and_three_letter_codes():
    languages = XlmrLanguages.load()
    assert languages.code("en-NIV11R") == "en"
    assert languages.code("swh-ONEN+en-BSB") == "sw"
    assert languages.code("cmn-CUV") == "zh"
    assert languages.code("qaa-ABC_2024_04_22") is None


def table() -> pd.DataFrame:
    # Two verses, three systems; the metric agrees with the silver label except on one pair in verse v2.
    return pd.DataFrame(
        {
            "test_hash": "t",
            "vref": ["v1", "v1", "v1", "v2", "v2", "v2"],
            "system": ["a", "b", "c", "a", "b", "c"],
            "silver": [0.9, 0.5, 0.1, 0.8, 0.6, 0.2],
            "metric": [70.0, 50.0, 10.0, 45.0, 60.0, 20.0],
        }
    )


def test_within_verse_agreement_compares_systems_on_the_same_verse():
    assert within_verse_agreement(table(), "silver", "metric").accuracy() == 5 / 6


def test_system_level_agreement_ranks_systems_by_mean_score():
    assert system_level_agreement(table(), "silver", "metric").accuracy() == 1.0


def test_silver_mean_averages_standardized_labels():
    frame = table().assign(other=lambda t: t["silver"] * 100)
    assert np.allclose(silver_mean(frame, ["silver", "other"]), silver_mean(frame, ["silver"]))


def test_report_adds_the_mean_of_several_silver_labels():
    frame = table().assign(other=lambda t: t["silver"] * 2)
    report = agreement_report(frame, ["silver", "other"], ["metric"])
    assert set(report["silver"]) == {"silver", "other", "silver-mean"}


def test_score_adds_feature_columns_per_test_set(tmp_path: Path):
    path = tmp_path / "rows.tsv"
    rows = pd.DataFrame(
        {
            "test_hash": ["t1", "t1", "t2"],
            "source_extract": "en-X",
            "target_extract": "es-Y",
            "language_pair": "en-es",
            "vref": ["v1", "v1", "v1"],
            "system": ["a", "b", "a"],
            "source": "s",
            "hypothesis": ["la casa", "el perro", "la casa"],
            "reference": "la casa",
        }
    )
    rows.to_csv(path, sep="\t", index=False)
    scored = SilverTable(path).score(["chrf3"], FeatureRegistry())
    assert scored["chrF3"].tolist()[0] == 100.0 and scored["chrF3"].tolist()[1] < 100.0
    assert "chrF3" in SilverTable(path).load().columns
