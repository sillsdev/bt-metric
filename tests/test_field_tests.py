from pathlib import Path

import openpyxl
import pandas as pd

from btmetric.analysis.field_test_agreement import comparison_preferences, scoring_rows, verse_preferences
from btmetric.datasets.field_tests import (
    _VerseReader,
    attach_references,
    expand_vref,
    preference,
    read_workbook,
    usable,
)


def test_responses_map_to_labels_in_each_language():
    assert preference("Draft A is better.") == "A"
    assert preference("Draf B lebih baik") == "B"
    assert preference("Both drafts are equally good.") == "good"
    assert preference("Keduanya sama-sama buruknya") == "poor"
    assert usable("Correct the draft verse") == 1
    assert usable("Perbiaki Ayat AI") == 1
    assert usable("Corriger") == 1
    assert usable("Start over from the source Bible") == 0
    assert usable("Recommencer") == 0
    assert usable("") is None


def test_verse_numbers_follow_the_passage_heading_into_its_next_chapter():
    reader = _VerseReader("MAT 17 and 20")
    assert [reader.read(v) for v in ["1.0", "27", "1", "5-6"]] == ["MAT 17:1", "MAT 17:27", "MAT 20:1", "MAT 20:5-6"]
    reader = _VerseReader("Kisah 17:21-28")
    assert reader.read("21.0") == "ACT 17:21"
    assert _VerseReader("Verse").read("LUK 10:19") == "LUK 10:19"
    assert expand_vref("MAT 20:5-6") == ["MAT 20:5", "MAT 20:6"]


def test_comparison_sheets_are_read_past_their_titles_and_blank_responses(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Draft Comparison - Team 2 (Group 1)"])
    sheet.append([])
    sheet.append(["Mark 10:17-31", "NIV11", "Draft A", "Draft B", "Response"])
    sheet.append([17, "As Jesus started", "a17", "b17", "Draft A is better."])
    sheet.append([18, "Why do you call me", "a18", "b18", None])
    path = tmp_path / "comparison.xlsx"
    workbook.save(path)
    [(name, kind, rows)] = list(read_workbook(path))
    assert kind == "comparison"
    assert rows[["vref", "source", "draft_a", "draft_b"]].values.tolist() == [
        ["MRK 10:17", "As Jesus started", "a17", "b17"]
    ]


def test_references_come_from_the_latest_extract_and_count_as_independent_only_if_they_predate_the_trial(
    tmp_path: Path,
):
    (tmp_path / "vref.txt").write_text("MRK 1:1\nMRK 1:2\n", encoding="utf-8")
    (tmp_path / "xyz-P_2025_08_01.txt").write_text("old one\n\n", encoding="utf-8")
    (tmp_path / "xyz-P_2025_12_01.txt").write_text("new one\nnew two\n", encoding="utf-8")
    table = pd.DataFrame({"trial": "nt_field_test", "target_extract": "xyz-P", "vref": ["MRK 1:1", "MRK 1:2"]})
    table = attach_references(table, [tmp_path], tmp_path / "vref.txt")
    assert table["reference"].tolist() == ["new one", "new two"]
    assert table["reference_first_seen"].tolist() == ["2025-08-01", "2025-12-01"]
    assert table["independent_reference"].tolist() == [True, False]


def test_metric_agreement_with_preferences_per_verse_and_per_comparison():
    judgments = pd.DataFrame(
        {
            "comparison": "c",
            "team": "t",
            "kind": "comparison",
            "label": ["A", "B", "good"],
            "source_extract": "s",
            "target_extract": "x",
            "vref": ["MRK 1:1", "MRK 1:2", "MRK 1:3"],
            "source": "src",
            "draft_a": ["a1", "a2", "a3"],
            "draft_b": ["b1", "b2", "b3"],
            "reference": ["r1", "r2", "r3"],
            "human_translation": "",
            "independent_reference": "True",
        }
    )
    rows = scoring_rows(judgments)
    assert rows["system"].tolist() == ["A", "B"] * 3
    rows["metric"] = [0.9, 0.1, 0.2, 0.8, 0.5, 0.4]
    assert verse_preferences(rows, "metric").accuracy() == 1.0
    [summary] = comparison_preferences(rows, "metric").to_dict("records")
    assert summary["verses"] == 3 and summary["human_margin"] == 0.0


def test_a_team_trialled_separately_uses_its_own_cutoff(tmp_path: Path):
    (tmp_path / "vref.txt").write_text("MRK 1:1\n", encoding="utf-8")
    (tmp_path / "xyz-P_2025_06_01.txt").write_text("one\n", encoding="utf-8")
    table = pd.DataFrame(
        {"trial": "nt_field_test", "target_extract": "xyz-P", "vref": "MRK 1:1", "trial_start": ["", "2025-05-01"]}
    )
    table = attach_references(table, [tmp_path], tmp_path / "vref.txt")
    assert table["independent_reference"].tolist() == [True, False]
