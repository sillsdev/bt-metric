from pathlib import Path

from btmetric.analysis.mqm_error_categories import RawAnnotations, normalize


def test_raw_annotations_label_outputs_by_their_error_categories(tmp_path: Path):
    rows = [
        "system\tdoc\tdoc_id\tseg_id\trater\tsource\ttarget\tcategory\tseverity",
        "A\td\t1\t1\tr1\tHello world.\tHallo <v>Welt</v>.\tAccuracy/Omission\tMajor",
        "A\td\t1\t1\tr2\tHello world.\tHallo Welt.\tFluency/Grammar\tMinor",
        "B\td\t1\t1\tr1\tHello world.\tHallo  Welt!\tno-error\tno-error",
        "C\td\t1\t1\tr1\tHello world.\tHallo schöne Welt.\tAccuracy/Addition\tMinor",
        "D\td\t1\t1\tr1\tHello world.\tHallo, Welt.\tFluency/Punctuation\tMinor",
    ]
    path = tmp_path / "raw.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    annotations = RawAnnotations.load(path)
    assert normalize("Hallo <v>Welt</v>.") == "Hallo Welt."
    assert annotations.label("Hello world.", "Hallo Welt.") == "omission"
    assert annotations.label("Hello world.", "Hallo Welt!") == "clean"
    assert annotations.label("Hello world.", "Hallo schöne Welt.") == "addition"
    assert annotations.label("Hello world.", "Hallo, Welt.") == ""
    assert annotations.label("Hello world.", "unannotated") == ""


def test_major_only_ignores_minor_omissions_and_additions(tmp_path: Path):
    rows = [
        "system\tdoc\tdoc_id\tseg_id\trater\tsource\ttarget\tcategory\tseverity",
        "A\td\t1\t1\tr1\tHello world.\tHallo.\tAccuracy/Omission\tmajor",
        "C\td\t1\t1\tr1\tHello world.\tHallo schöne Welt.\tAccuracy/Addition\tMinor",
    ]
    path = tmp_path / "raw.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    annotations = RawAnnotations.load(path)
    assert annotations.label("Hello world.", "Hallo.", major_only=True) == "omission"
    assert annotations.label("Hello world.", "Hallo schöne Welt.", major_only=True) == ""
