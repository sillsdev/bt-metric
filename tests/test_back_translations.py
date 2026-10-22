from pathlib import Path
from typing import Optional

from btmetric.scripture.back_translations import BackTranslationPairer, ProjectCatalog, ScriptureExtract


def write_project(projects_dir: Path, folder: str, name: str, guid: str, iso: str, info: str) -> None:
    (projects_dir / folder).mkdir(parents=True)
    (projects_dir / folder / "Settings.xml").write_text(
        f"<ScriptureText><Name>{name}</Name><Guid>{guid}</Guid><LanguageIsoCode>{iso}:::</LanguageIsoCode>"
        f"<TranslationInfo>{info}</TranslationInfo></ScriptureText>",
        encoding="utf-8",
    )


def catalog(tmp_path: Path) -> ProjectCatalog:
    write_project(tmp_path, "ABC_2023_11_30", "ABC", "base", "qaa", "Standard::")
    write_project(tmp_path, "ABC_2024_04_22", "ABC", "base", "qaa", "Standard::")
    write_project(tmp_path, "ABCBT", "ABCBT", "bt", "en", "BackTranslation:ABC:base")
    write_project(tmp_path, "ABCBT_2024_04_22", "ABCBT", "bt", "en", "BackTranslation:ABC:base")
    write_project(tmp_path, "ABC_BTE__2023_11_30", "ABC_BTE", "bt2", "en", "BackTranslation:ABC:base")
    # Created from the same template as ABC, so it shares the GUID but is a different project.
    write_project(tmp_path, "OtherNT", "OtherNT", "base", "abc", "Standard::")
    return ProjectCatalog.load(tmp_path, workers=2)


def extract(stem: str) -> ScriptureExtract:
    result: Optional[ScriptureExtract] = ScriptureExtract.from_stem(stem)
    assert result is not None
    return result


def test_extract_names_give_iso_project_and_date():
    parsed = extract("qaa-ABC_2023_11_30_Latn")
    assert (parsed.iso, parsed.project, parsed.date) == ("qaa", "ABC_2023_11_30_Latn", "2023_11_30")


def test_extracts_belong_to_the_longest_matching_project_name(tmp_path: Path):
    projects = catalog(tmp_path)
    assert projects.owner_key(extract("en-ABC_BTE_2023_11_30")) == "bt2:abc_bte"
    assert projects.owner_key(extract("en-ABCBT_2024_04_22")) == "bt:abcbt"
    assert projects.owner_key(extract("qaa-ABC_2023_11_30_Latn")) == "base:abc"
    assert projects.owner_key(extract("en-NIV11R")) is None


def test_back_translations_pair_with_same_date_base_extracts_or_the_latest(tmp_path: Path):
    stems = [
        "qaa-ABC_2023_11_30",
        "qaa-ABC_2023_11_30_Latn",
        "qaa-ABC_2024_04_22",
        "abc-OtherNT_2026_10_21",
        "en-ABCBT_2023_11_30",
        "en-ABCBT_2026_10_21",
    ]
    pairs = BackTranslationPairer(catalog(tmp_path), [extract(stem) for stem in stems]).pairs()
    by_bt = dict(zip(pairs["bt_extract"], zip(pairs["base_extract"], pairs["date_match"])))
    assert by_bt == {
        "en-ABCBT_2023_11_30": ("qaa-ABC_2023_11_30", True),
        "en-ABCBT_2026_10_21": ("qaa-ABC_2024_04_22", False),
    }
