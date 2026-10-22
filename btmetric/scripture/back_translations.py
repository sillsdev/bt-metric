"""Pairs extracted human back translations with extracts of the vernacular projects they translate."""

import argparse
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParatextProject:
    folder: str
    name: str
    guid: str
    iso: str
    translation_type: str
    base_name: Optional[str]
    base_guid: Optional[str]

    @classmethod
    def from_settings(cls, folder: Path) -> Optional["ParatextProject"]:
        settings_path = folder / "Settings.xml"
        try:
            root = ET.parse(settings_path).getroot()
        except (OSError, ET.ParseError):
            return None
        guid = (root.findtext("Guid") or "").strip()
        if not guid:
            return None
        # TranslationInfo is "<type>:<base project name>:<base project guid>", e.g. "BackTranslation:ABC:7774...".
        info = (root.findtext("TranslationInfo") or "").split(":")
        return cls(
            folder=folder.name,
            name=(root.findtext("Name") or folder.name).strip(),
            guid=guid,
            iso=(root.findtext("LanguageIsoCode") or "").split(":")[0],
            translation_type=info[0] if info else "",
            base_name=info[1] if len(info) > 1 and info[1] else None,
            base_guid=info[2] if len(info) > 2 and info[2] else None,
        )

    def is_back_translation(self) -> bool:
        return self.translation_type == "BackTranslation" and self.base_guid is not None


@dataclass(frozen=True)
class ScriptureExtract:
    """An extract file in MT/scripture, named <iso>-<project>[_<yyyy_mm_dd>][<variant>].txt."""

    stem: str
    iso: str
    project: str
    date: Optional[str]

    @classmethod
    def from_stem(cls, stem: str) -> Optional["ScriptureExtract"]:
        if "-" not in stem:
            return None
        iso, project = stem.split("-", 1)
        date = re.search(r"_(\d{4}_\d{2}_\d{2})", project)
        return cls(stem, iso, project, date.group(1) if date else None)


class ProjectCatalog:
    """Paratext projects and the extracts that belong to each of them.

    A project is identified by GUID and name together: copies made from a template keep the original's GUID,
    so some unrelated projects share one.
    """

    def __init__(self, projects: List[ParatextProject]) -> None:
        self._projects = projects
        self._key_by_folder = {project.folder.lower(): self.key(project) for project in projects}
        self._keys_by_name: Dict[str, Set[str]] = {}
        for project in projects:
            for name in (project.name, project.folder):
                self._keys_by_name.setdefault(name.lower(), set()).add(self.key(project))

    @classmethod
    def load(cls, projects_dir: Path, workers: int) -> "ProjectCatalog":
        folders = [path for path in projects_dir.iterdir() if path.is_dir()]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            projects = [project for project in executor.map(ParatextProject.from_settings, folders) if project]
        LOGGER.info(f"Read settings for {len(projects)} of {len(folders)} Paratext projects")
        return cls(projects)

    def key(self, project: ParatextProject) -> str:
        return f"{project.guid}:{project.name.lower()}"

    def base_key(self, back_translation: ParatextProject) -> Optional[str]:
        """The key of the project a back translation translates, allowing for the base project being renamed."""
        key = f"{back_translation.base_guid}:{(back_translation.base_name or '').lower()}"
        if any(self.key(project) == key for project in self._projects):
            return key
        candidates = {self.key(project) for project in self._projects if project.guid == back_translation.base_guid}
        return next(iter(candidates)) if len(candidates) == 1 else None

    def back_translations(self) -> List[ParatextProject]:
        """One entry per back translation project, although dated copies of its folder may exist."""
        by_key = {self.key(project): project for project in self._projects if project.is_back_translation()}
        return list(by_key.values())

    def owner_key(self, extract: ScriptureExtract) -> Optional[str]:
        """The key of the project an extract came from, or None if it can't be told apart from another project."""
        project = extract.project.lower()
        if project in self._key_by_folder:
            return self._key_by_folder[project]
        # Otherwise the longest project name the extract starts with; "ABC_BTE_2023_11_30" must go to ABC_BTE, not ABC.
        for end in sorted({match.start() for match in re.finditer("_", project)} | {len(project)}, reverse=True):
            keys = self._keys_by_name.get(project[:end])
            if keys:
                return next(iter(keys)) if len(keys) == 1 else None
        return None


@dataclass(frozen=True)
class BackTranslationPair:
    back_translation: str
    base: str
    bt_extract: str
    base_extract: str
    date_match: bool


class BackTranslationPairer:
    """Pairs each back translation extract with an extract of its base project, from the same date if possible."""

    def __init__(self, catalog: ProjectCatalog, extracts: List[ScriptureExtract]) -> None:
        self._catalog = catalog
        self._extracts_by_key: Dict[str, List[ScriptureExtract]] = {}
        for extract in extracts:
            key = catalog.owner_key(extract)
            if key is not None:
                self._extracts_by_key.setdefault(key, []).append(extract)

    def pairs(self) -> pd.DataFrame:
        rows = [asdict(pair) for bt in self._catalog.back_translations() for pair in self._project_pairs(bt)]
        return pd.DataFrame(rows, columns=[field for field in BackTranslationPair.__dataclass_fields__])

    def _project_pairs(self, bt: ParatextProject) -> Iterator[BackTranslationPair]:
        base_key = self._catalog.base_key(bt)
        bt_extracts = self._extracts_by_key.get(self._catalog.key(bt), [])
        base_extracts = self._extracts_by_key.get(base_key, []) if base_key else []
        if not bt_extracts or not base_extracts:
            return
        latest_base = max(base_extracts, key=lambda extract: (extract.date or "", -len(extract.stem)))
        for bt_extract in bt_extracts:
            same_date = [extract for extract in base_extracts if bt_extract.date and extract.date == bt_extract.date]
            # Prefer the plain extract over variants such as transliterations ("_Latn") or "-m" files.
            base_extract = min(same_date, key=lambda extract: len(extract.stem)) if same_date else latest_base
            yield BackTranslationPair(bt.name, bt.base_name or "", bt_extract.stem, base_extract.stem, bool(same_date))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair human back translation extracts with their base projects")
    parser.add_argument("--projects", type=Path, default=Path("M:/Paratext/projects"))
    parser.add_argument("--scripture", type=Path, default=Path("M:/MT/scripture"))
    parser.add_argument("--output", type=Path, default=Path("data/back_translations.tsv"))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    catalog = ProjectCatalog.load(args.projects, args.workers)
    extracts = [ScriptureExtract.from_stem(path.stem) for path in args.scripture.glob("*.txt")]
    pairs = BackTranslationPairer(catalog, [extract for extract in extracts if extract]).pairs()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(
        f"{len(catalog.back_translations())} back translation projects; "
        f"{pairs['back_translation'].nunique()} have extracts paired with their base project "
        f"({len(pairs)} extract pairs, {int(pairs['date_match'].sum())} from the same date) -> {args.output}"
    )


if __name__ == "__main__":
    main()
