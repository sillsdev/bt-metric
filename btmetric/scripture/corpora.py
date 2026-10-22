"""Finds Bible extracts for the languages of the human-judgment test sets, for the simulated low-resource setup."""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerseCounts:
    total: int
    ot: int
    nt: int


@dataclass(frozen=True)
class CorpusCandidate:
    language: str
    extract: str
    path: str
    total: int
    ot: int
    nt: int
    selected: bool


class EvaluationLanguages:
    """The test-set languages, each with the ISO codes its extracts may be named with.

    A preferred extract overrides "most complete" where that would pick something other than a published
    translation, e.g. an English back translation.
    """

    def __init__(self, table: pd.DataFrame) -> None:
        self._codes = {row.language: row.extract_codes.split(",") for row in table.itertuples()}
        self._preferred = {
            row.language: row.preferred_extract for row in table.itertuples() if isinstance(row.preferred_extract, str)
        }

    def is_preferred(self, language: str, extract: str) -> bool:
        return self._preferred.get(language) == extract

    def has_preference(self, language: str) -> bool:
        return language in self._preferred

    @classmethod
    def load(cls) -> "EvaluationLanguages":
        return cls(pd.read_csv(Path(__file__).with_name("evaluation_languages.tsv"), sep="\t"))

    def language_of(self, extract_path: Path) -> Optional[str]:
        return self.language_for_code(extract_path.stem.split("-", 1)[0])

    def language_for_code(self, code: str) -> Optional[str]:
        """The test-set language an ISO code (two- or three-letter, or a variant such as "pbt") belongs to."""
        return next((language for language, codes in self._codes.items() if code in codes), None)

    def languages(self) -> List[str]:
        return list(self._codes)


class VerseCounter:
    """Verse counts per extract: from silnlp's verse_counts.csv where present, otherwise counted from the file.

    Extracts follow silnlp's vref.txt, one line per verse in original versification, so the line number tells
    which testament a verse is in: OT up to MAL 3:24 (line 23,213), NT up to REV 22:21 (line 31,170), then the
    deuterocanon.
    """

    def __init__(self, known_counts: Dict[str, VerseCounts]) -> None:
        self._known_counts = known_counts

    @classmethod
    def load(cls, verse_counts_path: Optional[Path]) -> "VerseCounter":
        if verse_counts_path is None or not verse_counts_path.is_file():
            return cls({})
        table = pd.read_csv(verse_counts_path, usecols=["file", "Total", "OT", "NT"])
        return cls({row.file: VerseCounts(row.Total, row.OT, row.NT) for row in table.itertuples()})

    def count(self, path: Path) -> VerseCounts:
        if path.stem in self._known_counts:
            return self._known_counts[path.stem]
        nt_start, nt_end = 23213, 31170
        with path.open(encoding="utf-8-sig", errors="replace") as file:
            non_empty = [i for i, line in enumerate(file) if line.strip() and line.strip() != "<range>"]
        ot = sum(1 for i in non_empty if i < nt_start)
        nt = sum(1 for i in non_empty if nt_start <= i < nt_end)
        return VerseCounts(len(non_empty), ot, nt)


class CorpusFinder:
    def __init__(self, languages: EvaluationLanguages, counter: VerseCounter, workers: int) -> None:
        self._languages = languages
        self._counter = counter
        self._workers = workers

    def candidates(self, scripture_dirs: List[Path]) -> pd.DataFrame:
        paths = [path for directory in scripture_dirs for path in sorted(directory.glob("*.txt"))]
        matched = [(path, language) for path in paths if (language := self._languages.language_of(path))]
        LOGGER.info(f"{len(matched)} of {len(paths)} extracts are in a test-set language; counting verses")
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            counts = list(executor.map(lambda item: self._counter.count(item[0]), matched))
        table = pd.DataFrame(
            [
                asdict(CorpusCandidate(language, path.stem, str(path), c.total, c.ot, c.nt, False))
                for (path, language), c in zip(matched, counts)
            ],
            columns=[field for field in CorpusCandidate.__dataclass_fields__],
        )
        # More verses than vref.txt has lines means the file isn't a verse-aligned extract (e.g. a test file).
        vref_line_count = 41899
        valid = table[table["total"] <= vref_line_count]
        preferred = [self._languages.is_preferred(row.language, row.extract) for row in valid.itertuples(index=False)]
        # A language's preferred extract if it was found, otherwise its most complete one; the "Bible-only"
        # auxiliary models train on it.
        ranked = valid.assign(preferred=preferred).sort_values(["preferred", "total", "nt"], ascending=False)
        best = ranked.groupby("language").head(1)
        table.loc[best.index, "selected"] = True
        for row in best.itertuples(index=False):
            if self._languages.has_preference(row.language) and not row.preferred:
                LOGGER.warning(f"Preferred extract for {row.language} not found; using {row.extract} instead.")
        return table.sort_values(["language", "total"], ascending=[True, False])

    def missing_languages(self, table: pd.DataFrame) -> List[str]:
        return sorted(set(self._languages.languages()) - set(table["language"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Find Bible extracts for the languages of the human-judgment sets")
    parser.add_argument(
        "--scripture", type=Path, nargs="+", default=[Path("M:/MT/scripture"), Path("M:/eBible/MT/scripture")]
    )
    parser.add_argument("--verse-counts", type=Path, default=Path("M:/MT/experiments/verse_counts/verse_counts.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/bible_corpora.tsv"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    finder = CorpusFinder(EvaluationLanguages.load(), VerseCounter.load(args.verse_counts), args.workers)
    table = finder.candidates(args.scripture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    selected = table[table["selected"]][["language", "extract", "total", "ot", "nt"]]
    LOGGER.info(f"Most complete extract per language -> {args.output}\n{selected.to_string(index=False)}")
    LOGGER.info(f"No extract found for: {', '.join(finder.missing_languages(table)) or 'none'}")


if __name__ == "__main__":
    main()
