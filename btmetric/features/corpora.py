"""Training text for the features that train models (aligners, language models).

Models must be trained in the domain of the text they score: Bible text for Scripture, news for WMT. Low-resource
conditions are simulated by keeping the training text small (about a Bible's worth), not by changing its domain.
Each provider returns its text with a key naming exactly what it contains, so models can be cached and reused.
"""

import hashlib
import io
import logging
import random
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from ..paths import EBIBLE_SCRIPTURE_DIR, SCRIPTURE_DIR
from ..scripture.corpora import EvaluationLanguages
from ..scripture.verses import parallel_verses

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallelText:
    source: List[str]
    target: List[str]
    key: str


@dataclass(frozen=True)
class MonolingualText:
    lines: List[str]
    key: str


class TrainingCorpus(ABC):
    @abstractmethod
    def parallel(self, source_language: str, target_language: str, excluded: FrozenSet[str]) -> Optional[ParallelText]:
        """Parallel text for a language pair, without the excluded segments."""

    @abstractmethod
    def monolingual(self, language: str, excluded: FrozenSet[str]) -> Optional[MonolingualText]:
        """Text in one language, without the excluded segments."""


def _digest(items: FrozenSet[str]) -> str:
    return hashlib.sha1("\n".join(sorted(items)).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class BibleCorpus(TrainingCorpus):
    """Verse-aligned Scripture extracts. A language is an extract name ("qaa-ABC_2024_04_22") or an ISO code, which
    resolves to the extract btmetric-find-bible-corpora selected for it. Excluded segments are verse references."""

    # data/scripture holds extracts built here, such as the original-language Bible.
    scripture_dirs: Tuple[Path, ...] = (SCRIPTURE_DIR, EBIBLE_SCRIPTURE_DIR, Path("data/scripture"))
    corpora_table: Path = Path("data/bible_corpora.tsv")
    vref_path: Path = Path("../silnlp/silnlp/assets/vref.txt")

    def parallel(self, source_language: str, target_language: str, excluded: FrozenSet[str]) -> Optional[ParallelText]:
        source_path, target_path = self.extract_path(source_language), self.extract_path(target_language)
        if source_path is None or target_path is None:
            LOGGER.warning(f"No Bible text for {source_language} or {target_language}")
            return None
        verses = parallel_verses(
            self.vref_path.read_text(encoding="utf-8").splitlines(),
            source_path.read_text(encoding="utf-8-sig").splitlines(),
            target_path.read_text(encoding="utf-8-sig").splitlines(),
            excluded,
        )
        # The key says verse ranges are merged, so models cached from text that dropped them aren't reused.
        key = f"bible:{source_path.stem}:{target_path.stem}:{_digest(excluded)}:merged-ranges"
        return ParallelText([source for _, source, _ in verses], [target for _, _, target in verses], key)

    def monolingual(self, language: str, excluded: FrozenSet[str]) -> Optional[MonolingualText]:
        path = self.extract_path(language)
        if path is None:
            LOGGER.warning(f"No Bible text for {language}")
            return None
        vrefs = self.vref_path.read_text(encoding="utf-8").splitlines()
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        verses = [
            line.strip()
            for vref, line in zip(vrefs, lines)
            if line.strip() and line.strip() != "<range>" and vref not in excluded
        ]
        return MonolingualText(verses, f"bible:{path.stem}:{_digest(excluded)}")

    def extract_path(self, language: str) -> Optional[Path]:
        # Experiment configs don't always name an extract exactly as it is stored: some add a suffix
        # ("qac-ABCD-QAC" for qac-ABCD.txt) and many extracts carry a date ("qab-XYZ_2023_08_22.txt").
        parts = language.split("-")
        names = ["-".join(parts[:end]) for end in range(len(parts), 1, -1)]
        for name in names:
            for directory in self.scripture_dirs:
                if (directory / f"{name}.txt").is_file():
                    return directory / f"{name}.txt"
                dated = sorted(directory.glob(f"{name}_20*.txt"))
                if dated:
                    return dated[-1]
        evaluation_language = EvaluationLanguages.load().language_for_code(language.split("-")[0])
        if evaluation_language is None or not self.corpora_table.is_file():
            return None
        table = pd.read_csv(self.corpora_table, sep="\t")
        selected = table[(table["language"] == evaluation_language) & table["selected"]]
        return Path(selected["path"].iloc[0]) if not selected.empty else None


class NewsCorpus(TrainingCorpus):
    """News-domain training text for the WMT and AfriMTE test sets, about a Bible's worth per language pair.

    Where OPUS has News Commentary for the pair, a fixed sample of it, with every sentence that appears in any test
    set removed up front (so the excluded segments of a call never matter). Otherwise the pool of source-reference
    pairs from all judged sets of the pair, minus the excluded segments: the caller scores a set in folds, so each
    fold's models are trained on the rest. Excluded segments are source sentences.
    """

    def __init__(
        self,
        pool: Dict[Tuple[str, str], List[Tuple[str, str]]],
        cache_dir: Path = Path("data/external/opus"),
        sample_size: int = 30000,
        min_pairs: int = 500,
    ) -> None:
        self._languages = EvaluationLanguages.load()
        self._pool = {(self._code(s), self._code(t)): pairs for (s, t), pairs in pool.items()}
        self._cache_dir = cache_dir
        self._sample_size = sample_size
        self._min_pairs = min_pairs
        self._test_sentences: Set[str] = {text for pairs in pool.values() for pair in pairs for text in pair}
        self._opus: Dict[Tuple[str, str], Optional[List[Tuple[str, str]]]] = {}

    def parallel(self, source_language: str, target_language: str, excluded: FrozenSet[str]) -> Optional[ParallelText]:
        source, target = self._code(source_language), self._code(target_language)
        opus = self._opus_pairs(source, target)
        if opus:
            return ParallelText([s for s, _ in opus], [t for _, t in opus], f"opus:{source}-{target}")
        pairs = [(s, t) for s, t in self._pool.get((source, target), []) if s not in excluded]
        if len(pairs) < self._min_pairs:
            LOGGER.warning(f"Only {len(pairs)} in-domain pairs for {source}-{target}")
            return None
        return ParallelText([s for s, _ in pairs], [t for _, t in pairs], f"pool:{source}-{target}:{_digest(excluded)}")

    def monolingual(self, language: str, excluded: FrozenSet[str]) -> Optional[MonolingualText]:
        code = self._code(language)
        for source, target in [(s, t) for (s, t) in self._pool if code in (s, t)]:
            opus = self._opus_pairs(source, target)
            if opus:
                lines = [t if code == target else s for s, t in opus]
                return MonolingualText(lines, f"opus:{source}-{target}:{code}")
        # The references of the pool, minus those whose source is being scored.
        lines = sorted(
            {
                reference
                for (_, target_language), pairs in self._pool.items()
                if target_language == code
                for source, reference in pairs
                if source not in excluded
            }
        )
        if len(lines) < self._min_pairs:
            LOGGER.warning(f"Only {len(lines)} in-domain sentences for {code}")
            return None
        return MonolingualText(lines, f"pool:{code}:{_digest(excluded)}")

    def _opus_pairs(self, source: str, target: str) -> Optional[List[Tuple[str, str]]]:
        key = (source, target)
        if key not in self._opus:
            self._opus[key] = self._load_opus(source, target)
        return self._opus[key]

    def _load_opus(self, source: str, target: str) -> Optional[List[Tuple[str, str]]]:
        first, second = sorted((source, target))
        directory = self._cache_dir / "News-Commentary" / f"{first}-{second}.sample{self._sample_size}"
        if not directory.is_dir():
            lines = self._download(first, second)
            if lines is None:
                return None
            # Sentences that appear in any test set are removed, so no scored segment is ever trained on.
            usable = [
                (a, b) for a, b in lines if a and b and a not in self._test_sentences and b not in self._test_sentences
            ]
            random.Random(0).shuffle(usable)
            sample = usable[: self._sample_size]
            directory.mkdir(parents=True, exist_ok=True)
            for index, language in enumerate((first, second)):
                text = "\n".join(pair[index] for pair in sample) + "\n"
                (directory / f"{language}.txt").write_text(text, encoding="utf-8")
        read = {
            language: (directory / f"{language}.txt").read_text(encoding="utf-8").splitlines()
            for language in (first, second)
        }
        return list(zip(read[source], read[target]))

    def _download(self, first: str, second: str) -> Optional[List[Tuple[str, str]]]:
        url = f"https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/{first}-{second}.txt.zip"
        try:
            with urllib.request.urlopen(url) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
        except Exception as e:  # noqa: BLE001 - a missing pair is expected, not an error
            LOGGER.info(f"No News Commentary for {first}-{second} ({type(e).__name__})")
            return None
        # Moses format: one file per language, named "<corpus>.<pair>.<language>".
        names = {name.rsplit(".", 1)[-1]: name for name in archive.namelist() if name.count(".") >= 2}
        if first not in names or second not in names:
            return None
        LOGGER.info(f"Downloaded News Commentary {first}-{second}")
        first_lines = archive.read(names[first]).decode("utf-8").splitlines()
        second_lines = archive.read(names[second]).decode("utf-8").splitlines()
        return [(a.strip(), b.strip()) for a, b in zip(first_lines, second_lines)]

    def _code(self, language: str) -> str:
        # AfriMTE uses three-letter codes ("eng", "hau"); OPUS and WMT use two-letter ones where they exist.
        return self._languages.language_for_code(language) or language
