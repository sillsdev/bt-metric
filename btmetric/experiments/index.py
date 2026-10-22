"""Index of the system outputs saved by silnlp experiments, keyed by the test set they were scored on."""

import argparse
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemOutput:
    """One system's predictions for one test set: an experiment, test file prefix and checkpoint step."""

    experiment: str
    prefix: str
    step: str
    test_hash: str
    verses: int
    prediction_hash: str
    config_hash: Optional[str]
    config_json: Optional[str]
    model: Optional[str]
    src: Optional[str]
    trg: Optional[str]
    test_design: str


class TestFiles:
    """The vref, reference and prediction files that share one test prefix in an experiment directory."""

    def __init__(self, directory: Path, prefix: str) -> None:
        self._directory = directory
        self._prefix = prefix

    def exist(self) -> bool:
        return self._vref_path().is_file() and self._reference_path().is_file()

    def test_hash(self) -> str:
        digest = hashlib.sha1()
        digest.update(_normalized_bytes(self._vref_path()))
        digest.update(_normalized_bytes(self._reference_path()))
        return digest.hexdigest()[:16]

    def verse_count(self) -> int:
        return _normalized_bytes(self._vref_path()).count(b"\n")

    def prediction_hash(self, step: str) -> str:
        return hashlib.sha1(_normalized_bytes(self.prediction_path(step))).hexdigest()[:16]

    def prediction_path(self, step: str) -> Path:
        return self._directory / f"{self._prefix}.trg-predictions.detok.txt.{step}"

    def _vref_path(self) -> Path:
        return self._directory / f"{self._prefix}.vref.txt"

    def _reference_path(self) -> Path:
        return self._directory / f"{self._prefix}.trg.detok.txt"


class ExperimentConfig:
    """The parts of an experiment's config.yml that identify the system and its test design."""

    def __init__(self, config: Optional[dict]) -> None:
        self._config = config if isinstance(config, dict) else None

    @classmethod
    def load(cls, directory: Path) -> "ExperimentConfig":
        path = directory / "config.yml"
        if not path.is_file():
            return cls(None)
        try:
            return cls(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
        except (yaml.YAMLError, UnicodeDecodeError) as e:
            LOGGER.warning(f"Could not parse {path}: {e}")
            return cls(None)

    def as_json(self) -> Optional[str]:
        return None if self._config is None else json.dumps(self._config, sort_keys=True, default=str)

    def config_hash(self) -> Optional[str]:
        config_json = self.as_json()
        return None if config_json is None else hashlib.sha1(config_json.encode("utf-8")).hexdigest()[:16]

    def model(self) -> Optional[str]:
        return None if self._config is None else self._config.get("model")

    def sources(self) -> Optional[str]:
        return self._join_pair_field("src")

    def targets(self) -> Optional[str]:
        return self._join_pair_field("trg")

    def test_design(self) -> str:
        """'books' if whole books are held out for testing, 'random' if a random verse sample is."""
        pairs = self._corpus_pairs()
        if not pairs:
            return "unknown"
        if any("test_books" in pair for pair in pairs):
            return "books"
        return "random"

    def _corpus_pairs(self) -> List[dict]:
        if self._config is None:
            return []
        pairs = (self._config.get("data") or {}).get("corpus_pairs") or []
        return [pair for pair in pairs if isinstance(pair, dict)]

    def _join_pair_field(self, field: str) -> Optional[str]:
        names: List[str] = []
        for pair in self._corpus_pairs():
            value = pair.get(field)
            names.extend(value if isinstance(value, list) else [value] if value else [])
        return "+".join(dict.fromkeys(str(name) for name in names)) or None


class ExperimentDirectory:
    """A directory containing test predictions written by silnlp.nmt.test."""

    def __init__(self, root: Path, path: Path, file_names: List[str]) -> None:
        self._root = root
        self._path = path
        self._file_names = file_names

    def outputs(self) -> List[SystemOutput]:
        config = ExperimentConfig.load(self._path)
        experiment = self._path.relative_to(self._root).as_posix()
        outputs: List[SystemOutput] = []
        for prefix, step in self._prediction_files():
            test_files = TestFiles(self._path, prefix)
            if not test_files.exist():
                continue
            try:
                outputs.append(
                    SystemOutput(
                        experiment=experiment,
                        prefix=prefix,
                        step=step,
                        test_hash=test_files.test_hash(),
                        verses=test_files.verse_count(),
                        prediction_hash=test_files.prediction_hash(step),
                        config_hash=config.config_hash(),
                        config_json=config.as_json(),
                        model=config.model(),
                        src=config.sources(),
                        trg=config.targets(),
                        test_design=config.test_design(),
                    )
                )
            except OSError as e:
                LOGGER.warning(f"Could not read test files for {experiment}/{prefix} step {step}: {e}")
        return outputs

    def _prediction_files(self) -> Iterator[Tuple[str, str]]:
        pattern = re.compile(r"^(test(?:\.[^.]+\.[^.]+)?)\.trg-predictions\.detok\.txt\.(\w+)$")
        for name in self._file_names:
            match = pattern.match(name)
            if match:
                yield match.group(1), match.group(2)


class ExperimentScanner:
    """Finds experiment directories with test predictions under a root, down to a maximum depth."""

    def __init__(self, root: Path, max_depth: int, workers: int) -> None:
        self._root = root
        self._max_depth = max_depth
        self._workers = workers

    def scan(self) -> pd.DataFrame:
        directories = list(self._directories())
        LOGGER.info(f"Found {len(directories)} directories with predictions; reading test files")
        # Reads go to a network-mounted bucket, so they are I/O bound and parallelize well across threads.
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            batches = executor.map(lambda directory: directory.outputs(), directories)
            rows = [asdict(output) for batch in batches for output in batch]
        return pd.DataFrame(rows, columns=[field for field in SystemOutput.__dataclass_fields__])

    def _directories(self) -> Iterator[ExperimentDirectory]:
        for dir_path, dir_names, file_names in os.walk(self._root):
            path = Path(dir_path)
            if len(path.relative_to(self._root).parts) >= self._max_depth:
                dir_names.clear()
            # Checkpoint and run directories hold model weights, never test predictions.
            dir_names[:] = [name for name in dir_names if not name.startswith((".", "checkpoint", "run"))]
            if any(".trg-predictions.detok.txt." in name for name in file_names):
                yield ExperimentDirectory(self._root, path, file_names)


def _normalized_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the test predictions of silnlp experiments")
    parser.add_argument("root", type=Path, help="The experiments directory, e.g. M:/MT/experiments")
    parser.add_argument("--output", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--max-depth", type=int, default=4, help="How many directory levels below root to search")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = ExperimentScanner(args.root, args.max_depth, args.workers).scan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(
        f"Indexed {len(index)} system outputs in {index['experiment'].nunique()} experiments "
        f"with {index['test_hash'].nunique()} distinct test sets to {args.output}"
    )


if __name__ == "__main__":
    main()
