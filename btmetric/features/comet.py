"""COMET-family baselines (COMET, xCOMET, AfriCOMET), scored by a worker process in tools/comet.

unbabel-comet needs numpy<2, so it runs in its own uv environment; see tools/comet/pyproject.toml.
"""

import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import Feature

LOGGER = logging.getLogger(__name__)


class CometWorker:
    """A long-lived tools/comet/worker.py process with one model loaded, exchanging JSON lines."""

    def __init__(self, model: str, batch_size: int) -> None:
        self._model = model
        self._batch_size = batch_size
        self._process: Optional[subprocess.Popen] = None

    def score(self, request: dict) -> List[float]:
        process = self._started()
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline() or '{"error": "worker exited"}')
        if "error" in response:
            raise RuntimeError(f"COMET worker ({self._model}) failed: {response['error']}")
        return response["scores"]

    def close(self) -> None:
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.close()
            self._process.wait()
            self._process = None

    def _started(self) -> subprocess.Popen:
        if self._process is None:
            tool_dir = Path(__file__).resolve().parents[2] / "tools" / "comet"
            command = ["uv", "run", "--project", str(tool_dir), "python", str(tool_dir / "worker.py"), self._model]
            # The worker has its own environment; an inherited VIRTUAL_ENV would only make uv warn.
            env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
            LOGGER.info(f"Starting COMET worker for {self._model}")
            self._process = subprocess.Popen(
                command + ["--batch-size", str(self._batch_size)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
            )
            assert self._process.stdout is not None
            ready = json.loads(self._process.stdout.readline() or "{}")
            if "ready" not in ready:
                raise RuntimeError(f"COMET worker for {self._model} did not start.")
        return self._process


class ScoreCache:
    """Segment scores on disk (SQLite), keyed by model and (source, hypothesis, reference).

    Each chunk is committed as soon as it is scored, so a run that crashes or is stopped resumes where it left off
    instead of starting over.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        # Write-ahead logging keeps the file consistent if the process dies mid-write.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS scores (model TEXT, key TEXT, score REAL, PRIMARY KEY (model, key))"
        )

    def get(self, model: str, keys: Sequence[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], float]:
        digests = {self._digest(key): key for key in keys}
        found: Dict[Tuple[str, str, str], float] = {}
        names = list(digests)
        # SQLite limits the number of parameters in one statement.
        for start in range(0, len(names), 900):
            chunk = names[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            rows = self._connection.execute(
                f"SELECT key, score FROM scores WHERE model = ? AND key IN ({placeholders})", [model, *chunk]
            )
            found.update((digests[digest], score) for digest, score in rows)
        return found

    def put(self, model: str, scores: Dict[Tuple[str, str, str], float]) -> None:
        self._connection.executemany(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
            [(model, self._digest(key), score) for key, score in scores.items()],
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _digest(self, key: Tuple[str, str, str]) -> str:
        return hashlib.sha1("\x1f".join(key).encode("utf-8")).hexdigest()


class Comet(Feature):
    """A COMET model's segment scores; the system score is their mean, as COMET reports it.

    Scores are kept in memory and in a ScoreCache on disk, keyed by (source, hypothesis, reference): meta-evaluation
    scores the same outputs at system and segment level, and a stopped run should resume, not start over. New
    segments go to the worker in chunks, logging progress after each.
    """

    criterion = "accuracy"

    def __init__(
        self,
        name: str,
        model: str,
        uses_reference: bool = True,
        batch_size: int = 64,
        cache_path: Optional[Path] = Path("data/cache/comet_scores.sqlite"),
        chunk_size: int = 2048,
    ) -> None:
        self.name = name
        self._model = model
        self._uses_reference = uses_reference
        self._worker = CometWorker(model, batch_size)
        self._memory: Dict[Tuple[str, str, str], float] = {}
        self._disk = ScoreCache(cache_path) if cache_path else None
        self._chunk_size = chunk_size

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        keys = [
            (source, hypothesis, reference if self._uses_reference else "")
            for source, hypothesis, reference in zip(sources, hypotheses, references)
        ]
        missing = list(dict.fromkeys(key for key in keys if key not in self._memory))
        if missing and self._disk:
            from_disk = self._disk.get(self._model, missing)
            self._memory.update(from_disk)
            missing = [key for key in missing if key not in from_disk]
        if missing:
            self._score(missing, cached=len(set(keys)) - len(missing))
        return np.array([self._memory[key] for key in keys])

    def close(self) -> None:
        self._worker.close()
        if self._disk:
            self._disk.close()

    def _score(self, missing: List[Tuple[str, str, str]], cached: int) -> None:
        started = time.monotonic()
        for start in range(0, len(missing), self._chunk_size):
            chunk = missing[start : start + self._chunk_size]
            request = {"src": [key[0] for key in chunk], "mt": [key[1] for key in chunk]}
            if self._uses_reference:
                request["ref"] = [key[2] for key in chunk]
            scores = dict(zip(chunk, self._worker.score(request)))
            self._memory.update(scores)
            if self._disk:
                self._disk.put(self._model, scores)
            done = start + len(chunk)
            rate = done / max(time.monotonic() - started, 1e-9)
            remaining = (len(missing) - done) / rate if rate else 0.0
            LOGGER.info(
                f"{self.name}: {done:,}/{len(missing):,} new segments scored ({cached:,} already cached), "
                f"{rate:.0f}/s, about {remaining / 60:.0f} min left"
            )
