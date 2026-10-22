"""COMET-family baselines (COMET, xCOMET, AfriCOMET), scored by a worker process in tools/comet.

unbabel-comet needs numpy<2, so it runs in its own uv environment; see tools/comet/pyproject.toml.
"""

import json
import logging
import os
import subprocess
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


class Comet(Feature):
    """A COMET model's segment scores; the system score is their mean, as COMET reports it.

    Scores are cached per (source, hypothesis, reference), since meta-evaluation scores the same outputs at
    system and segment level and model inference is the expensive part.
    """

    criterion = "accuracy"

    def __init__(self, name: str, model: str, uses_reference: bool = True, batch_size: int = 64) -> None:
        self.name = name
        self._uses_reference = uses_reference
        self._worker = CometWorker(model, batch_size)
        self._cache: Dict[Tuple[str, str, str], float] = {}

    def segment_scores(
        self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]
    ) -> np.ndarray:
        keys = [
            (source, hypothesis, reference if self._uses_reference else "")
            for source, hypothesis, reference in zip(sources, hypotheses, references)
        ]
        missing = list(dict.fromkeys(key for key in keys if key not in self._cache))
        if missing:
            request = {"src": [key[0] for key in missing], "mt": [key[1] for key in missing]}
            if self._uses_reference:
                request["ref"] = [key[2] for key in missing]
            self._cache.update(zip(missing, self._worker.score(request)))
        return np.array([self._cache[key] for key in keys])

    def close(self) -> None:
        self._worker.close()
