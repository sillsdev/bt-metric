"""Test sets, system outputs and human judgments in one shape, whatever dataset they come from."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationSet:
    """Source segments and their references. Segment ids are verse references for Scripture, indices otherwise."""

    name: str
    source_language: str
    target_language: str
    sources: List[str]
    references: List[str]
    segment_ids: List[str]

    def __post_init__(self) -> None:
        if not len(self.sources) == len(self.references) == len(self.segment_ids):
            raise ValueError(f"{self.name}: sources, references and segment ids differ in length.")

    def __len__(self) -> int:
        return len(self.sources)


class SegmentJudgments:
    """Human scores for individual hypotheses, for segment-level meta-evaluation.

    Rows that share an item are different hypotheses for the same source segment, so they can be compared with
    each other directly. Scores are higher-is-better.
    """

    def __init__(self, name: str, table: pd.DataFrame, source_language: str = "", target_language: str = "") -> None:
        missing = {"item", "source", "hypothesis", "reference", "human"} - set(table.columns)
        if missing:
            raise ValueError(f"{name}: segment judgments are missing columns {sorted(missing)}.")
        self.name = name
        self.source_language = source_language
        self.target_language = target_language
        self._table = table.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._table)

    def column(self, name: str) -> List:
        return self._table[name].tolist()

    def human(self) -> np.ndarray:
        return self._table["human"].to_numpy(dtype=float)

    def items(self) -> np.ndarray:
        return self._table["item"].to_numpy()

    def systems(self) -> np.ndarray:
        """Which system produced each hypothesis, where known (WMT DA and AfriMTE don't say)."""
        return self._table["system"].to_numpy() if "system" in self._table else np.full(len(self._table), "")

    def sample_items(self, max_items: int, seed: int = 0) -> "SegmentJudgments":
        """A random subset of whole items, keeping every hypothesis of each chosen item."""
        items = self._table["item"].unique()
        if len(items) <= max_items:
            return self
        chosen = np.random.default_rng(seed).choice(items, size=max_items, replace=False)
        return SegmentJudgments(
            self.name, self._table[self._table["item"].isin(chosen)], self.source_language, self.target_language
        )


@dataclass
class JudgedOutputs:
    """Every system's hypotheses for one evaluation set, with human scores per segment where they exist.

    Human scores are higher-is-better; segments a system wasn't judged on are NaN.
    """

    evaluation_set: EvaluationSet
    hypotheses: Dict[str, List[str]] = field(default_factory=dict)
    human_scores: Dict[str, np.ndarray] = field(default_factory=dict)

    def add_system(self, system: str, hypotheses: List[str], human_scores: Optional[np.ndarray] = None) -> None:
        if len(hypotheses) != len(self.evaluation_set):
            raise ValueError(
                f"{self.evaluation_set.name}: {system} has {len(hypotheses)} hypotheses "
                f"for {len(self.evaluation_set)} segments."
            )
        self.hypotheses[system] = hypotheses
        if human_scores is not None:
            self.human_scores[system] = np.asarray(human_scores, dtype=float)

    def systems(self) -> List[str]:
        return list(self.hypotheses)

    def human_system_score(self, system: str) -> float:
        return float(np.nanmean(self.human_scores[system]))

    def without_systems(self, systems: List[str]) -> "JudgedOutputs":
        kept = JudgedOutputs(self.evaluation_set)
        for system in self.systems():
            if system not in systems:
                kept.add_system(system, self.hypotheses[system], self.human_scores.get(system))
        return kept

    def segment_judgments(self) -> SegmentJudgments:
        """The same judgments flattened to one row per system and segment, items being segments."""
        rows = [
            {
                "item": i,
                "source": self.evaluation_set.sources[i],
                "hypothesis": hypothesis,
                "reference": self.evaluation_set.references[i],
                "human": self.human_scores[system][i],
                "system": system,
            }
            for system in self.systems()
            if system in self.human_scores
            for i, hypothesis in enumerate(self.hypotheses[system])
        ]
        table = pd.DataFrame(rows, columns=["item", "source", "hypothesis", "reference", "human", "system"])
        return SegmentJudgments(
            self.evaluation_set.name,
            table.dropna(subset=["human"]),
            self.evaluation_set.source_language,
            self.evaluation_set.target_language,
        )
