"""Pairs of system outputs that were scored on the same test set, for system-level meta-evaluation."""

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemPair:
    test_hash: str
    verses: int
    test_design: str
    system_a: str
    system_b: str
    relation: str
    differences: str


class FlatConfig:
    """An experiment config flattened to dotted keys, so two configs can be compared key by key."""

    def __init__(self, config_json: Optional[str]) -> None:
        self._values: Dict[str, str] = {}
        if isinstance(config_json, str):
            self._flatten("", json.loads(config_json))

    def differing_keys(self, other: "FlatConfig") -> List[str]:
        keys = self._values.keys() | other._values.keys()
        return sorted(key for key in keys if self._values.get(key) != other._values.get(key))

    def _flatten(self, prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                self._flatten(f"{prefix}{key}.", child)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                self._flatten(f"{prefix}{i}.", child)
        else:
            self._values[prefix.rstrip(".")] = json.dumps(value, default=str)


class SystemPairFinder:
    """Finds every pair of distinct system outputs that share a test set in an experiment index."""

    def __init__(self, index: pd.DataFrame) -> None:
        self._index = index

    def pairs(self) -> pd.DataFrame:
        rows = [asdict(pair) for _, group in self._index.groupby("test_hash") for pair in self._group_pairs(group)]
        return pd.DataFrame(rows, columns=[field for field in SystemPair.__dataclass_fields__])

    def _group_pairs(self, group: pd.DataFrame) -> Iterator[SystemPair]:
        # Identical predictions say nothing about which system is better, so keep one output per prediction.
        outputs = group.drop_duplicates("prediction_hash").to_dict("records")
        configs = {id(output): FlatConfig(output["config_json"]) for output in outputs}
        for a, b in itertools.combinations(outputs, 2):
            yield SystemPair(
                test_hash=a["test_hash"],
                verses=int(a["verses"]),
                test_design=a["test_design"] if a["test_design"] == b["test_design"] else "mixed",
                system_a=self._system_id(a),
                system_b=self._system_id(b),
                relation=self._relation(a, b),
                differences=";".join(configs[id(a)].differing_keys(configs[id(b)])),
            )

    def _relation(self, a: dict, b: dict) -> str:
        """How the two systems differ: checkpoints of one experiment, reruns of one config, or different configs."""
        if a["experiment"] == b["experiment"] and a["prefix"] == b["prefix"]:
            return "checkpoint"
        if isinstance(a["config_hash"], str) and a["config_hash"] == b["config_hash"]:
            return "rerun"
        return "config"

    def _system_id(self, output: dict) -> str:
        return f"{output['experiment']}|{output['prefix']}|{output['step']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find pairs of system outputs scored on the same test set")
    parser.add_argument("--index", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/system_pairs.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = pd.read_csv(args.index, sep="\t", dtype={"step": str})
    pairs = SystemPairFinder(index).pairs()
    pairs.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(f"Found {len(pairs)} pairs in {pairs['test_hash'].nunique()} test sets; written to {args.output}")
    LOGGER.info("Pairs by relation and test design:\n" + pairs.groupby(["relation", "test_design"]).size().to_string())


if __name__ == "__main__":
    main()
