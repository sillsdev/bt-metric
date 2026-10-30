"""How much the two outputs in each system pair actually differ, so near-identical pairs can be set aside.

A pair whose outputs differ on only a few verses can't be ranked meaningfully by any metric or by humans.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class PredictionFiles:
    """Reads the predictions of system ids of the form "<experiment>|<prefix>|<step>"."""

    def __init__(self, experiments_root: Path) -> None:
        self._root = experiments_root

    def read(self, system: str) -> Optional[List[str]]:
        experiment, prefix, step = system.split("|")
        path = self._root / experiment / f"{prefix}.trg-predictions.detok.txt.{step}"
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
        except OSError as e:
            LOGGER.warning(f"Could not read {path}: {e}")
            return None
        return text[:-1].split("\n") if text.endswith("\n") else text.split("\n")


class PairDifferences:
    """Adds the number and fraction of verses on which each pair's outputs differ."""

    def __init__(self, files: PredictionFiles, workers: int) -> None:
        self._files = files
        self._workers = workers

    def measure(self, pairs: pd.DataFrame) -> pd.DataFrame:
        systems = pd.unique(pd.concat([pairs["system_a"], pairs["system_b"]]))
        LOGGER.info(f"Reading predictions of {len(systems)} system outputs")
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            predictions: Dict[str, Optional[List[str]]] = dict(zip(systems, executor.map(self._files.read, systems)))
        differing = np.array(
            [
                self._differing_verses(predictions[a], predictions[b])
                for a, b in zip(pairs["system_a"], pairs["system_b"])
            ],
            dtype=float,
        )
        return pairs.assign(differing_verses=differing, differing_fraction=differing / pairs["verses"])

    def _differing_verses(self, a: Optional[List[str]], b: Optional[List[str]]) -> float:
        if a is None or b is None or len(a) != len(b):
            return float("nan")
        return float(sum(line_a.strip() != line_b.strip() for line_a, line_b in zip(a, b)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how much the outputs in each system pair differ")
    parser.add_argument(
        "experiments_root", type=Path, help="The experiments directory, e.g. $SIL_NLP_DATA_PATH/MT/experiments"
    )
    parser.add_argument("--pairs", type=Path, default=Path("data/system_pairs.tsv"))
    parser.add_argument(
        "--min-differing-fraction",
        type=float,
        default=0.1,
        help="Pairs whose outputs differ on a smaller fraction of verses are flagged near_identical",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = pd.read_csv(args.pairs, sep="\t")
    pairs = PairDifferences(PredictionFiles(args.experiments_root), args.workers).measure(pairs)
    pairs["near_identical"] = pairs["differing_fraction"] < args.min_differing_fraction
    pairs.to_csv(args.pairs, sep="\t", index=False)

    quantiles = pairs.groupby("relation")["differing_fraction"].quantile([0.1, 0.25, 0.5, 0.75]).unstack()
    LOGGER.info(f"Fraction of verses that differ, by relation (quantiles):\n{quantiles.round(3).to_string()}")
    kept = pairs[~pairs["near_identical"] & pairs["differing_fraction"].notna()]
    LOGGER.info(
        f"{int(pairs['near_identical'].sum())} near-identical pairs, "
        f"{int(pairs['differing_fraction'].isna().sum())} unreadable; {len(kept)} usable pairs:\n"
        + kept.groupby(["relation", "test_design"]).size().to_string()
    )


if __name__ == "__main__":
    main()
