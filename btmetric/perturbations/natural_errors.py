"""Challenge rows made of real errors: for one verse, an MT output showing an error signature paired with another
system's output for the same verse that shows none.

The labels are noisier than synthetic perturbations, since the clean output isn't guaranteed to be better overall,
but the errors are the ones MT systems actually make.
"""

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..evaluation.data import JudgedOutputs
from ..evaluation.loaders import ExperimentOutputsLoader
from .calibration import ErrorSignatures
from .challenge_set import ChallengeRow, HeldOutTestSets

LOGGER = logging.getLogger(__name__)


class NaturalErrorPairs:
    """Pairs a faulty output with a clean one for the same verse, per error signature."""

    def __init__(
        self,
        signatures: ErrorSignatures,
        error_types: List[str],
        max_pairs_per_type: int,
        seed: int,
        clean_length_ratio: tuple = (0.8, 1.25),
    ) -> None:
        self._signatures = signatures
        self._error_types = error_types
        self._max_pairs_per_type = max_pairs_per_type
        self._rng = np.random.default_rng(seed)
        self._clean_length_ratio = clean_length_ratio

    def rows_for(self, outputs: JudgedOutputs, target: str) -> List[ChallengeRow]:
        evaluation_set = outputs.evaluation_set
        if len(outputs.systems()) < 2:
            return []
        by_type: Dict[str, List[ChallengeRow]] = {error_type: [] for error_type in self._error_types}
        for i, reference in enumerate(evaluation_set.references):
            if not reference.strip():
                continue
            faulty: Dict[str, List[str]] = {error_type: [] for error_type in self._error_types}
            clean: List[str] = []
            for hypotheses in outputs.hypotheses.values():
                flags = self._signatures.detect(evaluation_set.sources[i], hypotheses[i], reference)
                found = [error_type for error_type in self._error_types if flags[error_type]]
                for error_type in found:
                    faulty[error_type].append(hypotheses[i])
                if not any(flags.values()) and self._plausible_length(hypotheses[i], reference):
                    clean.append(hypotheses[i])
            if not clean:
                continue
            for error_type, candidates in faulty.items():
                if candidates:
                    by_type[error_type].append(
                        ChallengeRow(
                            evaluation_set.name,
                            target,
                            evaluation_set.segment_ids[i],
                            f"natural_{error_type}",
                            1,
                            evaluation_set.sources[i],
                            reference,
                            clean[int(self._rng.integers(len(clean)))],
                            candidates[int(self._rng.integers(len(candidates)))],
                        )
                    )
        rows: List[ChallengeRow] = []
        for candidates in by_type.values():
            if len(candidates) > self._max_pairs_per_type:
                chosen = self._rng.choice(len(candidates), size=self._max_pairs_per_type, replace=False)
                candidates = [candidates[int(k)] for k in sorted(chosen)]
            rows.extend(candidates)
        return rows

    def _plausible_length(self, hypothesis: str, reference: str) -> bool:
        low, high = self._clean_length_ratio
        return low <= len(hypothesis.strip()) / max(1, len(reference.strip())) <= high


def main() -> None:
    parser = argparse.ArgumentParser(description="Build challenge rows from real MT errors")
    parser.add_argument("--experiments-root", type=Path, default=Path("M:/MT/experiments"))
    parser.add_argument("--index", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--test-sets", type=int, default=60)
    parser.add_argument("--max-pairs-per-type", type=int, default=50, help="Per test set and error type")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/natural_errors.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = pd.read_csv(args.index, sep="\t", dtype={"step": str})
    loader = ExperimentOutputsLoader(args.experiments_root, index)
    pairs = NaturalErrorPairs(
        ErrorSignatures(), ["loop", "too_long", "too_short", "source_copy"], args.max_pairs_per_type, args.seed
    )
    rows: List[ChallengeRow] = []
    for test_hash, target in HeldOutTestSets(index).held_out_books(args.test_sets, args.seed).items():
        try:
            rows.extend(pairs.rows_for(loader.outputs_for(test_hash), target))
        except (OSError, ValueError, KeyError) as e:
            LOGGER.warning(f"Skipping {test_hash}: {e}")
    table = pd.DataFrame([asdict(row) for row in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(
        f"{len(table)} natural error pairs from {table['test_hash'].nunique()} test sets -> {args.output}\n"
        + table["perturbation"].value_counts().to_string()
    )


if __name__ == "__main__":
    main()
