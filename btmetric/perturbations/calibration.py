"""How often real MT outputs show the errors the challenge set simulates, so its mix of perturbations can match.

Only errors with a reference-free or length-based signature can be counted this way: omission and truncation show
up as outputs much shorter than the reference, additions as much longer ones, decoder loops as repeated n-grams the
reference doesn't have, and untranslated output as text closer to the source than to the reference. Wrong key
terms, negation errors and wrong meaning need alignment or human judgment and aren't counted here.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

from ..evaluation.data import JudgedOutputs
from ..evaluation.loaders import ExperimentOutputsLoader
from ..paths import EXPERIMENTS_DIR
from .challenge_set import HeldOutTestSets

LOGGER = logging.getLogger(__name__)


class ErrorSignatures:
    def __init__(self, short_ratio: float = 0.6, long_ratio: float = 1.5, ngram: int = 3) -> None:
        self._short_ratio = short_ratio
        self._long_ratio = long_ratio
        self._ngram = ngram
        self._chrf = CHRF()

    def detect(self, source: str, hypothesis: str, reference: str) -> Dict[str, bool]:
        ratio = len(hypothesis.strip()) / max(1, len(reference.strip()))
        return {
            "empty": not hypothesis.strip(),
            "too_short": 0 < ratio < self._short_ratio,
            "too_long": ratio > self._long_ratio,
            "repetition": self._repeated(hypothesis) > self._repeated(reference),
            "loop": self._has_loop(hypothesis) and not self._has_loop(reference),
            "source_copy": self._is_source_copy(source, hypothesis, reference),
        }

    def _has_loop(self, text: str) -> bool:
        """A span repeated back to back, the signature of a decoder loop. Repeats anywhere in the verse also catch
        legitimate parallelism ("you and your descendants ... you and your descendants"), so this is the strict
        count. A single word must occur three times in a row: twice is reduplication in many languages."""
        words = [word.strip(".,;:!?\"'()").lower() for word in text.split()]
        for length in range(1, 7):
            repeats_needed = 3 if length == 1 else 2
            for start in range(len(words) - length * repeats_needed + 1):
                span = words[start : start + length]
                if all(words[start + k * length : start + (k + 1) * length] == span for k in range(repeats_needed)):
                    return True
        return False

    def _repeated(self, text: str) -> int:
        words = text.lower().split()
        ngrams = Counter(tuple(words[i : i + self._ngram]) for i in range(len(words) - self._ngram + 1))
        return sum(count - 1 for count in ngrams.values() if count > 1)

    def _is_source_copy(self, source: str, hypothesis: str, reference: str) -> bool:
        if not source.strip() or not hypothesis.strip():
            return False
        to_source = self._chrf.sentence_score(hypothesis, [source]).score
        to_reference = self._chrf.sentence_score(hypothesis, [reference]).score
        return to_source > 50 and to_source > to_reference + 10


class ErrorRates:
    def __init__(self, signatures: ErrorSignatures, verses_per_output: int, seed: int) -> None:
        self._signatures = signatures
        self._verses_per_output = verses_per_output
        self._rng = np.random.default_rng(seed)

    def for_outputs(self, outputs: JudgedOutputs, target: str) -> List[dict]:
        evaluation_set = outputs.evaluation_set
        indices = [i for i, ref in enumerate(evaluation_set.references) if ref.strip()]
        if len(indices) > self._verses_per_output:
            indices = self._rng.choice(indices, size=self._verses_per_output, replace=False)
        rows = []
        for system, hypotheses in outputs.hypotheses.items():
            flags = [
                self._signatures.detect(evaluation_set.sources[i], hypotheses[i], evaluation_set.references[i])
                for i in indices
            ]
            rates = pd.DataFrame(flags).mean().to_dict()
            rows.append({"test_hash": evaluation_set.name, "target": target, "system": system, **rates})
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how often real MT outputs show each error signature")
    parser.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--index", type=Path, default=Path("data/experiments.tsv"))
    parser.add_argument("--test-sets", type=int, default=60)
    parser.add_argument("--verses-per-output", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/results/error_rates.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    index = pd.read_csv(args.index, sep="\t", dtype={"step": str})
    loader = ExperimentOutputsLoader(args.experiments_root, index)
    rates = ErrorRates(ErrorSignatures(), args.verses_per_output, args.seed)
    rows: List[dict] = []
    for test_hash, target in HeldOutTestSets(index).held_out_books(args.test_sets, args.seed).items():
        try:
            outputs = loader.outputs_for(test_hash)
        except (OSError, ValueError, KeyError) as e:
            LOGGER.warning(f"Skipping {test_hash}: {e}")
            continue
        rows.extend(rates.for_outputs(outputs, target))
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    signature_columns = ["empty", "too_short", "too_long", "repetition", "loop", "source_copy"]
    # Each test set counts once, however many systems it has.
    per_test_set = table.groupby("test_hash")[signature_columns].mean()
    LOGGER.info(
        f"{len(table)} outputs in {len(per_test_set)} test sets. Share of verses with each signature:\n"
        f"{per_test_set.describe().loc[['mean', '25%', '50%', '75%']].round(3).to_string()}"
    )


if __name__ == "__main__":
    main()
