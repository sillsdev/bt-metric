"""Do the features catch the omissions and additions that expert annotators marked? (PLAN.md 2.2.2)

Google's raw WMT MQM release keeps each annotation's error category. For every source sentence, an output that an
annotator marked with Accuracy/Omission (or Accuracy/Addition) is paired with another system's output for the same
sentence that every annotator marked No-error, and each feature is checked for preferring the clean one. Aligners
and language models train on in-domain news text (corpora.NewsCorpus).
"""

import argparse
import logging
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ..evaluation.loaders import WmtMqmLoader
from ..features.corpora import NewsCorpus
from ..meta_eval.runner import FeatureRegistry

LOGGER = logging.getLogger(__name__)

RAW_FILES = {
    "wmt20-mqm-en-de": "newstest2020/ende/mqm_newstest2020_ende.tsv",
    "wmt20-mqm-zh-en": "newstest2020/zhen/mqm_newstest2020_zhen.tsv",
    "wmt21-mqm-en-de": "newstest2021/ende/mqm_newstest2021_ende.tsv",
    "wmt21-mqm-zh-en": "newstest2021/zhen/mqm_newstest2021_zhen.tsv",
    "wmt22-mqm-en-de": "generalMT2022/ende/mqm_generalMT2022_ende.tsv",
    "wmt22-mqm-en-ru": "generalMT2022/enru/mqm_generalMT2022_enru.tsv",
    "wmt22-mqm-zh-en": "generalMT2022/zhen/mqm_generalMT2022_zhen.tsv",
}


def normalize(text: str) -> str:
    # Raw targets mark error spans with <v>...</v>; whitespace differs between the two releases.
    return re.sub(r"\s+", " ", re.sub(r"</?v>", "", str(text))).strip()


class RawAnnotations:
    """Error categories and severities per (source, output), from the raw MQM release."""

    def __init__(self, annotations: Dict[Tuple[str, str], Set[Tuple[str, str]]]) -> None:
        self._annotations = annotations

    @classmethod
    def load(cls, path: Path) -> "RawAnnotations":
        table = pd.read_csv(path, sep="\t", quoting=3, keep_default_na=False, on_bad_lines="skip")
        annotations: Dict[Tuple[str, str], Set[Tuple[str, str]]] = defaultdict(set)
        for row in table.itertuples(index=False):
            # Capitalization differs by year ("Major" in 2020-21, "major" in 2022; "No-error" and "no-error").
            labels = (str(row.category).lower(), str(row.severity).lower())
            annotations[(normalize(row.source), normalize(row.target))].add(labels)
        return cls(annotations)

    def label(self, source: str, output: str, major_only: bool = False) -> str:
        """'omission', 'addition', 'clean' (every annotation No-error), or '' (other errors, or not annotated).

        With major_only, an omission or addition counts only if an annotator rated it Major; minor ones are often
        a single word, which no coverage measure is meant to notice.
        """
        found = self._annotations.get((normalize(source), normalize(output)))
        if not found:
            return ""
        for name in ("omission", "addition"):
            if any(name in category and (not major_only or severity == "major") for category, severity in found):
                return name
        return "clean" if {category for category, _ in found} <= {"no-error"} else ""


def download(raw_dir: Path) -> None:
    for relative in RAW_FILES.values():
        path = raw_dir / Path(relative).name
        if not path.is_file():
            url = f"https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/{relative}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)


def labelled_pairs(mqm_csv: Path, raw_dir: Path, major_only: bool = False) -> pd.DataFrame:
    """Rows: one per (faulty, clean) pair of outputs for the same source sentence."""
    rows = []
    for outputs in WmtMqmLoader(mqm_csv).judged_outputs():
        name = outputs.evaluation_set.name
        raw_name = next((key for key in RAW_FILES if name.startswith(key)), None)
        if raw_name is None:
            continue
        annotations = RawAnnotations.load(raw_dir / Path(RAW_FILES[raw_name]).name)
        evaluation_set = outputs.evaluation_set
        for i, (source, reference) in enumerate(zip(evaluation_set.sources, evaluation_set.references)):
            labelled: Dict[str, List[str]] = defaultdict(list)
            for system, hypotheses in outputs.hypotheses.items():
                if system.startswith(("ref", "Human")):
                    continue
                label = annotations.label(source, hypotheses[i], major_only)
                if label:
                    labelled[label].append(hypotheses[i])
            for error in ("omission", "addition"):
                for faulty in labelled[error]:
                    for clean in labelled["clean"]:
                        rows.append(
                            {
                                "dataset": name,
                                "source_language": evaluation_set.source_language,
                                "target_language": evaluation_set.target_language,
                                "error": error,
                                "source": source,
                                "reference": reference,
                                "clean": clean,
                                "faulty": faulty,
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check features against MQM-annotated omissions and additions")
    parser.add_argument("--mqm", type=Path, default=Path("data/external/wmt-mqm/wmt-mqm-human-evaluation.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/wmt-mqm-raw"))
    parser.add_argument(
        "--features",
        nargs="+",
        default=[
            "chrf3", "bleu", "coverage-source", "coverage-target", "coverage-reference",
            "coverage-source-detector", "coverage-target-detector", "coverage-reference-detector",
        ],
    )  # fmt: skip
    parser.add_argument("--output", type=Path, default=Path("data/results/mqm_error_categories.tsv"))
    parser.add_argument("--major-only", action="store_true", help="Count only omissions and additions rated Major")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    download(args.raw_dir)
    pairs = labelled_pairs(args.mqm, args.raw_dir, args.major_only)
    LOGGER.info(f"{len(pairs)} labelled pairs:\n{pairs.groupby(['dataset', 'error']).size().to_string()}")
    # The scored sentences, so that any of them in News Commentary are dropped from the training sample.
    pool = {
        languages: list(zip(rows["source"], rows["reference"]))
        for languages, rows in pairs.groupby(["source_language", "target_language"])
    }
    registry = FeatureRegistry(NewsCorpus(pool))

    results = []
    for name in args.features:
        feature = registry.get(name)
        try:
            clean_scores = np.full(len(pairs), np.nan)
            faulty_scores = np.full(len(pairs), np.nan)
            for (source_language, target_language), rows in pairs.groupby(["source_language", "target_language"]):
                feature.prepare(source_language, target_language)
                index = rows.index.to_numpy()
                sources, references = rows["source"].tolist(), rows["reference"].tolist()
                clean_scores[index] = feature.segment_scores(sources, rows["clean"].tolist(), references)
                faulty_scores[index] = feature.segment_scores(sources, rows["faulty"].tolist(), references)
        finally:
            feature.close()
        for error, rows in pairs.groupby("error"):
            index = rows.index.to_numpy()
            clean, faulty = clean_scores[index], faulty_scores[index]
            scored = ~(np.isnan(clean) | np.isnan(faulty))
            results.append(
                {
                    "feature": feature.name,
                    "error": error,
                    "pairs": int(scored.sum()),
                    "prefers_clean": float(np.mean(clean[scored] > faulty[scored])),
                    "ties": float(np.mean(clean[scored] == faulty[scored])),
                    "prefers_faulty": float(np.mean(clean[scored] < faulty[scored])),
                }
            )
    table = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(f"Clean vs. annotated-faulty outputs of the same source:\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
