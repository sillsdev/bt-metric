"""Loads the downloaded human-judgment datasets and silnlp experiment outputs into the common data model."""

import logging
import re
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd

from .data import EvaluationSet, JudgedOutputs, SegmentJudgments

LOGGER = logging.getLogger(__name__)


class WmtMqmLoader:
    """WMT MQM judgments (RicardoRei/wmt-mqm-human-evaluation): one row per source segment and system."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path

    def judged_outputs(self) -> Iterator[JudgedOutputs]:
        table = pd.read_csv(self._csv_path).fillna({"mt": ""})
        for (year, lp), group in table.groupby(["year", "lp"]):
            name = f"wmt{year % 100}-mqm-{lp}"
            if self._has_full_coverage(group):
                yield self._build(name, lp, group)
            else:
                # Some years judged different systems per domain (2021: news vs. TED), so each domain is its own set.
                for domain, domain_group in group.groupby("domain"):
                    yield self._build(f"{name}-{domain}", lp, domain_group)

    def _has_full_coverage(self, group: pd.DataFrame) -> bool:
        systems_per_segment = group.groupby(["domain", "src"])["system"].nunique()
        return bool((systems_per_segment == group["system"].nunique()).all())

    def _build(self, name: str, lp: str, group: pd.DataFrame) -> JudgedOutputs:
        # Segments are keyed by source: in some years the reference given varies by system (refA or refB), so the
        # reference most rows use is the one the metric scores against. A sentence repeated within the test set
        # gets one segment, with its scores averaged per system.
        keys = ["domain", "src"]
        references = group.groupby(keys, sort=False)["ref"].agg(lambda refs: refs.mode().iloc[0])
        scores = group.groupby(keys + ["system"], sort=False).agg(mt=("mt", "first"), score=("score", "mean"))
        systems = scores.index.get_level_values("system").unique()
        covered = scores.groupby(keys).size() == len(systems)
        segments = references[covered.reindex(references.index, fill_value=False)]
        source_language, target_language = lp.split("-")
        evaluation_set = EvaluationSet(
            name,
            source_language,
            target_language,
            segments.index.get_level_values("src").tolist(),
            segments.tolist(),
            [f"{domain}:{i}" for i, domain in enumerate(segments.index.get_level_values("domain"))],
        )
        outputs = JudgedOutputs(evaluation_set)
        for system in systems:
            rows = scores.xs(system, level="system").loc[segments.index]
            outputs.add_system(system, rows["mt"].tolist(), rows["score"].to_numpy())
        return outputs


class WmtDaLoader:
    """WMT direct assessment judgments (RicardoRei/wmt-da-human-evaluation), which carry no system names."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path

    def segment_judgments(self) -> Iterator[SegmentJudgments]:
        table = pd.read_csv(self._csv_path, usecols=["lp", "src", "mt", "ref", "score", "year"])
        table = table.fillna({"src": "", "mt": "", "ref": ""})
        for (year, lp), group in table.groupby(["year", "lp"]):
            # Hypotheses of the same source and reference come from different systems: they form one item.
            items = group.groupby(["src", "ref"], sort=False).ngroup()
            yield SegmentJudgments(
                f"wmt{year % 100}-da-{lp}",
                pd.DataFrame(
                    {
                        "item": items.to_numpy(),
                        "source": group["src"].to_numpy(),
                        "hypothesis": group["mt"].to_numpy(),
                        "reference": group["ref"].to_numpy(),
                        "human": group["score"].to_numpy(),
                    }
                ),
                *lp.split("-"),
            )


class AfriMteLoader:
    """AfriMTE-ade-devtest-v2: per language pair, parallel src/hyp-1/ref/score files with one system."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def segment_judgments(self) -> Iterator[SegmentJudgments]:
        for score_path in sorted(self._directory.glob("*.score.txt")):
            lp = score_path.name.split(".")[-3]
            prefix = score_path.name[: -len(".score.txt")]
            source, hypothesis, reference = (
                self._lines(self._directory / f"{prefix}.{kind}.txt") for kind in ("src", "hyp-1", "ref")
            )
            human = np.array([float(line) for line in self._lines(score_path)])
            yield SegmentJudgments(
                f"afrimte-{lp}",
                pd.DataFrame(
                    {
                        "item": np.arange(len(human)),
                        "source": source,
                        "hypothesis": hypothesis,
                        "reference": reference,
                        "human": human,
                    }
                ),
                *lp.split("-"),
            )

    def _lines(self, path: Path) -> List[str]:
        return path.read_text(encoding="utf-8").rstrip("\n").split("\n")


class ExperimentOutputsLoader:
    """System outputs from silnlp experiments sharing one test set, as listed in the experiment index.

    There are no human scores, so these support metric computation and the label-free checks only.
    """

    def __init__(self, experiments_root: Path, index: pd.DataFrame) -> None:
        self._root = experiments_root
        self._index = index

    def outputs_for(self, test_hash: str) -> JudgedOutputs:
        rows = self._index[self._index["test_hash"] == test_hash].drop_duplicates("prediction_hash")
        if rows.empty:
            raise KeyError(f"No experiment outputs with test set {test_hash}.")
        first = rows.iloc[0]
        directory = self._root / first["experiment"]
        evaluation_set = EvaluationSet(
            name=test_hash,
            source_language=str(first["src"]),
            target_language=str(first["trg"]),
            sources=self._lines(self._source_path(directory, first["prefix"])),
            references=self._lines(directory / f"{first['prefix']}.trg.detok.txt"),
            segment_ids=self._lines(directory / f"{first['prefix']}.vref.txt"),
        )
        outputs = JudgedOutputs(evaluation_set)
        for row in rows.itertuples(index=False):
            path = self._root / row.experiment / f"{row.prefix}.trg-predictions.detok.txt.{row.step}"
            hypotheses = self._lines(path)
            # Outputs with extra line breaks or merged verses can't be aligned verse by verse; leaving them out
            # keeps the test set's other outputs usable.
            if len(hypotheses) != len(evaluation_set):
                LOGGER.debug(f"{path}: {len(hypotheses)} lines for {len(evaluation_set)} verses; skipped")
                continue
            outputs.add_system(f"{row.experiment}|{row.prefix}|{row.step}", hypotheses)
        return outputs

    def _source_path(self, directory: Path, prefix: str) -> Path:
        # Older experiments only kept the tokenized source; it is still usable as the source text.
        detokenized = directory / f"{prefix}.src.detok.txt"
        return detokenized if detokenized.is_file() else directory / f"{prefix}.src.txt"

    def _lines(self, path: Path) -> List[str]:
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        return re.sub(r"\n$", "", text).split("\n")
