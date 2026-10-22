"""Where metrics rank the human translations that WMT MQM judged alongside MT systems ("refB", "Human-A", ...).

MQM raters usually put them at or near the top; a metric that ranks them low is rewarding similarity to the
reference's wording rather than translation quality.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from ..evaluation.loaders import WmtMqmLoader
from ..meta_eval.runner import FeatureRegistry

LOGGER = logging.getLogger(__name__)


def main() -> None:
    registry = FeatureRegistry()
    parser = argparse.ArgumentParser(description="Rank WMT human translations with metrics and with MQM")
    parser.add_argument("--mqm", type=Path, default=Path("data/external/wmt-mqm/wmt-mqm-human-evaluation.csv"))
    parser.add_argument("--human-systems", default=r"^(ref|Human)")
    parser.add_argument("--features", nargs="+", default=registry.default_names(), choices=registry.names())
    parser.add_argument("--output", type=Path, default=Path("data/results/wmt_human_references.tsv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    human = re.compile(args.human_systems)
    outputs_list = [o for o in WmtMqmLoader(args.mqm).judged_outputs() if any(human.search(s) for s in o.systems())]
    rows = []
    for name in args.features:
        feature = registry.get(name)
        try:
            for outputs in outputs_list:
                evaluation_set = outputs.evaluation_set
                scores = pd.DataFrame(
                    {
                        "system": outputs.systems(),
                        "mqm": [outputs.human_system_score(system) for system in outputs.systems()],
                        "metric": [
                            feature.system_score(
                                evaluation_set.sources, outputs.hypotheses[s], evaluation_set.references
                            )
                            for s in outputs.systems()
                        ],
                    }
                )
                scores["mqm_rank"] = scores["mqm"].rank(ascending=False, method="min").astype(int)
                scores["metric_rank"] = scores["metric"].rank(ascending=False, method="min").astype(int)
                for row in scores[scores["system"].str.contains(human)].itertuples():
                    hypotheses = outputs.hypotheses[row.system]
                    same = sum(h.strip() == r.strip() for h, r in zip(hypotheses, evaluation_set.references))
                    rows.append(
                        {
                            "dataset": evaluation_set.name,
                            "feature": feature.name,
                            "human_system": row.system,
                            # The human translation used as the reference trivially gets the best metric score.
                            "is_reference": same > len(hypotheses) / 2,
                            "mqm_rank": row.mqm_rank,
                            "metric_rank": row.metric_rank,
                            "systems": len(scores),
                        }
                    )
        finally:
            feature.close()
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    LOGGER.info(f"Ranks of human translations (1 = best):\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
