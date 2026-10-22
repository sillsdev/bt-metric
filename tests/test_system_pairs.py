import numpy as np
import pandas as pd

from btmetric.analysis.system_pairs import OutputScores, SystemPairBaselines, summarize
from btmetric.evaluation.data import EvaluationSet, JudgedOutputs
from btmetric.features.chrf import ChrF
from btmetric.features.lexical import Bleu


class FakeLoader:
    def __init__(self, outputs: JudgedOutputs) -> None:
        self._outputs = outputs

    def outputs_for(self, test_hash: str) -> JudgedOutputs:
        return self._outputs


def outputs() -> JudgedOutputs:
    references = [f"the reference verse number {i} about the kingdom" for i in range(30)]
    evaluation_set = EvaluationSet("set", "en", "xx", ["s"] * 30, references, [str(i) for i in range(30)])
    judged = JudgedOutputs(evaluation_set)
    judged.add_system("good", references)
    judged.add_system("bad", ["something else entirely" for _ in range(30)])
    return judged


def test_output_scores_hold_segment_and_system_scores_per_feature():
    scores = OutputScores(outputs(), [ChrF()])
    assert scores.system["chrF3"]["good"] == 100.0
    assert len(scores.segment["chrF3"]["bad"]) == 30


def test_pairs_get_deltas_per_feature_and_a_significance_test():
    pairs = pd.DataFrame(
        [{"test_hash": "h", "verses": 30, "test_design": "books", "system_a": "good", "system_b": "bad"}]
    )
    table = SystemPairBaselines(FakeLoader(outputs()), [ChrF(), Bleu()], "chrF3").score(pairs, workers=1)
    assert table.loc[0, "delta_chrF3"] > 0 and table.loc[0, "delta_BLEU"] > 0
    assert table.loc[0, "p_value"] < 0.001
    assert "1.0" in summarize(table, ["chrF3", "BLEU"])


def test_summary_reports_how_often_baselines_pick_the_same_winner():
    table = pd.DataFrame(
        {
            "delta_A": [1.0, 1.0, -1.0, 2.0],
            "delta_B": [1.0, -1.0, -1.0, 1.0],
            "p_value": [0.01, 0.2, 0.01, 0.5],
            "test_design": ["books"] * 4,
            "verses": [250] * 4,
        }
    )
    text = summarize(table, ["A", "B"])
    assert "0.75" in text
    assert np.isclose(np.mean(np.sign(table["delta_A"]) == np.sign(table["delta_B"])), 0.75)
