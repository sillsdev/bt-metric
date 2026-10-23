import numpy as np
import pandas as pd

from btmetric.combination.field_tests import FieldTestPairs


def _rows(teams: int, verses: int, seed: int = 0) -> pd.DataFrame:
    """Compared drafts where the team prefers the draft with more coverage; chrF3 is noise."""
    rng = np.random.default_rng(seed)
    records = []
    for t in range(teams):
        for v in range(verses):
            coverage = rng.normal(size=2)
            label = "A" if coverage[0] > coverage[1] else "B"
            for k, system in enumerate("AB"):
                records.append(
                    {
                        "item": t * verses + v,
                        "team": f"t{t}",
                        "kind": "comparison",
                        "label": label,
                        "independent": t == 0,
                        "system": system,
                        "chrF3": rng.normal(),
                        "coverage-target": coverage[k],
                    }
                )
    return pd.DataFrame(records)


def test_weights_fit_on_other_teams_carry_over_to_a_held_out_team():
    pairs = FieldTestPairs(_rows(teams=4, verses=60), ["chrF3", "coverage-target"])
    assert len(pairs.labels) == 240 and pairs.independent.sum() == 60
    held_out = pairs.leave_one_team_out(l2=0.01)
    assert np.mean(held_out > 0) > 0.95
