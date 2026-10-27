import numpy as np

from btmetric.features.base import Feature
from btmetric.features.chrf_b import COVERAGE_TARGET_COEFFICIENT, KEY_TERMS_COEFFICIENT, ChrFB


class _Fixed(Feature):
    def __init__(self, scores, system=None) -> None:
        self._scores = np.array(scores, dtype=float)
        self._system = system
        self.prepared = None

    def prepare(self, source_language, target_language, excluded_segments=frozenset()) -> None:
        self.prepared = (source_language, target_language, excluded_segments)

    def segment_scores(self, sources, hypotheses, references):
        return self._scores

    def system_score(self, sources, hypotheses, references):
        return self._system if self._system is not None else super().system_score(sources, hypotheses, references)


def test_scores_are_chrf3_plus_coverage_and_key_terms_in_chrf3_points_with_missing_parts_counting_zero():
    coverage, key_terms = _Fixed([0.1, np.nan]), _Fixed([1.0, np.nan])
    metric = ChrFB(_Fixed([50.0, 40.0], system=45.0), coverage, key_terms)
    metric.prepare("en-X", "xyz-Y", frozenset({"MRK 1:1"}))
    assert coverage.prepared == key_terms.prepared == ("en-X", "xyz-Y", frozenset({"MRK 1:1"}))
    scores = metric.segment_scores(["s", "s"], ["h", "h"], ["r", "r"])
    assert np.allclose(scores, [50 + 0.1 * COVERAGE_TARGET_COEFFICIENT + KEY_TERMS_COEFFICIENT, 40.0])
    system = metric.system_score(["s", "s"], ["h", "h"], ["r", "r"])
    assert np.isclose(system, 45 + 0.1 * COVERAGE_TARGET_COEFFICIENT + KEY_TERMS_COEFFICIENT)


def test_normalize_rescales_chrf3_between_the_shuffled_references_floor_and_100():
    from btmetric.features.chrf import ChrF

    references = ["In the beginning God created", "the heavens and the earth.", "And the earth was without form"]
    hypotheses = ["In the beginning God made", "the heaven and earth.", "The earth was formless"]
    zeros = _Fixed([0.0, 0.0, 0.0])
    raw = ChrFB(ChrF(0), zeros, zeros)
    normalized = ChrFB(ChrF(0), zeros, zeros, normalize=True)
    assert normalized.name == "chrF-B-norm" and raw.name == "chrF-B"

    floor = normalized.floor(references, references)
    assert 0 < floor < 50
    system = raw.system_score(references, hypotheses, references)
    assert np.isclose(
        normalized.system_score(references, hypotheses, references), 100 * (system - floor) / (100 - floor)
    )
    assert np.isclose(normalized.system_score(references, references, references), 100.0)
    segments = raw.segment_scores(references, hypotheses, references)
    assert np.allclose(
        normalized.segment_scores(references, hypotheses, references), 100 * (segments - floor) / (100 - floor)
    )
