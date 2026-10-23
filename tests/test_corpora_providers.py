from pathlib import Path

import pandas as pd

from btmetric.combination.human import SegmentScoreTable, in_domain_pool
from btmetric.evaluation.data import SegmentJudgments
from btmetric.features.corpora import NewsCorpus
from btmetric.meta_eval.runner import FeatureRegistry


class OfflineNewsCorpus(NewsCorpus):
    def __init__(self, pool, cache_dir, opus=None):
        super().__init__(pool, cache_dir=cache_dir, min_pairs=2)
        self._fake_opus = opus or {}

    def _download(self, first, second):
        return self._fake_opus.get((first, second))


def pool():
    return {("en", "gu"): [(f"source {i}", f"reference {i}") for i in range(5)]}


def test_news_corpus_falls_back_to_the_pool_without_the_excluded_sources(tmp_path: Path):
    corpus = OfflineNewsCorpus(pool(), tmp_path)
    text = corpus.parallel("en", "gu", frozenset({"source 0", "source 1"}))
    assert text.source == ["source 2", "source 3", "source 4"] and text.key.startswith("pool:en-gu:")
    assert corpus.parallel("en", "gu", frozenset({"source 2"})).key != text.key
    assert corpus.monolingual("gu", frozenset({"source 4"})).lines == [f"reference {i}" for i in range(4)]


def test_news_corpus_uses_opus_where_available_and_drops_test_sentences(tmp_path: Path):
    lines = [("Guten Tag", "Good day"), ("source 0", "leaked"), ("Danke", "Thanks")]
    corpus = OfflineNewsCorpus({("de", "en"): [("source 0", "reference 0")]}, tmp_path, {("de", "en"): lines})
    text = corpus.parallel("de", "en", frozenset())
    assert sorted(text.source) == ["Danke", "Guten Tag"] and text.key == "opus:de-en"
    assert corpus.parallel("de", "en", frozenset({"Danke"})).key == "opus:de-en"
    assert sorted(corpus.monolingual("en", frozenset()).lines) == ["Good day", "Thanks"]


def test_scoring_in_folds_never_trains_a_segments_models_on_its_own_source(tmp_path: Path):
    rows = pd.DataFrame(
        {"item": range(6), "source": [f"s{i}" for i in range(6)], "hypothesis": "h", "reference": "r", "human": 1.0}
    )
    judged = [SegmentJudgments("wmt19-da-en-gu", rows, "en", "gu")]
    assert in_domain_pool(judged)[("en", "gu")][0] == ("s0", "r")

    seen = []

    class Recording:
        name = "recording"

        def prepare(self, source_language, target_language, excluded=frozenset()):
            self.excluded = excluded

        def segment_scores(self, sources, hypotheses, references):
            seen.append((set(sources), set(self.excluded)))
            return [0.0] * len(sources)

        def close(self):
            pass

    class Registry(FeatureRegistry):
        def get(self, name):
            return Recording()

    SegmentScoreTable(tmp_path / "scores.tsv", folds=3).build(judged, ["recording"], Registry())
    assert all(scored == excluded for scored, excluded in seen)
    assert set().union(*(scored for scored, _ in seen)) == {f"s{i}" for i in range(6)}
