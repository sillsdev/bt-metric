from pathlib import Path

from btmetric.features.comet import Comet, ScoreCache


class CountingWorker:
    """Stands in for the COMET worker process: scores by hypothesis length and records each request."""

    def __init__(self) -> None:
        self.requests = []

    def score(self, request):
        self.requests.append(len(request["mt"]))
        return [float(len(hypothesis)) for hypothesis in request["mt"]]

    def close(self):
        pass


def comet(tmp_path: Path, chunk_size: int = 2) -> Comet:
    feature = Comet("COMET-test", "test/model", cache_path=tmp_path / "scores.sqlite", chunk_size=chunk_size)
    feature._worker = CountingWorker()
    return feature


def test_scores_persist_so_a_restarted_run_only_scores_what_is_new(tmp_path: Path):
    sources, hypotheses, references = ["s"] * 3, ["a", "bb", "ccc"], ["r"] * 3
    first = comet(tmp_path)
    assert first.segment_scores(sources, hypotheses, references).tolist() == [1.0, 2.0, 3.0]
    assert first._worker.requests == [2, 1]
    first.close()

    restarted = comet(tmp_path)
    scores = restarted.segment_scores(sources + ["s"], hypotheses + ["dddd"], references + ["r"])
    assert scores.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert restarted._worker.requests == [1]
    restarted.close()


def test_cached_scores_are_kept_apart_by_model(tmp_path: Path):
    cache = ScoreCache(tmp_path / "scores.sqlite")
    cache.put("model-a", {("s", "h", "r"): 0.5})
    assert cache.get("model-a", [("s", "h", "r")]) == {("s", "h", "r"): 0.5}
    assert cache.get("model-b", [("s", "h", "r")]) == {}
    cache.close()
