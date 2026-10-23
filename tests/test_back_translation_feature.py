from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

from btmetric.features.back_translation import BackTranslationInfo, ForcedDecodingScorer, NllbLanguages, ReverseModels
from btmetric.features.corpora import BibleCorpus
from btmetric.features.reverse_models import ScratchArchitecture, ScratchReverseModel, TrainingSettings


class FakeScorer(ForcedDecodingScorer):
    """Log P(source | text): a fixed prior for empty text, else a penalty per source word beyond the text length."""

    def __init__(self, prior: float) -> None:
        super().__init__(batch_size=8)
        self._prior = prior

    def _score_batch(self, inputs: List[str], outputs: List[str]) -> List[Tuple[float, int]]:
        results = []
        for text, source in zip(inputs, outputs):
            words = source.split()
            shared = min(len(text.split()), len(words))
            results.append((self._prior if not text else float(shared - len(words)), len(words)))
        return results


class FakeModels(ReverseModels):
    name = "fake"

    def __init__(self, prior: float) -> None:
        self._prior = prior

    def scorer(self, source_language: str, target_language: str, excluded: FrozenSet[str]) -> Optional[FakeScorer]:
        return FakeScorer(self._prior)


def _feature(prior: float) -> BackTranslationInfo:
    feature = BackTranslationInfo(FakeModels(prior))
    feature.prepare("en", "de")
    return feature


def test_omissions_score_below_the_reference():
    feature = _feature(prior=-4.0)
    source = "god so loved the world"
    scores = feature.segment_scores([source] * 2, [source, "god loved"], [source] * 2)
    assert feature.name == "bt-fake" and scores[0] == 0.0 and scores[1] < 0.0


def test_verses_the_reference_adds_nothing_to_are_left_unscored():
    # With the prior equal to the reference's likelihood, the model already "knows" the verse.
    feature = _feature(prior=0.0)
    source = "god so loved the world"
    assert np.isnan(feature.segment_scores([source], ["god loved"], [source])[0])


def test_copying_the_source_gets_no_credit():
    feature = _feature(prior=-4.0)
    source = "god so loved the world"
    reference = "gott hat die welt so geliebt"
    scores = feature.segment_scores([source] * 2, [source, reference], [reference] * 2)
    assert scores[0] < scores[1] == 0.0


def test_uncovered_languages_give_nan():
    feature = BackTranslationInfo()
    feature.prepare("en", "qaa-ABC")
    assert np.isnan(feature.segment_scores(["a"], ["b"], ["c"])).all()
    assert NllbLanguages().code("de-GCLNR00") == "deu_Latn" and NllbLanguages().code("sw") == "swh_Latn"


def test_scratch_reverse_model_trains_caches_and_scores(tmp_path: Path):
    words = ["light", "water", "bread", "house", "stone", "river", "field", "night"]
    rng = np.random.default_rng(0)
    english = [" ".join(rng.choice(words, size=5)) for _ in range(1200)]
    # A "target language" that is English reversed word by word, so the reverse model has something to learn.
    target = [" ".join(word[::-1] for word in line.split()) for line in english]
    vrefs = [f"GEN 1:{i + 1}" for i in range(len(english))]
    (tmp_path / "vref.txt").write_text("\n".join(vrefs) + "\n", encoding="utf-8")
    (tmp_path / "en-TEST.txt").write_text("\n".join(english) + "\n", encoding="utf-8")
    (tmp_path / "xx-TEST.txt").write_text("\n".join(target) + "\n", encoding="utf-8")
    corpus = BibleCorpus(scripture_dirs=(tmp_path,), corpora_table=tmp_path / "x.tsv", vref_path=tmp_path / "vref.txt")
    models = ScratchReverseModel(
        corpus,
        cache_dir=tmp_path / "models",
        # Small transformers learn the output language first and only start using their input later; this toy model
        # still ignores its input at 1,000 steps (info(ref) ~ 0, so every score is NaN) and uses it by 1,500.
        settings=TrainingSettings(steps=1500, batch_size=32, learning_rate=2e-3, warmup_steps=20),
        architecture=ScratchArchitecture(vocab_size=60, layers=1, d_model=32, heads=2, ffn_dim=64, dropout=0.0),
    )
    feature = BackTranslationInfo(models)
    feature.prepare("en-TEST", "xx-TEST", frozenset(vrefs[:10]))
    source, reference = english[0], target[0]
    scores = feature.segment_scores([source] * 2, [reference, " ".join(reference.split()[:2])], [reference] * 2)
    assert scores[0] == 0.0 and scores[1] < 0.0
    assert len(list((tmp_path / "models" / "scratch").glob("*/done"))) == 1
