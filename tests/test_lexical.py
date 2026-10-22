import numpy as np

from btmetric.features.lexical import Bleu, MeanChrF, NegativeTer
from btmetric.meta_eval.runner import FeatureRegistry


def test_lexical_features_score_identical_text_best():
    sources, references = ["s", "s"], ["the cat sat on the mat", "a dog barked"]
    for feature in (Bleu(), MeanChrF(), NegativeTer()):
        perfect = feature.system_score(sources, references, references)
        worse = feature.system_score(sources, ["the cat sat", "a cat barked"], references)
        assert perfect > worse, feature.name


def test_negative_ter_is_zero_for_a_perfect_match():
    assert NegativeTer().segment_scores(["s"], ["same words"], ["same words"]).tolist() == [0.0]


def test_mean_chrf_averages_verse_scores_rather_than_pooling_ngrams():
    feature = MeanChrF()
    hypotheses, references = ["abcdef", "x"], ["abcdef", "yyyyyyyyyyyyyyyy"]
    assert feature.system_score(["s", "s"], hypotheses, references) == np.mean(
        feature.segment_scores(["s", "s"], hypotheses, references)
    )
    assert feature.name == "m-chrF3"


def test_registry_lists_every_baseline():
    assert FeatureRegistry().default_names() == ["chrf3", "chrf3++", "m-chrf3", "bleu", "spbleu"]
    assert {"ter", "comet22", "africomet", "xcomet-xl"} <= set(FeatureRegistry().names())
