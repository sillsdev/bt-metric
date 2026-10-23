from pathlib import Path

from btmetric.features.corpora import BibleCorpus
from btmetric.features.naturalness import CharNgramModel, PerplexityRatio

TRAINING = ["the lord is my shepherd i shall not want", "he makes me lie down in green pastures"] * 50


def test_char_model_finds_natural_text_less_perplexing_than_scrambled_text():
    model = CharNgramModel(order=4).train(TRAINING)
    assert model.log_perplexity("the lord is my shepherd") < model.log_perplexity("dhe pselord my sh i ehs")
    assert model.log_perplexity("unseen ŧext ŵith new characters") > model.log_perplexity("he makes me lie down")


def test_perplexity_feature_penalizes_distance_from_the_reference_either_way(tmp_path: Path):
    (tmp_path / "vref.txt").write_text("\n".join(f"PSA 23:{i}" for i in range(100)) + "\n", encoding="utf-8")
    (tmp_path / "en-TEST.txt").write_text("\n".join(TRAINING) + "\n", encoding="utf-8")
    resources = BibleCorpus(
        scripture_dirs=(tmp_path,), corpora_table=tmp_path / "x.tsv", vref_path=tmp_path / "vref.txt"
    )
    feature = PerplexityRatio(order=4, corpus=resources)
    feature.prepare("xx", "en-TEST")
    reference = "he makes me lie down in green pastures"
    scores = feature.segment_scores(
        ["s"] * 3, [reference, "he makes me lie in pastures green", "zq xv kj wp"], [reference] * 3
    )
    assert scores[0] == 0.0 and scores[2] < scores[1] < 0.0


def test_one_sided_perplexity_only_penalizes_being_less_predictable_than_the_reference(tmp_path: Path):
    (tmp_path / "vref.txt").write_text("\n".join(f"PSA 23:{i}" for i in range(100)) + "\n", encoding="utf-8")
    (tmp_path / "en-TEST.txt").write_text("\n".join(TRAINING) + "\n", encoding="utf-8")
    resources = BibleCorpus(
        scripture_dirs=(tmp_path,), corpora_table=tmp_path / "x.tsv", vref_path=tmp_path / "vref.txt"
    )
    feature = PerplexityRatio(order=4, corpus=resources, one_sided=True)
    feature.prepare("xx", "en-TEST")
    assert feature.name == "perplexity-one-sided"
    reference = "zq he makes me lie down xv"
    scores = feature.segment_scores(["s"] * 2, ["he makes me lie down", "zq xv kj wp qz vx"], [reference] * 2)
    assert scores[0] == 0.0 and scores[1] < 0.0
