from pathlib import Path

import numpy as np
import pandas as pd

from btmetric.evaluation.data import EvaluationSet, JudgedOutputs
from btmetric.meta_eval.runner import FeatureRegistry
from btmetric.perturbations.challenge_set import ChallengeSetBuilder, ProjectResources, evaluate
from btmetric.perturbations.perturbations import (
    Addition,
    ClauseOmission,
    KeyTermOccurrence,
    KeyTermSubstitution,
    LocalShuffle,
    NegationRemoval,
    Repetition,
    SourceCopy,
    Truncation,
    VerseContext,
    WrongVerse,
    all_perturbations,
)
from btmetric.terms.inference import AlignedVerses
from btmetric.terms.major_terms import MajorTermList, ProjectKeyTerms, find_renderings_file
from btmetric.terms.negation import NegationMarkers

VERSE = "Jesus went up the mountain, and he sat down; his disciples came to him."


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_omission_drops_a_whole_clause():
    result = ClauseOmission().apply(VERSE, VerseContext(), rng())
    clauses = ["Jesus went up the mountain,", "and he sat down;", "his disciples came to him."]
    assert sum(clause in result for clause in clauses) == 2


def test_omission_works_on_scripts_without_spaces_or_punctuation():
    result = ClauseOmission().apply("天地万物都是神创造的", VerseContext(), rng())
    assert result is not None and len(result) < len("天地万物都是神创造的")


def test_apostrophes_inside_words_do_not_split_clauses():
    assert ClauseOmission().apply("Ya'akov's people", VerseContext(), rng()) in ("Ya'akov's", "people")


def test_addition_inserts_a_clause_from_another_verse():
    context = VerseContext(other_verses=("Peter wept bitterly.",))
    result = Addition().apply(VERSE, context, rng())
    assert "Peter wept bitterly." in result and "Jesus went up the mountain," in result


def test_key_term_substitution_swaps_in_another_term_of_the_same_kind():
    context = VerseContext(key_terms=(KeyTermOccurrence("Jesus", ("Moses",)),))
    assert KeyTermSubstitution().apply(VERSE, context, rng()) == VERSE.replace("Jesus", "Moses")
    assert KeyTermSubstitution().apply(VERSE, VerseContext(), rng()) is None


def test_other_perturbations_change_the_text_as_described():
    context = VerseContext(source="Iesous anebe eis to oros", other_verses=("Peter wept bitterly.",))
    assert sorted(LocalShuffle().apply(VERSE, context, rng()).split()) == sorted(VERSE.split())
    assert WrongVerse().apply(VERSE, context, rng()) == "Peter wept bitterly."
    assert SourceCopy().apply(VERSE, context, rng()) == "Iesous anebe eis to oros"
    assert len(Repetition().apply(VERSE, context, rng()).split()) > len(VERSE.split())
    assert Truncation(2).apply(VERSE, context, rng()) == "Jesus went up the mountain, and he"
    assert Truncation().apply(VERSE, context, rng()) == "Jesus went up the mountain, and he sat down; his"


def test_perturbations_that_need_missing_context_do_not_apply():
    for perturbation in (Addition(), WrongVerse(), SourceCopy(), KeyTermSubstitution()):
        assert perturbation.apply(VERSE, VerseContext(), rng()) is None


def write_terms(tmp_path: Path) -> MajorTermList:
    (tmp_path / "Major-metadata.txt").write_text(
        "Ἰησοῦς\tPN\tperson\nΜωϋσῆς\tPN\tperson\nὄρος\tGEOG\tmountain\n", encoding="utf-8"
    )
    (tmp_path / "Major-vrefs.txt").write_text("MAT 5:1\tMAT 5:2\nEXO 3:1\nMAT 5:1\n", encoding="utf-8")
    return MajorTermList.load(tmp_path)


def test_project_key_terms_find_renderings_in_the_verse_with_same_kind_alternatives(tmp_path: Path):
    terms = write_terms(tmp_path)
    (tmp_path / "en-X-Major-renderings.txt").write_text("Jesus\nMoses*\nmountain\n", encoding="utf-8")
    project = ProjectKeyTerms.load(terms, tmp_path / "en-X-Major-renderings.txt")
    found = {
        occurrence.rendering: occurrence.alternatives for occurrence in project.occurrences("MAT 5:1", VERSE, rng())
    }
    assert found == {"Jesus": ("Moses",), "mountain": ()}


def test_renderings_files_are_found_for_dated_copies_of_the_same_project_only(tmp_path: Path):
    for name in ("qaa-ABC_2023_11_30", "qaa-ABC_2024_04_22", "qaa-ABC_BTE_2024_04_22"):
        (tmp_path / f"{name}-Major-renderings.txt").write_text("", encoding="utf-8")
    assert find_renderings_file(tmp_path, "qaa-ABC_2023_11_30").name == "qaa-ABC_2023_11_30-Major-renderings.txt"
    assert find_renderings_file(tmp_path, "qaa-ABC_2025_01_01").name == "qaa-ABC_2024_04_22-Major-renderings.txt"
    assert find_renderings_file(tmp_path, "abc-ABC") is None


class StubResources:
    def for_target(self, target, test_vrefs):
        return ProjectResources(None, "none", ("not",))


def test_negation_removal_deletes_a_negation_word_and_keeps_its_punctuation():
    context = VerseContext(negation_markers=("not",))
    assert NegationRemoval().apply("Do not be afraid, not yet.", context, rng()) in (
        "Do be afraid, not yet.",
        "Do not be afraid, yet.",
    )
    assert NegationRemoval().apply("They were not.", context, rng()) == "They were."
    assert NegationRemoval().apply(VERSE, context, rng()) is None


def test_negation_markers_are_the_words_aligned_to_english_negators():
    english = [["do", "not", "fear"], ["he", "did", "not", "go"], ["he", "went"]] * 10
    target = [["usi", "ogope"], ["hakwenda", "si"], ["alienda"]] * 10
    verses = AlignedVerses([str(i) for i in range(30)], english, target)
    verses.alignments = [[(1, 0), (2, 1)], [(2, 1), (3, 0)], [(1, 0)]] * 10
    assert NegationMarkers().infer(verses) == ["usi", "si"]


def test_inferred_lowercase_renderings_are_found_in_capitalized_text(tmp_path: Path):
    project = ProjectKeyTerms(write_terms(tmp_path), [["jesus"], ["moses"], []])
    [occurrence] = project.occurrences("MAT 5:1", VERSE, rng())
    assert occurrence.rendering == "Jesus" and occurrence.alternatives == ("moses",)


def test_challenge_set_rows_pair_an_output_with_its_perturbed_copy_and_evaluate(tmp_path: Path):
    references = [f"reference verse {i} about the kingdom of heaven, which is not near." for i in range(5)]
    outputs = JudgedOutputs(
        EvaluationSet("h", "en", "xx", [f"src {i}" for i in range(5)], references, [f"MAT 5:{i}" for i in range(5)])
    )
    outputs.add_system("a", references)
    builder = ChallengeSetBuilder(all_perturbations(), StubResources(), verses_per_set=5, seed=0)
    rows = pd.DataFrame([vars(row) for row in builder.rows_for(outputs, "xx-X")])
    assert set(rows["perturbation"]) >= {"omission", "addition", "word_order", "wrong_verse", "source_copy"}
    assert "negation" in set(rows["perturbation"])
    assert (rows["original"] != rows["perturbed"]).all()
    results = evaluate(rows, ["chrf3"], FeatureRegistry())
    assert (results["accuracy"] == 1.0).all()


def test_error_signatures_flag_length_repetition_and_source_copy():
    from btmetric.perturbations.calibration import ErrorSignatures

    signatures = ErrorSignatures()
    reference = "and the word was with god and the word was god"
    assert signatures.detect("src", "and the word", reference)["too_short"]
    assert signatures.detect("src", reference + " " + reference, reference)["too_long"]
    looping = "he said he said he said he said to them"
    assert signatures.detect("src", looping, "he said to them")["repetition"]
    source = "en arche en ho logos kai ho logos en pros ton theon"
    assert signatures.detect(source, source, reference)["source_copy"]
    assert not any(signatures.detect(source, reference, reference).values())


def test_loops_are_back_to_back_repeats_not_parallelism_or_reduplication():
    from btmetric.perturbations.calibration import ErrorSignatures

    signatures = ErrorSignatures()
    assert signatures.detect("s", "they came rmai pia rmai pia to him", "they came to him")["loop"]
    assert not signatures.detect("s", "you and your sons, and you and your sons", "you and yours")["loop"]
    assert not signatures.detect("s", "he walked slowly slowly home", "he walked home")["loop"]
    assert signatures.detect("s", "he said said said it", "he said it")["loop"]


def test_calibrated_weights_follow_real_error_rates():
    from btmetric.perturbations.challenge_set import CalibratedWeights

    rates = pd.DataFrame(
        {
            "test_hash": ["a", "b"],
            "target": "x",
            "system": "s",
            "too_short": [0.02, 0.02],
            "too_long": [0.04, 0.04],
            "loop": [0.02, 0.02],
            "source_copy": [0.0, 0.0],
            "empty": [0.0, 0.0],
            "repetition": [0.1, 0.1],
        }
    )
    table = pd.DataFrame(
        {"chrF3": [0.5, 1.0, 1.0, 0.7, 1.0, 0.9]},
        index=["addition", "omission", "truncation", "repetition", "source_copy", "key_term"],
    )
    weighted = CalibratedWeights.from_error_rates(rates).apply(table)
    assert abs(weighted["chrF3"] - (0.04 * 0.5 + 0.01 + 0.01 + 0.02 * 0.7) / 0.08) < 1e-3


def test_reference_mode_perturbs_the_reference():
    references = [f"reference verse {i} about the kingdom of heaven, which is not near." for i in range(3)]
    outputs = JudgedOutputs(
        EvaluationSet("h", "en", "xx", [f"src {i}" for i in range(3)], references, [f"MAT 5:{i}" for i in range(3)])
    )
    outputs.add_system("a", ["mt output one two three"] * 3)
    builder = ChallengeSetBuilder(all_perturbations(), StubResources(), 3, 0, perturb_reference=True)
    rows = builder.rows_for(outputs, "xx-X")
    assert all(row.original == row.reference for row in rows)


def test_severe_perturbations_inject_more_of_the_error():
    context = VerseContext(other_verses=("Peter wept bitterly.", "The rooster crowed."))
    assert len(ClauseOmission(2).apply(VERSE, context, rng())) < len(ClauseOmission(1).apply(VERSE, context, rng()))
    assert len(Addition(2).apply(VERSE, context, rng())) > len(Addition(1).apply(VERSE, context, rng()))
    assert len(Repetition(2).apply(VERSE, context, rng()).split()) > len(
        Repetition(1).apply(VERSE, context, rng()).split()
    )
    terms = VerseContext(key_terms=(KeyTermOccurrence("Jesus", ("Moses",)), KeyTermOccurrence("mountain", ("sea",))))
    assert KeyTermSubstitution(2).apply(VERSE, terms, rng()) == VERSE.replace("Jesus", "Moses").replace(
        "mountain", "sea"
    )
    assert KeyTermSubstitution(2).apply(VERSE, VerseContext(key_terms=terms.key_terms[:1]), rng()) is None


def test_every_graded_perturbation_comes_at_both_severities():
    pairs = [(p.name, p.severity) for p in all_perturbations()]
    for name in ("omission", "addition", "key_term", "word_order", "repetition", "truncation"):
        assert (name, 1) in pairs and (name, 2) in pairs


def test_evaluation_reports_whether_mild_errors_outscore_severe_ones():
    rows = pd.DataFrame(
        {
            "test_hash": "h",
            "vref": ["v1", "v1"],
            "perturbation": "omission",
            "severity": [1, 2],
            "source": "s",
            "reference": "a b c d e f g h",
            "original": "a b c d e f g h",
            "perturbed": ["a b c d e f", "a b c"],
        }
    )
    [result] = evaluate(rows, ["chrf3"], FeatureRegistry()).to_dict("records")
    assert result["graded"] == 1.0 and result["accuracy_mild"] == 1.0 and result["accuracy_severe"] == 1.0


def test_natural_error_pairs_match_a_faulty_output_with_a_clean_one():
    from btmetric.perturbations.calibration import ErrorSignatures
    from btmetric.perturbations.natural_errors import NaturalErrorPairs

    reference = "and he said to them, follow me and I will make you fishers of men"
    evaluation_set = EvaluationSet("h", "en", "xx", ["src"], [reference], ["MAT 4:19"])
    outputs = JudgedOutputs(evaluation_set)
    outputs.add_system("clean", ["he told them, follow me and I will make you fishers of people"])
    outputs.add_system("loop", ["he said to them follow me follow me follow me and fishers"])
    outputs.add_system("short", ["follow me"])
    pairs = NaturalErrorPairs(ErrorSignatures(), ["loop", "too_short"], max_pairs_per_type=5, seed=0)
    rows = {row.perturbation: row for row in pairs.rows_for(outputs, "xx-X")}
    assert set(rows) == {"natural_loop", "natural_too_short"}
    assert all(row.original.startswith("he told them") for row in rows.values())
    assert rows["natural_too_short"].perturbed == "follow me"


def test_evaluation_scores_each_test_set_after_preparing_its_language_pair():
    rows = pd.DataFrame(
        {
            "test_hash": ["a", "b"],
            "vref": ["v1", "v1"],
            "perturbation": "omission",
            "severity": 1,
            "source": "s",
            "reference": "a b c d e f g h",
            "original": "a b c d e f g h",
            "perturbed": ["a b c", "a b c d"],
        }
    )
    results = evaluate(rows, ["chrf3"], FeatureRegistry(), {"a": ("en-X", "xx-Y"), "b": ("en-X", "xx-Z")})
    assert results["accuracy"].tolist() == [1.0] and results["rows"].tolist() == [2]


def test_rows_a_feature_cannot_score_are_left_out_of_its_accuracy():
    rows = pd.DataFrame(
        {
            "test_hash": "a",
            "vref": ["v1", "v2"],
            "perturbation": "key_term",
            "severity": 1,
            "source": "s",
            "reference": ["Jesus said", "the people said"],
            "original": ["Jesus said", "the people said"],
            "perturbed": ["Moses said", "the folk said"],
        }
    )

    class OnlyFirstRow:
        name = "first-only"

        def segment_scores(self, sources, hypotheses, references):
            return np.array([1.0 if hypotheses[0].startswith("Jesus") else 0.0, np.nan])

        def close(self):
            pass

    class Registry:
        def get(self, name):
            return OnlyFirstRow()

    [result] = evaluate(rows, ["first-only"], Registry()).to_dict("records")
    assert result["accuracy"] == 1.0 and result["scored_rows"] == 1 and result["rows"] == 2


def test_row_scores_accumulate_features_for_the_same_challenge_set(tmp_path: Path):
    from btmetric.perturbations.challenge_set import RowScores

    challenge = pd.DataFrame({"test_hash": "a", "vref": ["v1", "v2"], "perturbation": "omission", "severity": 1})
    path = tmp_path / "rows.tsv"
    RowScores(path).merge(challenge, {"chrF3|original": np.array([1.0, 2.0])})
    table = RowScores(path).merge(challenge, {"BLEU|original": np.array([3.0, np.nan])})
    assert table["chrF3|original"].tolist() == [1.0, 2.0] and table["BLEU|original"].tolist()[0] == 3.0
    other = pd.DataFrame({"test_hash": "b", "vref": ["v9"], "perturbation": "omission", "severity": 1})
    assert list(RowScores(path).merge(other, {"BLEU|original": np.array([1.0])}).columns)[-1] == "BLEU|original"
