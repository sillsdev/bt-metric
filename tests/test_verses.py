from btmetric.scripture.verses import expand_vref, parallel_verses


def test_expand_vref_lists_the_verses_of_a_range():
    assert expand_vref("MAT 13:5-6") == ["MAT 13:5", "MAT 13:6"]
    assert expand_vref("MAT 13:5") == ["MAT 13:5"]


def test_a_verse_combined_with_the_one_before_joins_its_row():
    vrefs = ["GEN 1:1", "GEN 1:2", "GEN 1:3", "GEN 1:4"]
    source = ["one", "two", "<range>", "four"]
    target = ["uno", "dos", "tres", "cuatro"]
    assert parallel_verses(vrefs, source, target, set()) == [
        ("GEN 1:1", "one", "uno"),
        ("GEN 1:2-3", "two", "dos tres"),
        ("GEN 1:4", "four", "cuatro"),
    ]


def test_rows_missing_a_text_holding_ellipses_or_overlapping_excluded_verses_are_dropped():
    vrefs = ["GEN 1:1", "GEN 1:2", "GEN 1:3", "GEN 1:4", "GEN 1:5"]
    source = ["one", "two", "<range>", "four", "..."]
    target = ["uno", "dos", "tres", "", "cinco"]
    assert parallel_verses(vrefs, source, target, {"GEN 1:3"}) == [("GEN 1:1", "one", "uno")]
