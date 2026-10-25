from btmetric.scripture.original_languages import combine, plain


def test_plain_drops_marks_and_splits_hebrew_prefixes():
    assert plain("בְּ⁠רֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים") == "ב ראשית ברא אלהים"
    assert plain("βίβλος γενέσεως Ἰησοῦ") == "βιβλος γενεσεως Ιησου"
    assert plain("δήποτʼ οὖν") == "δηποτ’ ουν"
    assert plain("τί ποιεῖς;") == plain("τί ποιεῖς?") == "τι ποιεις?"
    assert plain("אֲשֶׁ֣ר ׀ לֹ֣א") == "אשר לא"


def test_combine_takes_each_verse_from_the_testament_that_has_it():
    assert combine(["בָּרָ֣א", "", ""], ["", "λόγος", ""]) == ["ברא", "λογος", ""]
