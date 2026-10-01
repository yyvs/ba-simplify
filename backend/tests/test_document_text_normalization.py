"""Tests for the D-Wikipedia text conventions in document_text.py.

The expected forms here are taken from the corpus itself
(``research/scratch/d_wikipedia_raw``), which is lowercased, PTB-pre-tokenized,
and one-document-per-line.
"""

from document_text import (
    denormalize_from_model,
    ensure_terminal_punctuation,
    join_segments,
    normalize_for_model,
)

# --- ensure_terminal_punctuation / join_segments ----------------------------


def test_terminal_punctuation_added_when_missing():
    assert ensure_terminal_punctuation('History') == 'History.'


def test_terminal_punctuation_left_alone_when_present():
    for text in ('Ends with a period.', 'Ends with a question?', 'Ends with a bang!'):
        assert ensure_terminal_punctuation(text) == text


def test_terminal_punctuation_ignores_empty_text():
    assert ensure_terminal_punctuation('   ') == ''


def test_join_segments_separates_unpunctuated_segments():
    """Without the added period, "History" would run into the next segment and be
    read as one phrase -- observed: "history the igda is a non-profit organization".
    """
    assert join_segments(['History', 'The IGDA is a nonprofit']) == (
        'History. The IGDA is a nonprofit.'
    )


def test_join_segments_does_not_double_up_existing_punctuation():
    assert join_segments(['One sentence.', 'Two sentences.']) == (
        'One sentence. Two sentences.'
    )


def test_join_segments_uses_no_newlines():
    """The corpus contains no newline inside a document body, and supplying one
    measurably degraded output -- joining must stay space-only."""
    assert '\n' not in join_segments(['First part.', 'Second part.', 'Third part.'])


def test_join_segments_drops_empty_segments():
    assert join_segments(['Real text.', '   ', 'More text.']) == 'Real text. More text.'


# --- normalize_for_model ---------------------------------------------------


def test_normalize_lowercases_and_splits_punctuation():
    assert (
        normalize_for_model(
            'Achtkarspelen is a municipality in Friesland, in the northern Netherlands.'
        )
        == 'achtkarspelen is a municipality in friesland , in the northern netherlands .'
    )


def test_normalize_keeps_numbers_intact():
    """The corpus writes "28,000" and "3.5" unsplit -- splitting on every comma or
    period would turn them into separate tokens the model never saw."""
    assert normalize_for_model('It has 28,000 people and grew 3.5% since 1984.') == (
        'it has 28,000 people and grew 3.5% since 1984 .'
    )


def test_normalize_keeps_abbreviation_periods():
    """D-Wikipedia's test split contains "u.s." 531 times and "u . s ." zero times, so
    splitting these periods invents a token the corpus never has -- and the same function
    normalises document input on the serving path, so the checkpoint saw it in production too.
    """
    got = normalize_for_model('The U.S. Army in 1990.')
    assert 'u . s .' not in got
    assert 'u.s.' in got
    for text, kept in (
        ('e.g. this', 'e.g.'),
        ('at 5 a.m. sharp', 'a.m.'),
        ('U.S.A. today', 'u.s.a.'),
    ):
        assert kept in normalize_for_model(text)


def test_normalize_still_splits_a_sentence_final_period():
    """The abbreviation guard is shape-based, so it must not swallow ordinary periods."""
    assert normalize_for_model('This ends.').endswith(' .')


def test_normalize_splits_possessives():
    assert normalize_for_model("women's oppression") == "women 's oppression"


def test_normalize_uses_ptb_quotes():
    assert normalize_for_model('She read "The Second Sex" today.') == (
        "she read `` the second sex '' today ."
    )


def test_normalize_alternates_unbalanced_quotes_without_failing():
    assert (
        normalize_for_model('a "quote that never closes')
        == 'a `` quote that never closes'
    )


def test_normalize_splits_parentheses():
    assert normalize_for_model('The IGDA (a nonprofit) has members.') == (
        'the igda ( a nonprofit ) has members .'
    )


# --- denormalize_from_model ------------------------------------------------


def test_denormalize_reattaches_punctuation_and_capitalizes_sentences():
    assert denormalize_from_model('it has 28,000 people . they live there .') == (
        'It has 28,000 people. They live there.'
    )


def test_denormalize_restores_proper_nouns_from_the_source():
    """The model only ever emits lowercase, so casing for proper nouns has to come
    from how they appeared in the input."""
    source = 'Achtkarspelen is a municipality in Friesland in the northern Netherlands.'
    assert (
        denormalize_from_model('achtkarspelen is a municipality in friesland .', source)
        == 'Achtkarspelen is a municipality in Friesland.'
    )


def test_denormalize_restores_acronyms():
    source = 'The IGDA is incorporated in the United States.'
    assert denormalize_from_model('the igda is a non-profit .', source) == (
        'The IGDA is a non-profit.'
    )


def test_denormalize_does_not_capitalize_mid_sentence_from_sentence_starts():
    """ "The" leading the source sentence must not make every later "the" capitalized."""
    source = 'The organization is large. The members are many.'
    assert denormalize_from_model(
        'groups exist and the members are many .', source
    ) == ('Groups exist and the members are many.')


def test_denormalize_converts_ptb_quotes_back():
    assert denormalize_from_model("she wrote `` the second sex '' .") == (
        'She wrote "the second sex".'
    )


def test_denormalize_restores_possessives():
    assert denormalize_from_model("about women 's lives .") == "About women's lives."


def test_denormalize_drops_empty_parens():
    """ "( )" is a corpus artifact from stripped pronunciation blocks -- it carries no
    text, so it shouldn't be rendered."""
    assert denormalize_from_model('achtkarspelen ( ) is a municipality .') == (
        'Achtkarspelen is a municipality.'
    )


def test_denormalize_restores_the_space_the_model_drops_after_a_sentence():
    """The decoder sometimes emits the token after a punctuation mark without its
    leading space marker, so "boundaries . the" arrives as "boundaries .the" -- and
    the punctuation re-attachment then closes it into one sentence. Two, here."""
    source = (
        'The movement referred to its national leader. The movement argued that '
        'attempts to rewrite the last election would not strengthen democracy.'
    )
    assert denormalize_from_model('boundaries .the movement argued', source) == (
        'Boundaries. The movement argued'
    )


def test_denormalize_restores_the_space_after_a_question_mark_and_a_comma():
    assert (
        denormalize_from_model('why ?the answer is simple')
        == 'Why? The answer is simple'
    )
    assert denormalize_from_model('many words ,more words follow') == (
        'Many words, more words follow'
    )


def test_denormalize_inserts_no_space_inside_an_abbreviation():
    """Same guard as the normalizer's: a period whose preceding letter isn't itself
    preceded by a letter ends an abbreviation, not a sentence, and gets no space.

    Asserted on the spacing alone. These strings keep the odd capitals
    _restore_sentence_capitals has always given them ("u.S.Congress"), which is a
    separate matter from whether a space was inserted."""
    assert denormalize_from_model('the u.s.congress met') == 'The u.S.Congress met'
    assert denormalize_from_model('e.g.this one') == 'E.G.This one'


def test_denormalize_inserts_no_space_inside_a_number():
    """The letter lookahead is what protects these -- a digit after the mark is never
    the start of a sentence, so "3.5" and "28,000" keep their own separators."""
    result = denormalize_from_model('it rose 3.5 percent among 28,000 people')
    assert '3.5' in result and '28,000' in result
    assert '3. 5' not in result and '28, 000' not in result


def test_normalize_denormalize_round_trip_preserves_readable_text():
    original = 'She read "The Second Sex" in Friesland, and women\'s lives changed.'
    assert denormalize_from_model(normalize_for_model(original), original) == original
