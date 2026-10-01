"""The guard that decides whether a model's output is a simplification at all.

The cases below are the ones that motivated it -- output taken from real pages
where the extension highlighted a sentence as simplified and the only difference
was a lost full stop or a lost number -- plus the edits that must keep passing.
That second list is the more important one: a single dropped word can already be a
real simplification, so only punctuation-level edits and lost digits are rejected,
and anything that changes a word is left to the model.
"""

import change_guard
import pytest
from change_guard import NO_MEANINGFUL_CHANGE, check_change, flesch_kincaid_grade


@pytest.mark.parametrize(
    'original, simplified, label',
    [
        (
            ': heading-delimited sections for the document model.',
            ': heading-delimited sections for the document model',
            'a dropped full stop',
        ),
        (
            'The dog, which was old, sat down slowly.',
            'The dog which was old sat down slowly.',
            'dropped commas',
        ),
        (
            'Rufai Oseni disputes the NBS 15.43 percent inflation figure again',
            'Rufai Oseni disputes the NBS percent inflation figure again',
            'a dropped number, every word still in place -- a lost fact, not a simpler sentence',
        ),
        (
            '‘Osun election has proven every evil has an expiry date’',
            'Osun election has proven every evil has an expiry date',
            'dropped quotation marks',
        ),
    ],
)
def test_cosmetic_edits_are_rejected(original, simplified, label):
    assert check_change(original, simplified) == NO_MEANINGFUL_CHANGE, label


@pytest.mark.parametrize(
    'original, simplified, label',
    [
        (
            'Although many factors obfuscate causality, the regulations mandate uniform compliance.',
            'Even though many things make cause and effect unclear, the rules require the same standards.',
            'a rephrasing',
        ),
        (
            'The committee ultimately decided to postpone the vote.',
            'The committee decided to postpone the vote.',
            'a single dropped word -- already a simplification, and not ours to second-guess',
        ),
        (
            'The committee, after extensive deliberation over several months, ultimately decided '
            'to postpone the vote.',
            'The committee decided to postpone the vote.',
            'a cut big enough to be an editorial decision',
        ),
        (
            'The regulations mandate uniform compliance across all 27 member states.',
            'The rules say all 27 member states must follow the same standards.',
            'a rephrasing that keeps the number',
        ),
        (
            'Achtkarspelen is a municipality in Friesland, in the northern Netherlands.',
            'Achtkarspelen is a municipality in Friesland.',
            'a document model dropping a clause',
        ),
        (
            'He came he saw he conquered the whole region without any resistance',
            'He came. He saw. He conquered the whole region without any resistance.',
            'a run-on split into sentences, where punctuation is the whole edit',
        ),
        (
            'A sentence the model left exactly as it was.',
            'A sentence the model left exactly as it was.',
            'output identical to the input -- nothing was rejected, so no reason',
        ),
    ],
)
def test_real_edits_pass(original, simplified, label):
    assert check_change(original, simplified) is None, label


def test_apply_serves_the_original_when_the_edit_is_rejected():
    original = 'Sowore blasts Nigerians celebrating IBB at 85 years.'
    text, reason = change_guard.apply(
        original, 'Sowore blasts Nigerians celebrating IBB at 85 years'
    )
    assert text == original
    assert reason == NO_MEANINGFUL_CHANGE


def test_apply_passes_a_real_simplification_through():
    text, reason = change_guard.apply(
        'The regulations mandate uniform compliance.',
        'The rules say everyone must follow the same standards.',
    )
    assert text == 'The rules say everyone must follow the same standards.'
    assert reason is None


def test_grade_level_falls_when_text_gets_simpler():
    hard = 'The utilisation of sophisticated terminology invariably obfuscates comprehension.'
    easy = 'Big words make it hard to understand.'
    assert flesch_kincaid_grade(hard) > flesch_kincaid_grade(easy)


def test_grade_level_of_empty_text_is_defined():
    assert flesch_kincaid_grade('') == 0.0
