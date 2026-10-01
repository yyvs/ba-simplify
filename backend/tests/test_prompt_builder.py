"""Prompt construction and output sanitising, checked as pure functions.

Imports `prompting` and nothing else from the backend: the module has no model or
HTTP dependency, and keeping torch and transformers out of this file is what makes
it run in milliseconds. The prompted path's behaviour *through* the API -- including
document granularity -- is covered in test_ollama_backend.py instead.

The BLESS reproduction claim is anchored here: if INSTRUCTION_TEMPLATE is edited,
test_default_audience_reproduces_bless_prompt_2_verbatim fails rather than the thesis
claim quietly becoming false.
"""

import re

import prompting
import pytest
from prompting import Audience, Example

# The instruction text exactly as printed in BLESS (Kew et al., EMNLP 2023),
# Figure 1c "Prompt 2", reflowed into a single line. Reproduction anchor: editing
# INSTRUCTION_TEMPLATE fails this test, so the thesis claim "we use BLESS Prompt 2"
# can't silently become false.
BLESS_PROMPT_2_INSTRUCTION = (
    'Please rewrite the following complex sentence in order to make it easier to '
    'understand by non-native speakers of English. You can do so by replacing '
    'complex words with simpler synonyms (i.e. paraphrasing), deleting unimportant '
    'information (i.e. compression), and/or splitting a long complex sentence into '
    'several simpler ones. The final simplified sentence needs to be grammatical, '
    'fluent, and retain the main ideas of its original counterpart without altering '
    'its meaning.'
)


def test_default_audience_reproduces_bless_prompt_2_verbatim():
    assert prompting.render_instruction() == BLESS_PROMPT_2_INSTRUCTION


def test_default_audience_is_the_one_bless_used():
    assert prompting.DEFAULT_AUDIENCE is Audience.NON_NATIVE_SPEAKERS


def test_audience_substitution_only_changes_the_audience_phrase():
    children = prompting.render_instruction(Audience.CHILDREN)
    assert 'children aged 8 to 12' in children
    assert 'non-native speakers' not in children
    # everything after the audience clause is untouched
    tail = 'You can do so by replacing complex words'
    assert children.split(tail)[1] == BLESS_PROMPT_2_INSTRUCTION.split(tail)[1]


@pytest.mark.parametrize('audience', list(Audience))
def test_every_audience_is_fully_configured(audience):
    """A new enum member must not silently ship without a description, a UI
    label, or demonstrations -- each omission is a distinct runtime failure
    (KeyError, blank dropdown entry, accidental zero-shot)."""
    assert audience in prompting.AUDIENCE_DESCRIPTIONS
    assert audience in prompting.AUDIENCE_LABELS
    assert len(prompting.FEWSHOT_EXAMPLES[audience]) == prompting.FEWSHOT_N


@pytest.mark.parametrize('audience', list(Audience))
def test_audience_description_reads_grammatically_in_the_slot(audience):
    """The description is spliced in after "understand by ", so it has to be a
    bare noun phrase -- no leading article-less verb, no trailing period."""
    description = prompting.audience_description(audience)
    assert description == description.strip()
    assert not description.endswith('.')


def test_audience_is_a_plain_string_enum():
    # main.SimplifyRequest uses this directly as a Pydantic field type, and it is
    # serialised straight into the /simplify and /health JSON.
    assert Audience.CHILDREN == 'children'
    assert Audience('children') is Audience.CHILDREN


def test_build_prompt_has_instruction_examples_and_dangling_prefix():
    prompt = prompting.build_prompt('A very complex sentence indeed.')

    assert prompt.startswith(BLESS_PROMPT_2_INSTRUCTION)
    # N demonstrations plus the input itself
    assert prompt.count('Complex: ') == prompting.FEWSHOT_N + 1
    assert prompt.count('Simple: ') == prompting.FEWSHOT_N
    # the model continues from a bare, empty "Simple:"
    assert prompt.endswith('Complex: A very complex sentence indeed.\nSimple:')


def test_build_prompt_uses_the_pool_for_the_requested_audience():
    prompt = prompting.build_prompt('Input.', Audience.CHILDREN)
    first_child_example = prompting.FEWSHOT_EXAMPLES[Audience.CHILDREN][0]
    assert first_child_example.simple in prompt

    non_native_only = prompting.FEWSHOT_EXAMPLES[Audience.NON_NATIVE_SPEAKERS][0].simple
    assert non_native_only not in prompt


def test_build_prompt_supports_zero_shot_fallback():
    prompt = prompting.build_prompt('Input.', Audience.CHILDREN, examples=[])
    assert (
        prompt
        == prompting.render_instruction(Audience.CHILDREN)
        + '\n\nComplex: Input.\nSimple:'
    )


def test_build_prompt_accepts_explicit_examples():
    examples = [Example(complex='Big word.', simple='Small word.')]
    prompt = prompting.build_prompt('Input.', examples=examples)
    assert 'Complex: Big word.\nSimple: Small word.' in prompt
    assert prompt.count('Complex: ') == 2


# --- strip_scaffolding ---


@pytest.mark.parametrize(
    'raw',
    [
        ' The cat sat down.',
        'Sure! The cat sat down.',
        "Sure, here's a simplified version: The cat sat down.",
        'Here is the simpler sentence: The cat sat down.',
        'Simplified version: The cat sat down.',
        'Simple: The cat sat down.',
        '"The cat sat down."',
        '“The cat sat down.”',
        '\n\nThe cat sat down.',
    ],
)
def test_strip_scaffolding_recovers_the_bare_sentence(raw):
    assert prompting.strip_scaffolding(raw) == 'The cat sat down.'


def test_strip_scaffolding_truncates_at_a_fabricated_next_turn():
    raw = 'The cat sat down.\nComplex: Another sentence.\nSimple: Another simple one.'
    assert prompting.strip_scaffolding(raw) == 'The cat sat down.'


def test_strip_scaffolding_takes_only_the_first_line():
    raw = 'The cat sat down.\nI hope this helps!'
    assert prompting.strip_scaffolding(raw) == 'The cat sat down.'


def test_strip_scaffolding_fixes_spaces_before_punctuation():
    assert prompting.strip_scaffolding('The cat sat down .') == 'The cat sat down.'


def test_strip_scaffolding_returns_empty_for_blank_output():
    assert prompting.strip_scaffolding('   \n  \n ') == ''


def test_strip_scaffolding_keeps_a_legitimate_multi_sentence_split():
    """Splitting one complex sentence into several is an explicitly requested
    operation, so two sentences on one line must survive intact."""
    raw = 'The cat sat down. Then it slept.'
    assert prompting.strip_scaffolding(raw) == raw


# --- classify_output ---

ORIGINAL = 'The felid subsequently assumed a seated posture upon the floor covering.'


def test_classify_output_accepts_a_good_simplification():
    assert prompting.classify_output(ORIGINAL, 'The cat sat down on the rug.') is None


def test_classify_output_accepts_an_unchanged_copy():
    # Copying the source is a measurable behaviour, not an error -- see the
    # docstring on classify_output.
    assert prompting.classify_output(ORIGINAL, ORIGINAL) is None


@pytest.mark.parametrize(
    'candidate,expected',
    [
        ('', 'empty'),
        ('   ', 'empty'),
        ('Complex: The felid sat.', 'prompt_echo'),
        ("I'm sorry, I can't help with that request.", 'refusal'),
        ('As an AI language model, I cannot rewrite this.', 'refusal'),
        ('The cat sat sat sat sat sat down.', 'degenerate_repetition'),
        ('...', 'no_words'),
    ],
)
def test_classify_output_flags_failure_modes(candidate, expected):
    assert prompting.classify_output(ORIGINAL, candidate) == expected


def test_classify_output_flags_runaway_length():
    # Distinct filler words, so this trips the length ratio and not the
    # repetition check -- repeating one word 100 times is degenerate output,
    # which is a different failure mode with its own reason code.
    candidate = ' '.join(f'filler{i}' for i in range(100))
    assert prompting.classify_output(ORIGINAL, candidate) == 'too_long'


@pytest.mark.parametrize(
    'fragment,rewrite',
    [
        ('Published', 'Published on this date'),
        ('Home', 'Home page'),
        ('By', 'Written by'),
        ('Skip to content', 'Go to the main content of the page'),
        ('Cookie settings', 'Cookie preferences you can change'),
    ],
)
def test_short_fragments_are_not_rejected_as_too_long(fragment, rewrite):
    """Regression: measured against qwen2.5-7b on 2026-08-14, the bare 3x ratio
    rejected essentially every short navigation fragment a real page contains,
    because 3x one word rejects any rewrite over three words. These must pass the
    length check -- LENGTH_HEADROOM_WORDS is what makes that true."""
    assert prompting.classify_output(fragment, rewrite) != 'too_long'


def test_short_fragment_rewritten_into_a_non_sequitur_is_flagged():
    """The failure the seq2seq path already guards against (main._looks_like_
    hallucination), reproduced on the LLM path: a one-word input coming back as an
    unrelated phrase is a fabrication, and must be distinguishable in the logs from
    a merely over-long generation."""
    assert (
        prompting.classify_output('Published', 'Other websites')
        == 'unrelated_short_output'
    )


def test_long_input_rewritten_unrelatedly_is_not_flagged_as_short_output():
    """The non-sequitur check applies only to short inputs -- a full sentence has
    enough context that a low-overlap rewrite is plausibly a real paraphrase."""
    long_input = 'The commission dismissed the complaint in its entirety after review.'
    assert prompting.classify_output(long_input, 'They threw it out.') is None


def test_classify_output_allows_a_reasonable_expansion():
    """Splitting and elaborating legitimately lengthen the output; only a
    blow-up past MAX_LENGTH_RATIO counts as runaway."""
    candidate = (
        'The cat sat down on the floor covering. It stayed there. The floor '
        'covering was a rug.'
    )
    assert prompting.classify_output(ORIGINAL, candidate) is None


# --- sanitize ---


def test_sanitize_returns_cleaned_text_and_no_reason_on_success():
    text, reason, model_result = prompting.sanitize(
        ORIGINAL, 'Sure! The cat sat down on the rug.'
    )
    assert text == 'The cat sat down on the rug.'
    assert reason is None
    assert model_result == text


def test_sanitize_falls_back_to_the_original_on_a_refusal():
    text, reason, _ = prompting.sanitize(ORIGINAL, "I'm sorry, I cannot do that.")
    assert text == ORIGINAL
    assert reason == 'refusal'


def test_sanitize_falls_back_to_the_original_on_empty_output():
    text, reason, _ = prompting.sanitize(ORIGINAL, '')
    assert text == ORIGINAL
    assert reason == 'empty'


def test_sanitize_strips_before_classifying():
    """A refusal wrapped in a conversational preamble must still be caught as a
    refusal, not slip through because the preamble shifted the match."""
    text, reason, _ = prompting.sanitize(
        ORIGINAL, "Sure! I can't rewrite this sentence."
    )
    assert text == ORIGINAL
    assert reason == 'refusal'


def test_sanitize_keeps_the_rejected_output_as_the_model_result():
    """The served text falls back to the original, but what the model actually said is
    still reported -- it is the whole point of the History log's middle column."""
    _, reason, model_result = prompting.sanitize(
        ORIGINAL, "Sure! I'm sorry, I cannot do that."
    )
    assert reason == 'refusal'
    assert model_result == "I'm sorry, I cannot do that."


# --- decoding settings ---


def test_serving_decoding_is_deterministic():
    assert prompting.SERVING_DECODING['temperature'] == 0.0


def test_evaluation_decoding_matches_bless_section_3_4():
    assert prompting.EVALUATION_DECODING['temperature'] == 1.0
    assert prompting.EVALUATION_DECODING['top_p'] == 0.9
    assert prompting.EVALUATION_DECODING['num_predict'] == 100


def test_both_decoding_configs_carry_the_stop_sequences():
    for config in (prompting.SERVING_DECODING, prompting.EVALUATION_DECODING):
        assert config['stop'] == list(prompting.STOP_SEQUENCES)


def test_tidy_punctuation_matches_the_seq2seq_paths_behaviour():
    # Deliberately doesn't import main: without transformers/torch this file runs in
    # milliseconds. main's seq2seq path calls this same function, so there is one
    # implementation.
    assert prompting.tidy_punctuation('Hello , world . Is this right ?') == (
        'Hello, world. Is this right?'
    )


# --- few-shot provenance ---


def test_default_pool_comes_from_asset_and_covers_every_named_operation():
    """The instruction names paraphrasing, compression and splitting. A demonstration
    set that never splits anything argues against a third of its own instruction --
    which is what an unfiltered seed-42 draw produced, and why
    research/select_fewshot.py filters on source length. Guards the property, not the exact rows.
    """
    pool = prompting.FEWSHOT_EXAMPLES[Audience.NON_NATIVE_SPEAKERS]
    assert len(pool) == prompting.FEWSHOT_N

    # at least one demonstration splits one sentence into more than one
    assert any(ex.simple.count('.') > ex.complex.count('.') for ex in pool)
    # at least one compresses
    assert any(len(ex.simple.split()) < len(ex.complex.split()) for ex in pool)


@pytest.mark.parametrize('audience', list(Audience))
def test_no_demonstration_carries_a_crowdworker_punctuation_typo(audience):
    """ASSET is crowdsourced and contains malformed references (e.g. "mistakes,.").
    A demonstration carrying one teaches the model to reproduce it."""
    for ex in prompting.FEWSHOT_EXAMPLES[audience]:
        for field in (ex.complex, ex.simple):
            assert not re.search(r'[,;:]\s*[.!?]$|\.\.|,,|\s,', field), field
            assert field[0].isupper(), field
            assert field.rstrip()[-1] in '.!?', field
