"""The one vocabulary, checked against every place that is supposed to use it.

`method` and `granularity` are defined twice by necessity -- once in Python
(`backend/vocabulary.py`) and once in JavaScript
(`extension/shared/model-labels.js`), because the two halves of this project don't
share a runtime. This reads the JavaScript as text and fails if either file says
anything the other doesn't.

It also checks the values are actually *used* rather than re-typed as literals: a
constant nobody references is not a single source of truth.
"""

import re
from pathlib import Path

import main
import prompting
import pytest
import vocabulary

REPO = Path(__file__).resolve().parents[2]
MODEL_LABELS_JS = REPO / 'extension' / 'shared' / 'model-labels.js'


def _js_const(source: str, name: str) -> str:
    """The string a `const NAME = "value";` line assigns."""
    match = re.search(rf'^const {name} = "([^"]+)";$', source, re.MULTILINE)
    assert match, f'{name} is not declared as a plain string const in model-labels.js'
    return match.group(1)


def _js_label(source: str, const_name: str) -> str:
    """The label a `[CONST_NAME]: "Label",` entry maps to."""
    match = re.search(rf'^\s*\[{const_name}\]: "([^"]+)",$', source, re.MULTILINE)
    assert match, f'no label entry for {const_name} in model-labels.js'
    return match.group(1)


@pytest.fixture(scope='module')
def js() -> str:
    return MODEL_LABELS_JS.read_text()


# --- the two files agree ----------------------------------------------------

VALUE_CONSTS = [
    ('METHOD_FINE_TUNED_SEQ2SEQ', vocabulary.METHOD_FINE_TUNED_SEQ2SEQ),
    ('METHOD_PROMPTED_LLM', vocabulary.METHOD_PROMPTED_LLM),
    ('GRANULARITY_SENTENCE_BY_SENTENCE', vocabulary.GRANULARITY_SENTENCE_BY_SENTENCE),
    ('GRANULARITY_WHOLE_SECTIONS', vocabulary.GRANULARITY_WHOLE_SECTIONS),
]


@pytest.mark.parametrize('name,expected', VALUE_CONSTS)
def test_javascript_declares_the_same_value(js, name, expected):
    assert _js_const(js, name) == expected


@pytest.mark.parametrize(
    'name,expected',
    [
        (
            'METHOD_FINE_TUNED_SEQ2SEQ',
            vocabulary.METHOD_LABELS[vocabulary.METHOD_FINE_TUNED_SEQ2SEQ],
        ),
        (
            'METHOD_PROMPTED_LLM',
            vocabulary.METHOD_LABELS[vocabulary.METHOD_PROMPTED_LLM],
        ),
        (
            'GRANULARITY_SENTENCE_BY_SENTENCE',
            vocabulary.GRANULARITY_LABELS[vocabulary.GRANULARITY_SENTENCE_BY_SENTENCE],
        ),
        (
            'GRANULARITY_WHOLE_SECTIONS',
            vocabulary.GRANULARITY_LABELS[vocabulary.GRANULARITY_WHOLE_SECTIONS],
        ),
    ],
)
def test_javascript_uses_the_same_label(js, name, expected):
    """One label per value, on both sides."""
    assert _js_label(js, name) == expected


def test_neither_side_has_a_value_the_other_lacks(js):
    py_values = set(vocabulary.METHODS) | set(vocabulary.GRANULARITIES)
    # only the `const NAME = "value";` declarations -- METHOD_LABELS and
    # GRANULARITY_LABELS share the prefix but are the label maps, not values
    js_values = set(
        re.findall(
            r'^const (?:METHOD|GRANULARITY)_[A-Z0-9_]+ = "([^"]+)";$', js, re.MULTILINE
        )
    )
    assert js_values == py_values


# --- the values are used, not re-typed --------------------------------------


def test_every_configured_model_declares_a_known_method_and_granularity():
    for model_key, (
        _env,
        _default,
        method,
        granularity,
    ) in main.MODEL_ENV_CONFIG.items():
        assert method in vocabulary.METHODS, f'{model_key} has an unknown method'
        assert (
            granularity in vocabulary.GRANULARITIES
        ), f'{model_key} has an unknown granularity'


def test_both_methods_are_actually_configured():
    """If a method has no model behind it the comparison this project exists to make
    has quietly lost half of itself."""
    configured = {method for _e, _d, method, _g in main.MODEL_ENV_CONFIG.values()}
    assert configured == set(vocabulary.METHODS)


def test_the_simplifier_implementations_carry_the_matching_method():
    assert main.Seq2SeqSimplifier.method == vocabulary.METHOD_FINE_TUNED_SEQ2SEQ
    assert main.PromptedLLMSimplifier.method == vocabulary.METHOD_PROMPTED_LLM


def test_prompting_reuses_the_vocabulary_rather_than_its_own_words():
    assert prompting.SENTENCE_BY_SENTENCE == vocabulary.GRANULARITY_SENTENCE_BY_SENTENCE
    assert prompting.WHOLE_SECTIONS == vocabulary.GRANULARITY_WHOLE_SECTIONS


def test_the_max_length_map_is_keyed_by_granularity():
    assert set(main.SEQ2SEQ_MAX_LENGTH) == set(vocabulary.GRANULARITIES)


def test_no_python_module_still_uses_the_old_words():
    """The values these replaced were bare `"sentence"` / `"document"` strings, which
    are ordinary English and so can't just be grepped for. What can be checked is the
    shape they appeared in: a granularity compared against, or a dict keyed by them."""
    stale = re.compile(r'granularity\s*[=!]=\s*"(?:sentence|document)"')
    for path in [main.__file__, prompting.__file__]:
        source = Path(path).read_text()
        assert not stale.search(
            source
        ), f'{Path(path).name} still compares granularity to an old value'
