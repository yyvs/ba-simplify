"""Prompt construction and output sanitising for the prompted-LLM simplification
method ("Method 3" -- see ``docs/roadmap-open-llm.md``).

This module is deliberately free of any model/HTTP dependency: everything here is
a pure function over strings, so it can be unit-tested in the existing fully-mocked,
offline test suite without pulling in Ollama or Hugging Face weights. ``main.py``
imports from here; nothing in this file imports from there.

The prompt is BLESS **Prompt 2** (Kew et al., "BLESS: Benchmarking Large Language
Models on Sentence Simplification", EMNLP 2023, Figure 1c), which the paper found to
be the best-performing of its three candidates and used for all of its analysis. Its
wording is itself repurposed from the instructions given to the crowdworkers who
built ASSET (Alva-Manchego et al., 2020a).

The one deviation from the published prompt is the ``{target_audience}`` slot. BLESS
hard-codes "non-native speakers of English" (because that is who ASSET's references
were written for); we parameterise it so the same system can target children, readers
with low literacy, and so on. Note that the *whole* noun phrase is substituted,
including the "of English" part -- "children of English" would not be grammatical, so
that suffix has to live in the audience description rather than in the template.

Rendering with the default audience therefore reproduces BLESS Prompt 2 **verbatim**;
``tests/test_prompt_builder.py`` asserts this, so the reproduction can't silently rot.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from vocabulary import GRANULARITY_SENTENCE_BY_SENTENCE, GRANULARITY_WHOLE_SECTIONS

#: The unit of text a prompt targets. Prompt building and output sanitising both
#: depend on it, most importantly the stop sequences and line handling (see
#: DOCUMENT_STOP_SEQUENCES and strip_scaffolding). Re-exported from `vocabulary`.
SENTENCE_BY_SENTENCE = GRANULARITY_SENTENCE_BY_SENTENCE
WHOLE_SECTIONS = GRANULARITY_WHOLE_SECTIONS


class Audience(str, Enum):
    """Target audiences the user can pick from.

    Deliberately a closed enum rather than a free-text field, for two reasons:
    the rendered prompt also carries attacker-controlled web page text, so a
    free-text audience would be a second injection vector; and free text would
    make evaluation runs non-reproducible.

    Inherits from ``str`` so it serialises straight to JSON and is usable directly
    as a Pydantic field type (``main.SimplifyRequest.audience``) with no custom
    encoder.
    """

    NON_NATIVE_SPEAKERS = 'non_native_speakers'
    CHILDREN = 'children'
    LOW_LITERACY = 'low_literacy'
    COGNITIVE_DISABILITY = 'cognitive_disability'
    GENERAL_ADULT = 'general_adult'


DEFAULT_AUDIENCE = Audience.NON_NATIVE_SPEAKERS
"""BLESS's own target audience -- the only one for which reference-based metrics
(SARI, BERTScore against ASSET) are actually valid, and therefore the default."""


# The noun phrase substituted into the prompt. Must read grammatically after
# "...easier to understand by ", and must carry its own "of English" where that
# makes sense (see the module docstring).
AUDIENCE_DESCRIPTIONS: Dict[Audience, str] = {
    Audience.NON_NATIVE_SPEAKERS: 'non-native speakers of English',
    Audience.CHILDREN: 'children aged 8 to 12',
    Audience.LOW_LITERACY: 'adult readers with low literacy skills',
    Audience.COGNITIVE_DISABILITY: 'readers with cognitive disabilities',
    Audience.GENERAL_ADULT: 'a general adult audience with no specialist knowledge',
}

# Labels for the extension's audience picker (in-page panel and toolbar), served via
# /health so the frontend holds no copy of the wording.
AUDIENCE_LABELS: Dict[Audience, str] = {
    Audience.NON_NATIVE_SPEAKERS: 'Non-native speakers',
    Audience.CHILDREN: 'Children · 8–12',
    Audience.LOW_LITERACY: 'Low literacy',
    Audience.COGNITIVE_DISABILITY: 'Cognitive disabilities',
    Audience.GENERAL_ADULT: 'General adult',
}


INSTRUCTION_TEMPLATE = (
    'Please rewrite the following complex sentence in order to make it easier to '
    'understand by {target_audience}. You can do so by replacing complex words with '
    'simpler synonyms (i.e. paraphrasing), deleting unimportant information (i.e. '
    'compression), and/or splitting a long complex sentence into several simpler '
    'ones. The final simplified sentence needs to be grammatical, fluent, and retain '
    'the main ideas of its original counterpart without altering its meaning.'
)


class Example(NamedTuple):
    """One in-context complex/simple demonstration pair."""

    complex: str
    simple: str


FEWSHOT_N = 3
"""BLESS §3.3 uses N=3 few-shot examples for every generation setting."""


# Few-shot pools, one per audience. Provenance differs per pool.
#
# NON_NATIVE_SPEAKERS (the pool BLESS comparability depends on) is sampled from the
# ASSET validation split, as in BLESS §3.3, by `research/select_fewshot.py`: seed 42,
# indices [285, 1516, 1116]. Never the test split (the 359 evaluated sentences; that
# would leak). That script's docstring gives the four selection rules (an unfiltered
# draw at this seed yields three short, substitution-only pairs, and ASSET's
# crowdsourced references include malformed and meaning-changing ones).
#
# The other four pools are hand-authored: no parallel corpus exists for those
# audiences, so non-default audience results rest on demonstrations with no corpus
# backing (a limitation). Newsela's graded levels 1-4 are the one plausible source if
# data access comes through.
#
# Per-audience pools because the ASSET demonstrations target non-native speakers;
# paired with e.g. a "children" instruction they contradict it, and demonstrations
# generally win. In the Phase 1 spike all five audiences produced distinct, on-target
# output.
FEWSHOT_EXAMPLES: Dict[Audience, Tuple[Example, ...]] = {
    # Sampled from ASSET validation by research/select_fewshot.py (seed 42). Between
    # them the three cover all the operations the instruction names: [285] paraphrases and
    # drops a hedge, [1516] paraphrases and compresses, [1116] splits one sentence
    # into two.
    Audience.NON_NATIVE_SPEAKERS: (
        # ASSET validation[285]
        Example(
            complex=(
                'To avoid adversaries from guessing the key using a brute-force attack, '
                'the key space is usually designed to be extremely large.'
            ),
            simple=(
                'To keep foes from guessing the key using a brute-force attack, the key '
                'space is designed to be extremely large.'
            ),
        ),
        # ASSET validation[1516]
        Example(
            complex=(
                'She is the daughter of well-known Hong Kong singer Teresa Carpio, with '
                'whom she has performed on stage as a backup singer.'
            ),
            simple=(
                'She is the daughter of famous Hong Kong singer Teresa Carpio, with whom '
                'she performed as a backup singer.'
            ),
        ),
        # ASSET validation[1116]
        Example(
            complex=(
                'Other Mario sports games include the Camelot-developed series Mario Golf '
                'and Mario Tennis, and, respectively, the baseball and soccer games Mario '
                'Superstar Baseball and Super Mario Strikers.'
            ),
            simple=(
                'Other Mario sports games are the Mario Golf and Mario Tennis. Also, the '
                'baseball and soccer games Mario Superstar Baseball and Super Mario '
                'Strikers.'
            ),
        ),
    ),
    Audience.CHILDREN: (
        Example(
            complex=(
                'The commission subsequently determined that the allegations were '
                'unsubstantiated and dismissed the complaint in its entirety.'
            ),
            simple='The group looked at the claims. They found no proof, so they said no to the complaint.',
        ),
        Example(
            complex=(
                'Owing to inclement weather conditions, the outdoor ceremony was '
                'relocated to an adjacent indoor venue.'
            ),
            simple='The weather was bad. So the party moved inside to a room next door.',
        ),
        Example(
            complex=(
                'He is widely regarded as one of the most influential composers of the '
                'Romantic era, a period spanning roughly from 1800 to 1910.'
            ),
            simple=(
                'He wrote music a long time ago. Lots of people think he was one of the '
                'best music writers of his time.'
            ),
        ),
    ),
    Audience.LOW_LITERACY: (
        Example(
            complex=(
                'The commission subsequently determined that the allegations were '
                'unsubstantiated and dismissed the complaint in its entirety.'
            ),
            simple='The commission checked the claims. They found no proof. They rejected the complaint.',
        ),
        Example(
            complex=(
                'Owing to inclement weather conditions, the outdoor ceremony was '
                'relocated to an adjacent indoor venue.'
            ),
            simple='The weather was bad. The ceremony moved indoors. It took place in a room next door.',
        ),
        Example(
            complex=(
                'He is widely regarded as one of the most influential composers of the '
                'Romantic era, a period spanning roughly from 1800 to 1910.'
            ),
            simple=(
                'He wrote music. Many people say he was one of the best. He lived in a '
                'time called the Romantic era.'
            ),
        ),
    ),
    Audience.COGNITIVE_DISABILITY: (
        Example(
            complex=(
                'The commission subsequently determined that the allegations were '
                'unsubstantiated and dismissed the complaint in its entirety.'
            ),
            simple='A group of people checked the complaint. They found no proof. They said no to it.',
        ),
        Example(
            complex=(
                'Owing to inclement weather conditions, the outdoor ceremony was '
                'relocated to an adjacent indoor venue.'
            ),
            simple='The weather was bad. The event went inside. It was in the room next door.',
        ),
        Example(
            complex=(
                'He is widely regarded as one of the most influential composers of the '
                'Romantic era, a period spanning roughly from 1800 to 1910.'
            ),
            simple='He made music. People liked his music very much. He lived long ago.',
        ),
    ),
    Audience.GENERAL_ADULT: (
        Example(
            complex=(
                'The commission subsequently determined that the allegations were '
                'unsubstantiated and dismissed the complaint in its entirety.'
            ),
            simple='The commission found no evidence for the claims and threw out the complaint.',
        ),
        Example(
            complex=(
                'Owing to inclement weather conditions, the outdoor ceremony was '
                'relocated to an adjacent indoor venue.'
            ),
            simple='Bad weather forced the outdoor ceremony to move to an indoor venue next door.',
        ),
        Example(
            complex=(
                'He is widely regarded as one of the most influential composers of the '
                'Romantic era, a period spanning roughly from 1800 to 1910.'
            ),
            simple=(
                'He is seen as one of the most influential composers of the Romantic era, '
                'which ran from about 1800 to 1910.'
            ),
        ),
    ),
}


def audience_description(audience: Audience) -> str:
    """The noun phrase substituted into the prompt for ``audience``."""
    return AUDIENCE_DESCRIPTIONS[Audience(audience)]


def render_instruction(audience: Audience = DEFAULT_AUDIENCE) -> str:
    """Just the task-instruction block (the blue box in BLESS Figure 1)."""
    return INSTRUCTION_TEMPLATE.format(target_audience=audience_description(audience))


def render_document_instruction(audience: Audience = DEFAULT_AUDIENCE) -> str:
    """The document-scope instruction block. An adaptation, not BLESS -- see
    ``DOCUMENT_INSTRUCTION_TEMPLATE``."""
    return DOCUMENT_INSTRUCTION_TEMPLATE.format(
        target_audience=audience_description(audience)
    )


def build_prompt(
    text: str,
    audience: Audience = DEFAULT_AUDIENCE,
    examples: Optional[Sequence[Example]] = None,
    granularity: str = SENTENCE_BY_SENTENCE,
) -> str:
    """Render the prompt for ``text`` at the given granularity.

    **Sentence** (default) renders BLESS Prompt 2 exactly as Figure 1c lays it out: the
    instruction, then N structured ``Complex:``/``Simple:`` demonstration pairs, then
    the input sentence and a dangling ``Simple:`` prefix to continue from.
    ``examples`` defaults to the pool for ``audience``; an empty sequence gives the
    instruction-only (zero-shot) variant, the documented fallback for any audience with
    no aligned demonstrations.

    **Document** renders the adapted instruction with a ``Text:``/``Simplified text:``
    frame, and is **zero-shot by default**, deliberately:

    1. *Context budget.* A document demonstration is a whole document pair. Three of
       them at D-Wikipedia's measured mean (~172 source tokens) would spend well over a
       thousand tokens of context before the actual input, and the input here is a whole
       page section.
    2. *No usable corpus to sample from.* The obvious source, D-Wikipedia, is lowercased
       and PTB-pre-tokenized (§2.3). Demonstrating with it would teach the model to emit
       lowercase, pre-tokenized text -- the exact convention `document_text.py` exists to
       undo for the seq2seq model. A demonstration that has to be post-processed is a
       bad demonstration.
    3. *Honesty about provenance.* Hand-authoring a document pair and presenting it
       alongside corpus-sampled sentence demonstrations would blur where the examples
       came from. Zero-shot is a cleaner claim.

    Passing ``examples`` explicitly overrides this for anyone who wants to test few-shot
    document prompting; nothing in the design prevents it.
    """
    blocks: List[str]
    if granularity == WHOLE_SECTIONS:
        blocks = [render_document_instruction(audience)]
        for ex in examples or ():
            blocks.append(f'Text: {ex.complex}\nSimplified text: {ex.simple}')
        blocks.append(f'Text: {text}\nSimplified text:')
        return '\n\n'.join(blocks)

    if examples is None:
        examples = FEWSHOT_EXAMPLES.get(Audience(audience), ())

    blocks = [render_instruction(audience)]
    for ex in examples:
        blocks.append(f'Complex: {ex.complex}\nSimple: {ex.simple}')
    blocks.append(f'Complex: {text}\nSimple:')
    return '\n\n'.join(blocks)


# --- Decoding settings ---
# Serving and evaluation configs, explicit rather than server defaults. Keys follow
# Ollama's ``options`` object; ``main.PromptedLLMSimplifier`` passes them through.

STOP_SEQUENCES: Tuple[str, ...] = ('\nComplex:', '\nSimple:', '\n\n')
"""Cut generation off before the model invents a further demonstration pair, which
is the single most common failure of a few-shot completion prompt."""

SERVING_DECODING: Dict[str, object] = {
    'temperature': 0.0,
    'num_predict': 200,
    'stop': list(STOP_SEQUENCES),
}
"""Greedy, for the extension. Users re-simplifying the same page should get the
same text back; sampling would make the output jitter between runs."""

EVALUATION_DECODING: Dict[str, object] = {
    'temperature': 1.0,
    'top_p': 0.9,
    'num_predict': 100,
    'stop': list(STOP_SEQUENCES),
}
"""BLESS §3.4: nucleus sampling with p=0.9, temperature 1.0, max 100 new tokens.
The paper runs every configuration under 3 random seeds and aggregates; the seed is
supplied per-run by ``research/evaluate_llm.py`` rather than baked in here."""


# --- Document granularity ---
# Everything above is sentence-scoped, as BLESS is a sentence simplification benchmark.
# Document-level prompting is this project's own extension; nothing below is from BLESS.

DOCUMENT_INSTRUCTION_TEMPLATE = (
    'Please rewrite the following text in order to make it easier to understand by '
    '{target_audience}. You can do so by replacing complex words with simpler synonyms '
    '(i.e. paraphrasing), deleting unimportant information (i.e. compression), '
    'splitting long complex sentences into several simpler ones, and merging or '
    'reordering sentences where that makes the whole text clearer. The result needs to '
    'be grammatical, fluent, and retain the main ideas of the original without altering '
    'its meaning. Return only the rewritten text as continuous prose -- no commentary, '
    'no headings, no bullet points, and no introduction such as "Here is the '
    'simplified text".'
)
"""An **adaptation** of BLESS Prompt 2 to document scope -- explicitly not a citation.

BLESS is sentence-only, so no part of the paper's evaluation applies to this template
and it must never be described as "BLESS Prompt 2" in the write-up. What is carried over
is the *structure* (name the audience, enumerate the permitted operations, state the
fluency/meaning constraints), because that structure is what the paper found to work.

Three deliberate differences from the sentence template, each with a reason:

- **Merging and reordering are added to the permitted operations.** They are the
  operations a document model has and a sentence model structurally cannot -- rewriting
  across sentence boundaries is the whole point of document-level simplification, and
  is why D-SARI exists as a separate metric (the thesis §2.6).
- **"Return only the rewritten text..."** is an explicit instruction because at document
  scale an instruction-tuned chat model reliably adds a preamble, markdown headings, or
  bullet points. The output sanitiser strips what it can, but not adding it in the first
  place is better than removing it afterwards.
- **"continuous prose"** because the extension replaces a heading-delimited section with
  one block (§5.6); a bulleted rewrite would look like a different page rather than a
  simpler one.
"""

DOCUMENT_STOP_SEQUENCES: Tuple[str, ...] = ('\nText:', '\nSimplified text:')
"""Deliberately **excludes** ``"\\n\\n"``, which the sentence stop list includes.

A blank line is a paragraph break, which is legitimate -- and common -- inside a
document. Reusing the sentence stop sequences here would silently truncate every
multi-paragraph rewrite at its first paragraph break, which looks like the model
producing a short answer rather than like a parsing bug.
"""

# num_predict must fit a whole section; num_ctx the instruction, section and rewrite.
# Ollama's default context depends on VRAM (4096 on the dev machine), and an exceeded
# context silently drops the start of the prompt, i.e. the instruction. Explicit
# num_ctx costs KV-cache memory.
DOCUMENT_SERVING_DECODING: Dict[str, object] = {
    'temperature': 0.0,
    'num_predict': 1024,
    'num_ctx': 8192,
    'stop': list(DOCUMENT_STOP_SEQUENCES),
}

DOCUMENT_EVALUATION_DECODING: Dict[str, object] = {
    'temperature': 1.0,
    'top_p': 0.9,
    'num_predict': 1024,
    'num_ctx': 8192,
    'stop': list(DOCUMENT_STOP_SEQUENCES),
}
"""Same sampling parameters as the sentence evaluation config (so the two granularities
differ only in scope, not in decoding strategy), but with the document token budget.
Note that BLESS's 100-token cap is *not* carried over: it would truncate mid-document."""


def serving_decoding(granularity: str = SENTENCE_BY_SENTENCE) -> Dict[str, object]:
    return dict(
        DOCUMENT_SERVING_DECODING if granularity == WHOLE_SECTIONS else SERVING_DECODING
    )


def evaluation_decoding(granularity: str = SENTENCE_BY_SENTENCE) -> Dict[str, object]:
    return dict(
        DOCUMENT_EVALUATION_DECODING
        if granularity == WHOLE_SECTIONS
        else EVALUATION_DECODING
    )


# --- Output sanitising ---

# Conversational scaffolding that instruction-tuned chat models prepend even when
# the prompt is a bare completion. Anchored at the start and matched case-insensitively.
_PREFIX_RE = re.compile(
    r"""^\s*(?:
        (?:sure|certainly|of\s+course|okay|ok)\s*[,!.]?\s*
      | here(?:'s|\s+is)\s+(?:the\s+|a\s+)?(?:simplified|simpler|rewritten)\s+
        (?:version|sentence|text)\s*[:\-]?\s*
      | (?:simplified|simpler|rewritten)\s+(?:version|sentence|text)\s*[:\-]\s*
      | simple\s*:\s*
    )+""",
    re.IGNORECASE | re.VERBOSE,
)

# A model that ignores the completion format and starts a new turn instead.
_ECHO_RE = re.compile(r'^\s*complex\s*:', re.IGNORECASE)
# The document prompt frames its turns as Text:/Simplified text:, so a model that
# starts a new turn echoes those instead of "Complex:".
_DOCUMENT_ECHO_RE = re.compile(r'^\s*(?:text|simplified\s+text)\s*:', re.IGNORECASE)

_REFUSAL_RE = re.compile(
    r"\b(?:i(?:'m| am) sorry|i can(?:'t|not)|i'm unable to|as an ai\b|"
    r'i do not have the ability)',
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z']+")

# Markdown list markers at line start. The document instruction forbids bullets, but
# models don't always comply, and a bulleted section changes the page's structure.
_LIST_BULLET_RE = re.compile(r'^\s*(?:[-*\u2022]|\d+[.)])\s+')

# Matches the same word repeated four or more times in a row (allowing punctuation
# between), the classic shape of a degenerate loop.
_REPEAT_RE = re.compile(r'\b(\w+)\b(?:[\s,;:.]+\b\1\b){3,}', re.IGNORECASE)

MAX_LENGTH_RATIO = 3.0
"""Reject output more than this many times longer (in words) than the input. A
simplification may legitimately grow -- splitting one sentence into three, or
elaborating a medical term, as MED-EASI's references do -- but a 3x blow-up is
rambling, not simplification."""

LENGTH_HEADROOM_WORDS = 12
"""Absolute word allowance added on top of the ratio, so the length check means
"rambling" rather than "longer than a very short input".

Measured 2026-08-14 against qwen2.5-7b: the ratio on its own rejected almost every
short navigation fragment a real page is full of -- "Published", "Home", "By",
"Skip to content" -- because 3x a one-word input rejects any rewrite over three
words. Those all *should* end up serving the original text, but as
``unrelated_short_output`` or not at all, not as ``too_long``: miscoded rejections
would swamp the error categorisation in the thesis §6.3 with a guard
artifact and hide whatever genuine rambling exists."""

MIN_DOCUMENT_RETENTION = 0.25
"""Reject a document rewrite retaining less than this fraction of the source's word
count. Compression is an explicitly permitted operation, so the threshold is
deliberately lenient -- it is meant to catch a model that answered with a one-line
summary (or a heading) instead of a rewritten passage, not to police legitimate
shortening. Only applied at document granularity."""

SHORT_INPUT_WORDS = 3
"""At or below this many words, an input carries too little context to simplify
reliably, so a rewrite sharing none of its words is treated as a non-sequitur.

Mirrors ``main._looks_like_hallucination``, the equivalent guard on the seq2seq
path -- kept as a separate implementation here rather than imported, because
prompting.py deliberately has no dependency on main.py. Measured on the same run:
qwen2.5-7b turned "Published" into unrelated phrases, exactly the failure the
seq2seq path already had to defend against."""


def tidy_punctuation(s: str) -> str:
    """Remove stray spaces before punctuation.

    The one implementation both methods share: the seq2seq path imports it into
    ``main.Seq2SeqSimplifier.generate``, and the prompted path applies it as the last
    step of ``strip_scaffolding``.
    """
    return re.sub(r'\s+([\.,;:!?])', r'\1', s)


def truncate_at_stop(raw: str, granularity: str = SENTENCE_BY_SENTENCE) -> str:
    r"""Cut ``raw`` at the earliest stop sequence for this granularity.

    Ollama honours ``stop`` server-side, but not every backend does, and a stop
    sequence can still slip through when it straddles a streaming chunk boundary.
    Applying it again here makes the parsing independent of that.

    Leading whitespace is dropped *before* the search, which is load-bearing rather
    than cosmetic: completions very often begin with a newline (the prompt ends with
    a bare ``Simple:``), and ``"\\n\\n"`` is a stop sequence -- so searching the raw
    string would find a stop at index 0 and truncate the entire generation to the
    empty string. The stop sequences are meant to end the answer, not to precede it.
    """
    raw = raw.lstrip()
    cut = len(raw)
    stops = DOCUMENT_STOP_SEQUENCES if granularity == WHOLE_SECTIONS else STOP_SEQUENCES
    for stop in stops:
        idx = raw.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return raw[:cut]


def strip_scaffolding(raw: str, granularity: str = SENTENCE_BY_SENTENCE) -> str:
    """Reduce a raw completion to the bare simplified text.

    Drops conversational preambles and wrapping quotes -- models frequently return
    the text quoted, which would otherwise show up literally in the page.

    **The line handling differs by granularity, and getting it wrong is silent.** For a
    sentence the answer is one line, so everything after the first non-empty line is
    scaffolding ("I hope this helps!") and is dropped. For a document, a blank line is a
    paragraph break and later lines are the rest of the answer -- applying the
    sentence rule there would reduce a whole section to its opening sentence and look
    like the model being terse rather than like a parsing bug.
    """
    text = truncate_at_stop(raw, granularity)
    text = _PREFIX_RE.sub('', text)

    if granularity == WHOLE_SECTIONS:
        # Keep every line; just normalise the blank-line runs a chat model tends to
        # emit, and drop any markdown list bullets that slipped past the instruction.
        lines = [_LIST_BULLET_RE.sub('', ln.rstrip()) for ln in text.splitlines()]
        text = '\n'.join(lines).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                text = line
                break
        else:
            return ''

    # A second pass: the preamble and the answer sometimes share a line
    # ("Sure! Here's a simpler version: The cat sat down.").
    text = _PREFIX_RE.sub('', text).strip()

    if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()

    return tidy_punctuation(text)


def classify_output(
    original: str, candidate: str, granularity: str = SENTENCE_BY_SENTENCE
) -> Optional[str]:
    """Return a short reason code if ``candidate`` is unusable, else ``None``.

    A reason code rather than a bare bool so the caller can log *why* a generation
    was rejected; those counts feed the error categorisation planned for
    the thesis §6.3 directly.

    Note what is deliberately **not** rejected: output identical to the input.
    Copying the source is a real and well-documented LLM behaviour on this task
    (BLESS §6 shows BERTScore actively rewards it), so it is a *result to measure*,
    not an error to suppress.
    """
    text = candidate.strip()
    if not text:
        return 'empty'
    echo_re = _DOCUMENT_ECHO_RE if granularity == WHOLE_SECTIONS else _ECHO_RE
    if echo_re.search(text):
        return 'prompt_echo'
    if _REFUSAL_RE.search(text):
        return 'refusal'
    if _REPEAT_RE.search(text):
        return 'degenerate_repetition'

    original_words = _WORD_RE.findall(original)
    candidate_words = _WORD_RE.findall(text)
    if not candidate_words:
        return 'no_words'

    if original_words:
        # The ratio needs an absolute floor to be meaningful -- see
        # LENGTH_HEADROOM_WORDS for the measurement that forced this.
        budget = max(
            MAX_LENGTH_RATIO * len(original_words),
            len(original_words) + LENGTH_HEADROOM_WORDS,
        )
        if len(candidate_words) > budget:
            return 'too_long'

        # Document-only: 200 words in, 15 out is deletion, not simplification. At
        # document scale wholesale content loss is the dominant risk, which D-SARI
        # penalises and plain SARI does not (thesis §2.6).
        if granularity == WHOLE_SECTIONS:
            if len(candidate_words) < MIN_DOCUMENT_RETENTION * len(original_words):
                return 'too_short'
            # the short-input non-sequitur rule below is meaningless for documents
            return None

        # Short fragments (nav links, bylines, button labels) carry too little
        # context; a rewrite that shares none of their words is a fabrication
        # rather than a simplification.
        if len(original_words) <= SHORT_INPUT_WORDS:
            original_set = {w.lower() for w in original_words}
            candidate_set = {w.lower() for w in candidate_words}
            if original_set.isdisjoint(candidate_set):
                return 'unrelated_short_output'

    return None


def sanitize(
    original: str, raw: str, granularity: str = SENTENCE_BY_SENTENCE
) -> Tuple[str, Optional[str], str]:
    """Turn a raw completion into the text to actually serve.

    Returns ``(text, reason, model_result)``. On success ``reason`` is ``None``. On
    failure the original text is returned unchanged alongside the reason code -- the
    same fall-back-to-source policy the seq2seq path already applies via
    ``main._looks_like_hallucination``, so a bad generation degrades to "this
    sentence wasn't simplified" rather than to a non-sequitur on the page.

    ``model_result`` is the model's own answer either way (the de-scaffolded
    candidate, kept even when rejected), for the History log's
    input -> model result -> output chain.
    """
    candidate = strip_scaffolding(raw, granularity)
    reason = classify_output(original, candidate, granularity)
    if reason is not None:
        return original, reason, candidate
    return candidate, None, candidate
