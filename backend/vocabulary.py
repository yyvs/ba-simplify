"""The two dimensions every run is described by; the only place either is defined.

- **method** -- *how* the output is produced: a checkpoint fine-tuned on
  simplification pairs, or an instruction-tuned model prompted at inference time.
  The project's central comparison, so every report, log line and label names it.

- **granularity** -- *what unit of text* a request carries: one sentence at a time,
  or one self-contained run of prose passed whole.

Both are snake_case machine values: sent over the wire (``/health``), stored and
logged. The labels below are the only place they become display text, so wording can
change without stored values going stale. ``extension/shared/model-labels.js`` holds
the same pair for JavaScript; ``backend/tests/test_vocabulary.py`` catches a mismatch.

``whole_sections`` names the caller's unit, not the model's: the extension sends one
heading-delimited page section per request, ``research/evaluate_document.py`` one
whole article. To the backend both mean "one piece of prose, do not split it".
"""

from typing import Dict, Tuple

# --- method ----------------------------------------------------------------
# Also selects the Simplifier implementation in main.lifespan().
METHOD_FINE_TUNED_SEQ2SEQ = 'fine_tuned_seq2seq'
METHOD_PROMPTED_LLM = 'prompted_llm'

METHODS: Tuple[str, ...] = (METHOD_FINE_TUNED_SEQ2SEQ, METHOD_PROMPTED_LLM)

METHOD_LABELS: Dict[str, str] = {
    METHOD_FINE_TUNED_SEQ2SEQ: 'Fine-tuned seq2seq',
    METHOD_PROMPTED_LLM: 'Prompted LLM',
}

# --- granularity -----------------------------------------------------------
# SENTENCE_BY_SENTENCE: trained or prompted on single sentences (both seq2seq
#   checkpoints are fine-tuned on WikiLarge sentence pairs; the LLM uses the
#   single-sentence BLESS Prompt 2 template). ``simplify_text`` splits multi-sentence
#   input (see split_sentences()), simplifies each sentence independently and rejoins;
#   a multi-sentence block fed whole to a sentence-trained model gives context-free,
#   fractured output.
# WHOLE_SECTIONS: trained on whole documents (D-Wikipedia), given the cleaned input
#   unsplit up to its trained max length. Callers must stay under that ceiling:
#   anything past it is truncated, i.e. silently dropped (hence the extension chunks
#   per heading-delimited section). Needs its own text conventions in both
#   directions; see document_text.py.
GRANULARITY_SENTENCE_BY_SENTENCE = 'sentence_by_sentence'
GRANULARITY_WHOLE_SECTIONS = 'whole_sections'

GRANULARITIES: Tuple[str, ...] = (
    GRANULARITY_SENTENCE_BY_SENTENCE,
    GRANULARITY_WHOLE_SECTIONS,
)

GRANULARITY_LABELS: Dict[str, str] = {
    GRANULARITY_SENTENCE_BY_SENTENCE: 'Sentence by sentence',
    GRANULARITY_WHOLE_SECTIONS: 'Whole sections',
}


def method_label(method: str) -> str:
    """A stored method value read back for display.

    An unknown value is returned as it is rather than dropped: a log written by a
    later build naming a method this one doesn't have is still information about
    which run it was.
    """
    return METHOD_LABELS.get(method, method)


def granularity_label(granularity: str) -> str:
    return GRANULARITY_LABELS.get(granularity, granularity)
