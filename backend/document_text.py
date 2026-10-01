r"""Text conventions for the document-level (D-Wikipedia) simplifier.

The D-Wikipedia corpus this project's document checkpoint is trained on (see
``research/scratch/d_wikipedia_raw``) is written in a very specific style, which
inference has to match or the model drifts badly:

- **one document per line** -- there is no newline anywhere inside a document
  body. Sentences are separated only by their own terminal punctuation. Feeding
  the model ``\\n`` as a structural separator measurably degrades output (it
  dropped facts and hallucinated in testing), because it never saw one.
- **fully lowercased** -- there is no uppercase character anywhere in the
  corpus. Mixed-case input alone was enough to produce a factual hallucination
  in testing ("northern Netherlands" -> "northern hemisphere").
- **PTB-style pre-tokenized** -- punctuation is split off into its own token
  (``writer , intellectual``, ``netherlands .``), possessives are separated
  (``women 's``), and quotes are doubled backticks/apostrophes
  (```` `` the second sex '' ````). Numbers keep their internal separators
  (``28,000``, ``3.5``).

Since the model both consumes *and* emits that style, this module is a matched
pair: ``normalize_for_model`` converts ordinary web prose into it, and
``denormalize_from_model`` converts the output back into something presentable
on a web page (de-tokenized punctuation, restored capitalization). Without the
second half the model's raw output -- ``achtkarspelen is a municipality in
friesland .`` -- reads as broken text.
"""

import re
from typing import Dict, List

# Sentence-final punctuation. Used both to decide whether a segment already ends
# a sentence and to find where the next one starts when restoring capitals.
TERMINAL_PUNCTUATION = '.?!'

# Splits punctuation off as its own token, mirroring the corpus. Commas and
# periods are handled separately below, so numbers like '28,000' and '3.5' (which
# the corpus keeps intact) survive.
_PUNCT_TO_SPLIT = re.compile(r'([;:!?()\[\]])')
# a comma/period only becomes its own token when it isn't sitting between two
# digits -- '28,000' and '3.5' stay whole, 'words, more' and 'end.' split.
_COMMA_NOT_IN_NUMBER = re.compile(r'(?<!\d),|,(?!\d)')
# ...and a period stays whole when it closes a single-letter abbreviation: D-Wikipedia's
# test split has "u.s." 531 times and "u . s ." never (this function serves both the
# metric normalisation and the served document path). Matched by shape, not a list: a
# period after a letter not itself preceded by a letter ("u.s.", "e.g.", "a.m.", but
# not "end."). Cost: a sentence ending in a one-letter word ("... is a.") keeps its
# period attached, three orders of magnitude rarer here.
_ABBREV_LETTER = r'(?<![^\W\d_])[^\W\d_]'
_PERIOD_NOT_IN_NUMBER = re.compile(
    rf'(?<!\d)(?<!{_ABBREV_LETTER})\.|(?<!{_ABBREV_LETTER})\.(?!\d)'
)
_POSSESSIVE = re.compile(r"(\w)'(s\b|\s|$)")
_DOUBLE_QUOTE = re.compile(r'["“”]')

# Re-attaches punctuation to the preceding word on the way out.
_SPACE_BEFORE_PUNCT = re.compile(r'\s+([,.;:!?)\]])')
_SPACE_AFTER_OPEN = re.compile(r'([(\[])\s+')
_SPACE_BEFORE_POSSESSIVE = re.compile(r"\s+'(s\b)")
# The corpus writes quotes as `` ... '' (PTB convention); turn both back into ".
_OPEN_BACKTICKS = re.compile(r'``\s*')
_CLOSE_APOSTROPHES = re.compile(r"\s*''")
# An empty "( )" is an artifact of the corpus stripping parenthesized
# pronunciation/IPA blocks; it carries no text, so drop it rather than render it.
_EMPTY_PARENS = re.compile(r'\(\s*\)')
# ...and the space the model drops: the decoder sometimes omits the space after a
# punctuation token, so "boundaries . the movement" comes back as
# "boundaries .the movement", which _SPACE_BEFORE_PUNCT then fuses into
# "boundaries.the movement".
#
# Document mode splits the answer on sentence boundaries to assign sentences to
# paragraphs (content.js's distributeSentences), so one fused pair shifts every later
# paragraph by one (observed on a BBC article, for eight paragraphs).
#
# Same abbreviation guard as _PERIOD_NOT_IN_NUMBER. No URL guard ("example.com/a"):
# isSimplifiableChunk() drops chunks containing "http://" or "https://" before sending.
_MISSING_SPACE_AFTER_SENTENCE = re.compile(rf'(?<!{_ABBREV_LETTER})([.?!])(?=[^\W\d_])')
# The same dropped marker on a comma or semicolon. No abbreviation exists to protect
# here, and the letter lookahead is what keeps "28,000" and "3.5" intact.
_MISSING_SPACE_AFTER_COMMA = re.compile(r'([,;])(?=[^\W\d_])')
_MULTI_SPACE = re.compile(r'\s+')

_WORD = re.compile(r'[^\W\d_]+', re.UNICODE)


def ensure_terminal_punctuation(text: str) -> str:
    """Appends a period unless `text` already ends a sentence.

    Applied per segment before segments are joined into one document. Without
    it, a segment that doesn't end in punctuation (a heading, a bare list item)
    runs straight into the next one and the model reads them as a single
    thought -- e.g. "history" + "the igda is..." was simplified as the single
    phrase "history the igda is a non-profit organization".
    """
    stripped = text.rstrip()
    if not stripped:
        return stripped
    return stripped if stripped[-1] in TERMINAL_PUNCTUATION else stripped + '.'


def join_segments(segments: List[str]) -> str:
    """Joins per-element texts into one document body the model will recognise.

    Plain spaces, never newlines (the corpus has none), with each segment
    guaranteed to end a sentence first -- joining on ". " instead would produce
    ".." wherever a segment already ended in a period.
    """
    prepared = [ensure_terminal_punctuation(s) for s in segments]
    return ' '.join(s for s in prepared if s)


def _ptb_quotes(text: str) -> str:
    """Rewrites "quoted" spans the way the corpus does: `` to open, '' to close.

    Alternates on each quote character rather than pairing them up, which is
    what PTB tokenization does and is robust to an unbalanced quote (the run
    just continues alternating rather than failing).
    """
    state = {'open': True}

    def swap(_match):
        marker = ' `` ' if state['open'] else " '' "
        state['open'] = not state['open']
        return marker

    return _DOUBLE_QUOTE.sub(swap, text)


def normalize_for_model(text: str) -> str:
    """Rewrites ordinary prose into the corpus's lowercased, pre-tokenized form."""
    text = text.lower()
    text = _ptb_quotes(text)
    text = _POSSESSIVE.sub(r"\1 '\2", text)
    text = _PUNCT_TO_SPLIT.sub(r' \1 ', text)
    text = _COMMA_NOT_IN_NUMBER.sub(' , ', text)
    text = _PERIOD_NOT_IN_NUMBER.sub(' . ', text)
    return _MULTI_SPACE.sub(' ', text).strip()


def _casing_map(source: str) -> Dict[str, str]:
    """Maps lowercase word -> how it was capitalized in the source text.

    Only words capitalized somewhere *other* than a sentence start are
    recorded: a sentence-initial "The" says nothing about how "the" should look
    mid-sentence, whereas a mid-sentence "Friesland" or "IGDA" is real evidence
    of a proper noun or acronym. The model always emits lowercase, so this is
    the only way to get that casing back.
    """
    mapping: Dict[str, str] = {}
    # True while the next word would be sentence-initial (start of text, or
    # straight after terminal punctuation).
    at_sentence_start = True
    for match in re.finditer(r'[^\W\d_]+|[.?!]', source, re.UNICODE):
        token = match.group(0)
        if token in TERMINAL_PUNCTUATION:
            at_sentence_start = True
            continue
        if not at_sentence_start and token[:1].isupper():
            mapping.setdefault(token.lower(), token)
        at_sentence_start = False
    return mapping


def _restore_sentence_capitals(text: str) -> str:
    """Upper-cases the first letter of the text and of every later sentence."""
    out = list(text)
    capitalize_next = True
    for i, ch in enumerate(out):
        if capitalize_next and ch.isalpha():
            out[i] = ch.upper()
            capitalize_next = False
        elif ch in TERMINAL_PUNCTUATION:
            capitalize_next = True
    return ''.join(out)


def denormalize_from_model(text: str, source: str = '') -> str:
    """Turns raw model output back into presentable prose.

    Re-attaches split punctuation, restores quotes, then restores casing:
    proper nouns and acronyms from `source` (see ``_casing_map``) followed by
    sentence-initial capitals. Pass the model's *input* as `source` -- without
    it only sentence starts can be recovered, and every proper noun stays
    lowercase.
    """
    text = _OPEN_BACKTICKS.sub('"', text)
    text = _CLOSE_APOSTROPHES.sub('"', text)
    text = _EMPTY_PARENS.sub(' ', text)
    text = _SPACE_BEFORE_POSSESSIVE.sub(r"'\1", text)
    text = _SPACE_BEFORE_PUNCT.sub(r'\1', text)
    text = _SPACE_AFTER_OPEN.sub(r'\1', text)
    # after the re-attachment above, never before it: the space this puts back is the
    # one _SPACE_BEFORE_PUNCT has just closed up ("boundaries .the" -> "boundaries.the")
    text = _MISSING_SPACE_AFTER_SENTENCE.sub(r'\1 ', text)
    text = _MISSING_SPACE_AFTER_COMMA.sub(r'\1 ', text)
    text = _MULTI_SPACE.sub(' ', text).strip()

    casing = _casing_map(source)
    if casing:
        text = _WORD.sub(lambda m: casing.get(m.group(0).lower(), m.group(0)), text)
    return _restore_sentence_capitals(text)
