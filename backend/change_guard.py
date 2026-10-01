"""Rejects output that differs from its input without any gain for the reader.

The seq2seq path already rejects hallucinations (``main._looks_like_hallucination``)
and the prompted path empty/echoed/refused output (``prompting.classify_output``).
Both let through output that differs only by a dropped full stop, a straightened
quote, or a deleted number, which the extension would then highlight as simplified.

Two narrow rules; word-level edits are never rejected here (dropping one word can be a
real simplification, and deletion quality is the offline evaluation's business).

1. **Nothing but punctuation moved.** Word tokens identical, ignoring case,
   punctuation and whitespace. Exception: splitting a run-on into two sentences moves
   no words but does simplify, so a punctuation-only edit is kept when it lowers the
   Flesch-Kincaid grade level by ``MIN_FKGL_GAIN`` or more. FKGL because both
   conditions in this project are scored on it (research/evaluate_*.py, via ``easse``).

2. **Only a number went missing.** Words unchanged, digits gone: "15.43 percent
   inflation" -> "percent inflation" loses a fact.

FKGL here is the plain formula over a vowel-group syllable count (no new dependency).
It only compares two texts measured the same way, so the approximation's bias cancels
out; it is not reported anywhere and is not the number the thesis quotes.
"""

import re
from typing import List, Optional, Tuple

# Letters and digits, so '15.43' becomes '15' + '43' and a dropped number registers
# as dropped words.
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
# Sentence-like boundaries. Units here are usually single sentences; for the rare
# multi-sentence unit it only needs to be consistent between input and output.
_SENTENCE_END_RE = re.compile(r'[.!?]+(?=\s|$)')
_VOWEL_GROUP_RE = re.compile(r'[aeiouy]+')

# Minimum FKGL drop for a punctuation-only edit to be kept (rule 1 above).
MIN_FKGL_GAIN = 0.5

NO_MEANINGFUL_CHANGE = 'no_meaningful_change'


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def count_syllables(word: str) -> int:
    """Vowel groups, less a silent trailing "e", at least one.

    The usual textbook approximation. It miscounts plenty of words in isolation
    ("queue", "fire"), which does not matter here: the same word miscounted the same
    way on both sides of the comparison cancels out.
    """
    if word.isdigit():
        # one syllable, so numbers don't dominate the ratio in the sentences this
        # guard targets
        return 1
    groups = len(_VOWEL_GROUP_RE.findall(word))
    if word.endswith('e') and not word.endswith(('le', 'ee')) and groups > 1:
        groups -= 1
    return max(groups, 1)


def flesch_kincaid_grade(text: str) -> float:
    """US grade level required to read `text`. Lower is simpler."""
    tokens = words(text)
    if not tokens:
        return 0.0
    sentences = max(len(_SENTENCE_END_RE.findall(text)), 1)
    syllables = sum(count_syllables(t) for t in tokens)
    return 0.39 * (len(tokens) / sentences) + 11.8 * (syllables / len(tokens)) - 15.59


def check_change(original: str, simplified: str) -> Optional[str]:
    """Returns a fallback reason if `simplified` isn't a real simplification of
    `original`, or None if it is.

    Output identical to the input returns None, not a reason: nothing was rejected
    there -- the model left the text alone, which is a legitimate answer and is
    already reported as "unchanged" by everything downstream.
    """
    original_tokens = words(original)
    simplified_tokens = words(simplified)

    if not original_tokens or original.strip() == simplified.strip():
        return None

    if original_tokens == simplified_tokens:
        # punctuation/case/spacing only: kept only if it simplifies (a run-on split)
        gain = flesch_kincaid_grade(original) - flesch_kincaid_grade(simplified)
        return None if gain >= MIN_FKGL_GAIN else NO_MEANINGFUL_CHANGE

    if _lost_only_numbers(original_tokens, simplified_tokens):
        return NO_MEANINGFUL_CHANGE

    return None


def _lost_only_numbers(
    original_tokens: List[str], simplified_tokens: List[str]
) -> bool:
    """Every word is still there in the same order, and some of the digits aren't."""
    without_numbers = [t for t in original_tokens if not t.isdigit()]
    return without_numbers == [t for t in simplified_tokens if not t.isdigit()] and len(
        simplified_tokens
    ) < len(original_tokens)


def apply(original: str, simplified: str) -> Tuple[str, Optional[str]]:
    """(text to serve, fallback reason) -- the original text and a reason when the
    edit doesn't survive `check_change`, the model's own output and None when it does.
    """
    reason = check_change(original, simplified)
    return (original, reason) if reason else (simplified, None)
