"""Selects the few-shot demonstrations used by the prompted-LLM method.

Run this to regenerate the ``NON_NATIVE_SPEAKERS`` pool in ``prompting.py``:

    python select_fewshot.py

BLESS (Kew et al., 2023) §3.3 samples its N=3 few-shot examples from the *validation*
split of the dataset being evaluated, and finds LLMs highly sensitive to which examples
are used. So the demonstrations are part of the experimental setup and must be
reproducible, not authored: a stated, deterministic rule set on a fixed seed. No
example was chosen by how well it read.

The rules:

1. **Seed 42, sample the whole validation split, take them in that order.** BLESS's
   random sampling, made reproducible.

2. **Source must be >= MIN_SOURCE_WORDS words.** An unfiltered draw (indices 51, 228,
   1309 at this seed) returned three sentences of 8-16 words whose references only
   substituted words. The instruction names paraphrasing, compression *and* splitting.

3. **Reference must be well-formed**: starts uppercase, ends in terminal punctuation.
   ASSET is crowdsourced and reference quality varies; ``validation[285]``'s first
   reference is lowercase with no final stop, against the instruction's
   "grammatical, fluent".

4. **>= MIN_CONTENT_OVERLAP of the reference's content words must occur in the source.**
   The rule that matters most. ``validation[285]``'s first well-formed reference reads
   "The input box is usually long to allow for better password protection." for a
   source about brute-force attacks on a key space: fluent, and a total meaning
   change, violating the instruction's "the main ideas of its original counterpart
   without altering its meaning". Content-word overlap is a coarse but mechanical
   proxy, and it catches this case.

The chosen indices are printed because they belong in the thesis's methodology
section, so the exact prompt can be reconstructed.
"""

import random
import re
from typing import List, Tuple

from hf_revisions import HF_REVISIONS

SEED = 42
N_EXAMPLES = 3
MIN_SOURCE_WORDS = 20
MIN_CONTENT_OVERLAP = 0.5

# Deliberately a small, explicit list rather than an NLTK dependency: it only has to be
# good enough to stop function words from inflating the overlap score.
STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'been',
    'but',
    'by',
    'for',
    'from',
    'had',
    'has',
    'have',
    'he',
    'her',
    'his',
    'in',
    'is',
    'it',
    'its',
    'of',
    'on',
    'or',
    'she',
    'that',
    'the',
    'their',
    'they',
    'this',
    'to',
    'was',
    'were',
    'which',
    'with',
    'you',
    'your',
    'usually',
    'also',
    'other',
    'not',
    'can',
    'will',
    'would',
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def content_words(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text)} - STOPWORDS


# Crowdworker punctuation typos: a comma immediately before the final stop
# ("mistakes,."), doubled marks, or a space before a comma. ASSET has these; a
# demonstration carrying one teaches the model to reproduce it.
_MALFORMED_PUNCT_RE = re.compile(r'[,;:]\s*[.!?]$|\.\.|,,|\s,')


def is_well_formed(text: str) -> bool:
    stripped = text.strip()
    if not stripped or not stripped[0].isupper() or stripped[-1] not in '.!?':
        return False
    return not _MALFORMED_PUNCT_RE.search(stripped)


def retains_meaning(source: str, reference: str) -> bool:
    """Fraction of the *reference's* content words that also appear in the source.

    Deliberately measured on the reference, not the source: a simplification is
    allowed to drop source material (that's compression) but inventing content the
    source never mentioned is what we're screening out.
    """
    ref_words = content_words(reference)
    if not ref_words:
        return False
    return (
        len(ref_words & content_words(source)) / len(ref_words) >= MIN_CONTENT_OVERLAP
    )


def select(split: str = 'validation') -> List[Tuple[int, str, str]]:
    from datasets import load_dataset

    ds = load_dataset(
        'facebook/asset',
        'simplification',
        split=split,
        revision=HF_REVISIONS.get('facebook/asset'),
    )
    order = random.Random(SEED).sample(range(len(ds)), len(ds))

    chosen: List[Tuple[int, str, str]] = []
    for i in order:
        row = ds[i]
        source = row['original']
        if len(source.split()) < MIN_SOURCE_WORDS:
            continue
        reference = next(
            (
                r
                for r in row['simplifications']
                if is_well_formed(r) and retains_meaning(source, r)
            ),
            None,
        )
        if reference is None:
            continue
        chosen.append((i, source, reference.strip()))
        if len(chosen) == N_EXAMPLES:
            break
    return chosen


if __name__ == '__main__':
    picked = select()
    print(f'# ASSET validation, seed={SEED}, indices {[i for i, _, _ in picked]}')
    print('(\n')
    for i, source, reference in picked:
        print(f'    # ASSET validation[{i}]')
        print('    Example(')
        print(f'        complex={source!r},')
        print(f'        simple={reference!r},')
        print('    ),')
    print(')')
