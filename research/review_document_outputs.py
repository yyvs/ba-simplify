"""Qualitative review of the Step 4B document-level generations.

    python review_document_outputs.py                        # the n=500 seed-42 run
    python review_document_outputs.py --generations path.json --limit 20

evaluate_document.py answers "is the fine-tuned model better?" with one number per
metric. It does not answer "better how, and wrong how?", and the generations it cached
(results/document_eval/generations_n500_seed42.json) sat unread. This script reads them
back and produces the counts and the worked examples Ch. 6.3 needs, so the qualitative
section rests on the full sample rather than on whichever document happened to be
scrolled past.

Two design points, both about not overclaiming:

- **Every category here is a surface heuristic, not a judgement.** "A year in the output
  that is absent from the source" is machine-checkable; "a hallucination" is not. The
  script reports the mechanical property and stores the examples so the property can be
  read by a human before being called an error. Where the write-up calls something a
  factual error, that is a manual reading of an example this script surfaced, not the
  script's own verdict.
- **The sample is re-derived, not assumed.** load_dwikipedia_test() is imported from
  evaluate_document.py with the same seed and limit, and the resulting source list is
  fingerprinted against the sources_sha the generation run recorded. A mismatch aborts:
  pairing predictions against the wrong documents would produce confident nonsense.
"""

import argparse
import json
import logging
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from evaluate_document import _sha16, load_dwikipedia_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# D-Wikipedia is lowercased and moses-tokenized (2.3), so a plain word regex suffices.
_WORD = re.compile(r"[a-z0-9']+")
_SENT = re.compile(r'(?<=[.!?])\s+')
_YEAR = re.compile(r'\b(1[5-9]\d\d|20[0-2]\d)\b')
_NUMBER = re.compile(r'\d[\d,.]*')


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def sentences(text: str) -> List[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def ratio(numerator: Sequence, denominator: Sequence) -> float:
    return len(numerator) / max(1, len(denominator))


def describe_system(
    sources: List[str], refs: List[str], preds: List[str]
) -> Dict[str, object]:
    """Corpus-level shape of one system's output, relative to source and reference."""
    length_ratios = [
        ratio(words(p), words(s)) for p, s in zip(preds, sources, strict=True)
    ]
    sentence_ratios = [
        ratio(sentences(p), sentences(s)) for p, s in zip(preds, sources, strict=True)
    ]
    words_per_sentence = [ratio(words(p), sentences(p)) for p in preds]

    identical = sum(
        1 for p, s in zip(preds, sources, strict=True) if p.strip() == s.strip()
    )
    # "Copied" is deliberately stricter than "identical": returning the source with one
    # comma moved is not simplifying either (22 vs. 84 is a materially different claim
    # about how often the model declines to act).
    copied = 0
    for pred, source in zip(preds, sources, strict=True):
        pw, sw = words(pred), set(words(source))
        if (
            pw
            and sum(1 for t in pw if t in sw) / len(pw) > 0.98
            and ratio(pw, words(source)) > 0.9
        ):
            copied += 1

    split = sum(
        1
        for p, s in zip(preds, sources, strict=True)
        if len(sentences(p)) > len(sentences(s))
    )
    shorter_than_ref = sum(
        1
        for p, r in zip(preds, refs, strict=True)
        if len(words(p)) < 0.5 * len(words(r))
    )
    new_years = sum(
        1
        for p, s in zip(preds, sources, strict=True)
        if set(_YEAR.findall(p)) - set(_YEAR.findall(s))
    )
    new_numbers = sum(
        1
        for p, s in zip(preds, sources, strict=True)
        if set(_NUMBER.findall(p)) - set(_NUMBER.findall(s))
    )

    # A 5-gram three times in one document is the classic seq2seq degeneration loop;
    # checks that no_repeat_ngram_size=3 prevented it.
    looping = 0
    for pred in preds:
        tokens = words(pred)
        counts = Counter(tuple(tokens[i : i + 5]) for i in range(len(tokens) - 4))
        if counts and max(counts.values()) >= 3:
            looping += 1

    return {
        'length_ratio_vs_source_median': round(statistics.median(length_ratios), 3),
        'sentence_ratio_vs_source_median': round(statistics.median(sentence_ratios), 3),
        'words_per_sentence_median': round(statistics.median(words_per_sentence), 2),
        'identical_to_source': identical,
        'near_copy_of_source': copied,
        'split_a_sentence': split,
        'under_half_reference_length': shorter_than_ref,
        'introduces_absent_year': new_years,
        'introduces_absent_number': new_numbers,
        'repetition_loops': looping,
        'empty': sum(1 for p in preds if not p.strip()),
    }


def length_buckets(
    sources: List[str], refs: List[str], preds: List[str]
) -> List[Dict[str, object]]:
    """Compression against source length.

    The single headline "median length ratio" hides the finding: the model barely edits
    a 40-word stub and cuts a 400-word article to a quarter. Whether that tracks the
    reference is the question, so both are reported per bucket.
    """
    edges = [(0, 60), (60, 120), (120, 250), (250, 10**6)]
    rows = []
    for low, high in edges:
        idx = [i for i, s in enumerate(sources) if low <= len(words(s)) < high]
        if not idx:
            continue
        rows.append(
            {
                'source_words': f"{low}-{high if high < 10 ** 6 else '+'}",
                'n': len(idx),
                'prediction_length_ratio_median': round(
                    statistics.median(
                        [ratio(words(preds[i]), words(sources[i])) for i in idx]
                    ),
                    3,
                ),
                'reference_length_ratio_median': round(
                    statistics.median(
                        [ratio(words(refs[i]), words(sources[i])) for i in idx]
                    ),
                    3,
                ),
            }
        )
    return rows


def census_year_probe(sources, refs, preds, baseline) -> Dict[str, object]:
    """The one error class specific enough to count exactly rather than approximate.

    US settlement stubs are a large, formulaic slice of Wikipedia, and the source
    documents here consistently cite the 2010 census. If the fine-tuned model rewrites
    that year while the references keep it, the model is reciting its training corpus
    over its input -- which is a different and more worrying failure than paraphrasing
    badly, because the output stays fluent and plausible.

    The says/keeps counts overlap: an output that copies a source mentioning both
    censuses contains both strings. The substitution is 2000 without 2010, and the
    three finetuned_* outcomes that exclude each other are that, keeps, and neither.
    """
    idx = [i for i, s in enumerate(sources) if '2010 census' in s]
    return {
        'sources_saying_2010_census': len(idx),
        'finetuned_says_2000_census': sum(1 for i in idx if '2000 census' in preds[i]),
        'finetuned_keeps_2010_census': sum(1 for i in idx if '2010 census' in preds[i]),
        'finetuned_says_2000_not_2010': sum(
            1 for i in idx if '2000 census' in preds[i] and '2010 census' not in preds[i]
        ),
        'finetuned_says_neither': sum(
            1
            for i in idx
            if '2000 census' not in preds[i] and '2010 census' not in preds[i]
        ),
        'baseline_says_2000_census': sum(
            1 for i in idx if '2000 census' in baseline[i]
        ),
        'references_saying_2010_census': sum(
            1 for i in idx if '2010 census' in refs[i]
        ),
        'references_saying_2000_census': sum(
            1 for i in idx if '2000 census' in refs[i]
        ),
        'example_indices': idx[:20],
    }


# Disambiguation pages list what a title could mean: not prose, nothing to simplify.
# D-Wikipedia stores them as a source ending in one of these phrases plus a colon.
_DISAMBIGUATION = re.compile(
    r'(may (also )?refer to|may (also )?mean|might (also )?mean|is the name of'
    r'|can (also )?refer to|can mean)\s*:\s*$'
)


def corpus_pairing_probe(sources, refs) -> Dict[str, object]:
    """How often is the corpus's "simplification" not a simplification of its source?

    Added 2026-08-24, after an instance found while reading the n=500 sample turned out
    to be a class. D-Wikipedia is aligned automatically and admits pairs where the target
    is an *expansion*: a disambiguation stub ("gig or gig may refer to :") paired with a
    full article about one of the meanings. No correct output exists for these: D-SARI
    charges the model both for the deletion it did not make and for the addition it
    could not have invented.

    Two counts: disambiguation pages in the split, and *mispaired* ones (the evaluation
    problem). "Expanded" is deliberately conservative (reference >= 20 words and >= 3x
    the source), so a stub paired with another stub is not counted.
    """
    idx = [i for i, s in enumerate(sources) if _DISAMBIGUATION.search(s.strip())]
    expanded = [
        i
        for i in idx
        if len(words(refs[i])) >= 20
        and len(words(refs[i])) >= 3 * max(len(words(sources[i])), 1)
    ]
    return {
        'disambiguation_stub_sources': len(idx),
        'share_of_split_pct': round(100.0 * len(idx) / max(len(sources), 1), 2),
        'reference_is_an_expansion': len(expanded),
        'mispaired_share_of_split_pct': round(
            100.0 * len(expanded) / max(len(sources), 1), 2
        ),
        'expansion_threshold': 'reference >= 20 words and >= 3x the source',
        'median_source_words': (
            statistics.median([len(words(sources[i])) for i in expanded])
            if expanded
            else None
        ),
        'median_reference_words': (
            statistics.median([len(words(refs[i])) for i in expanded])
            if expanded
            else None
        ),
        'example_indices': expanded[:20],
    }


def collect_examples(
    sources, refs, preds, baseline, per_category: int
) -> Dict[str, List[Dict]]:
    """Worked examples per category, so each count in the summary is inspectable."""

    def record(i: int) -> Dict[str, object]:
        return {
            'index': i,
            'source_words': len(words(sources[i])),
            'source': sources[i],
            'reference': refs[i],
            'finetuned': preds[i],
            'baseline': baseline[i],
        }

    changed_year = [
        i
        for i, (s, p) in enumerate(zip(sources, preds, strict=True))
        if set(_YEAR.findall(p)) - set(_YEAR.findall(s))
    ]
    unchanged = [
        i
        for i, (s, p) in enumerate(zip(sources, preds, strict=True))
        if s.strip() == p.strip()
    ]
    over_deleted = [
        i
        for i, (r, p) in enumerate(zip(refs, preds, strict=True))
        if len(words(p)) < 0.5 * len(words(r)) and len(words(sources[i])) > 150
    ]
    novel_content = []
    for i, (s, p) in enumerate(zip(sources, preds, strict=True)):
        source_vocab = set(words(s))
        content = [t for t in words(p) if len(t) > 3]
        if (
            content
            and sum(1 for t in content if t not in source_vocab) / len(content) > 0.4
        ):
            novel_content.append(i)

    return {
        'introduced_year': [record(i) for i in changed_year[:per_category]],
        'returned_source_unchanged': [record(i) for i in unchanged[:per_category]],
        'compressed_far_below_reference': [
            record(i) for i in over_deleted[:per_category]
        ],
        'high_novel_vocabulary': [record(i) for i in novel_content[:per_category]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data-dir', default='scratch/d_wikipedia_raw')
    parser.add_argument(
        '--generations', default='results/document_eval/generations_n500_seed42.json'
    )
    parser.add_argument(
        '--partial',
        default='results/document_eval/partial_finetuned_n500_seed42.json',
        help='run whose sources_sha the re-derived sample is checked against',
    )
    parser.add_argument(
        '--limit', type=int, default=500, help='must match the generation run'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='must match the generation run'
    )
    parser.add_argument(
        '--examples', type=int, default=6, help='worked examples stored per category'
    )
    parser.add_argument(
        '--out', default='results/document_eval/qualitative_review_n500_seed42.json'
    )
    args = parser.parse_args()

    sources, references, sampling = load_dwikipedia_test(
        args.data_dir, args.limit or None, args.seed
    )
    refs = [r[0] for r in references]

    generations = json.loads(Path(args.generations).read_text())
    finetuned = generations['finetuned_predictions']
    baseline = generations['baseline_predictions']
    if not len(sources) == len(finetuned) == len(baseline):
        raise SystemExit(
            f'length mismatch: {len(sources)} sources vs {len(finetuned)}/{len(baseline)} predictions'
        )

    sources_sha = _sha16('\n'.join(sources))
    expected = json.loads(Path(args.partial).read_text()).get('sources_sha')
    if expected and expected != sources_sha:
        raise SystemExit(
            f'sample mismatch: re-derived {sources_sha}, generation run used {expected}. '
            'The predictions belong to a different sample; refusing to pair them.'
        )
    logger.info(
        'Sample verified against the generation run (sources_sha=%s)', sources_sha
    )

    report = {
        'run': {
            'script': 'review_document_outputs.py',
            'script_version': '1.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'generations_file': args.generations,
            'sources_sha': sources_sha,
        },
        'dataset': sampling | {'n_reviewed': len(sources)},
        'corpus_shape': {
            'source': {
                'words_median': statistics.median([len(words(s)) for s in sources]),
                'words_mean': round(
                    statistics.mean([len(words(s)) for s in sources]), 1
                ),
                'sentences_median': statistics.median(
                    [len(sentences(s)) for s in sources]
                ),
                'words_per_sentence_median': round(
                    statistics.median([ratio(words(s), sentences(s)) for s in sources]),
                    2,
                ),
            },
            'reference': {
                'words_median': statistics.median([len(words(r)) for r in refs]),
                'words_mean': round(statistics.mean([len(words(r)) for r in refs]), 1),
                'words_per_sentence_median': round(
                    statistics.median([ratio(words(r), sentences(r)) for r in refs]), 2
                ),
            },
        },
        'systems': {
            'finetuned': describe_system(sources, refs, finetuned),
            'baseline': describe_system(sources, refs, baseline),
            'reference': describe_system(sources, refs, refs),
        },
        'compression_by_source_length': {
            'finetuned': length_buckets(sources, refs, finetuned),
            'baseline': length_buckets(sources, refs, baseline),
        },
        'census_year_probe': census_year_probe(sources, refs, finetuned, baseline),
        'corpus_pairing_probe': corpus_pairing_probe(sources, refs),
        'examples': collect_examples(sources, refs, finetuned, baseline, args.examples),
        'notes': [
            'Categories are surface heuristics over the cached generations, not human judgements.',
            "'near_copy_of_source' means >98% of output tokens occur in the source and length is "
            'within 10% of it -- i.e. the system effectively declined to edit.',
            'Reference rows are the human simplifications scored as if they were a system, to give '
            'each count a ceiling to be read against.',
            'corpus_pairing_probe counts documents whose reference is not a simplification of '
            'their source at all -- a corpus-quality finding, not a model result.',
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info('Wrote %s', out)

    fine = report['systems']['finetuned']
    logger.info(
        'finetuned: len-ratio %.2f | unchanged %d | near-copy %d | split %d | '
        'below-half-reference %d | new year %d',
        fine['length_ratio_vs_source_median'],
        fine['identical_to_source'],
        fine['near_copy_of_source'],
        fine['split_a_sentence'],
        fine['under_half_reference_length'],
        fine['introduces_absent_year'],
    )
    logger.info('census probe: %s', report['census_year_probe'])


if __name__ == '__main__':
    main()
