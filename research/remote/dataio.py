"""Canonical corpus loading and cleaning for the GPU runs -- one implementation, and a
report on how it differs from the two it replaces.

**Why this file exists, since adding a fourth copy of cleaning logic would be the wrong
move.** The project currently has three, and they disagree:

  1. `research/prepare_data.py`         -- drops any row containing a bare "â" or "Ã".
  2. the notebook's `contains_mojibake` -- drops rows matching seven specific regexes.
  3. `research/fine_tune.py`            -- copy of (1), and dead: it reads columns
                                           "complex"/"simple" from a dataset whose columns
                                           are "source"/"target", so it raises KeyError on
                                           the first row and cannot have produced a result.

That matters more than tidiness. (1) produced the tracked `data/stats.json` and the data
statement -- the "123,862 raw -> 117,656 kept, 5.0% dropped" figures that the thesis
§4.3 and §6.1 report. (2) is what actually ran during training. They disagree on ~4,000
rows, so **the documented corpus is not the corpus the model was trained on**, and the
`"Ã "` class that (2) misses is real mojibake rather than a false positive.

This module implements both filters explicitly, makes the choice a recorded flag rather
than a side effect of which file you happened to run, and `--report` writes an artifact
quantifying the divergence. That converts a silent inconsistency into a measured, citable
one, which is the only version of it that can go in a thesis.

Default is `strict` (filter 1). Two reasons, in order: it removes the mojibake class the
targeted filter misses, and it is the filter the published data statement already
describes -- so choosing it makes an existing thesis claim true instead of requiring the
numbers in §4.3/§6.1 to be rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from common import GPU_RESULTS, SCRATCH, write_artifact

# hf_revisions lives in research/; `python remote/x.py` only puts remote/ on sys.path
# (same shim as lens_only.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hf_revisions import HF_REVISIONS  # noqa: E402

# whitespace
_WS_RE = re.compile(r'\s+')


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip.

    Note for §4.3, which says this matches backend `extract_and_clean()`: that function no
    longer collapses whitespace (backend/main.py:180, it keeps newlines/tabs for sentence
    layout). Collapsing now happens client-side in content.js's chunk walk, conditionally
    per call site.
    """
    if not text:
        return ''
    return _WS_RE.sub(' ', text).strip()


# mojibake filters
STRICT_MARKERS = ('â', 'Ã')

TARGETED_PATTERNS = [
    r"â\s*''",  # broken en-dash or punctuation
    r'â\s*€\s*™',  # broken single curly quote
    r'Ã\s*©',  # broken e-acute
    r'Ã\s*¹',  # broken u-grave
    r'Ã\s*¶',  # broken o-umlaut
    r'â\s*€\s*œ',  # broken opening double quote
    r'â\s*€\s*',  # broken closing double quote
]
_TARGETED_RE = [re.compile(p) for p in TARGETED_PATTERNS]

FILTERS = ('strict', 'targeted', 'none')


def contains_mojibake(text: str, mode: str = 'strict') -> bool:
    if not text or mode == 'none':
        return False
    if mode == 'strict':
        return any(marker in text for marker in STRICT_MARKERS)
    if mode == 'targeted':
        return any(rx.search(text) for rx in _TARGETED_RE)
    raise ValueError(f'unknown filter mode {mode!r}; expected one of {FILTERS}')


def clean_and_filter_split(
    sources: Sequence[str], targets: Sequence[str], split: str, mode: str = 'strict'
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Collapse whitespace, drop mojibake rows, and return the counts alongside the data.

    Counts are returned so the run manifest records the corpus actually trained on.
    """
    keep_src: List[str] = []
    keep_tgt: List[str] = []
    dropped = 0
    for src, tgt in zip(sources, targets, strict=True):
        if contains_mojibake(src, mode) or contains_mojibake(tgt, mode):
            dropped += 1
            continue
        keep_src.append(collapse_whitespace(src))
        keep_tgt.append(collapse_whitespace(tgt))
    stats = {
        'raw_rows': len(sources),
        'dropped_mojibake_rows': dropped,
        'kept_rows': len(keep_src),
    }
    print(
        f"  [{split}] {stats['raw_rows']} raw -> {stats['kept_rows']} kept "
        f'({dropped} dropped, filter={mode})'
    )
    return keep_src, keep_tgt, stats


# sentence corpus: WikiLarge
WIKILARGE_ID = 'eilamc14/wikilarge-clean'

SENTENCE_SLICES = {
    'prove_loop': ('train[:200]', 'validation[:50]', 'test[:20]'),
    'reduced': ('train[:20000]', 'validation[:200]', 'test[:50]'),
    'full': ('train', 'validation', 'test'),
}


def load_wikilarge(scope: str = 'full', mode: str = 'strict') -> Dict[str, Dict]:
    from datasets import load_dataset

    train_split, val_split, test_split = SENTENCE_SLICES[scope]
    out: Dict[str, Dict] = {'stats': {}}
    for name, split in (
        ('train', train_split),
        ('validation', val_split),
        ('test', test_split),
    ):
        raw = load_dataset(
            WIKILARGE_ID, split=split, revision=HF_REVISIONS.get(WIKILARGE_ID)
        )
        src, tgt, stats = clean_and_filter_split(
            raw['source'], raw['target'], name, mode
        )
        out[name] = {'source': src, 'target': tgt}
        out['stats'][name] = stats
    return out


# document corpus: D-Wikipedia
D_WIKI_BASE = 'https://raw.githubusercontent.com/RLSNLP/Document-level-text-simplification/main/Dataset'
D_WIKI_CACHE = SCRATCH / 'd_wikipedia_raw'

# Documents per split by scope. At "full", train and test are uncapped (test is generated
# over once, at the end). Validation stays capped: 300 in Colab (generate() on every eval
# vs. a 5h20m session limit); 1000 here, affordable because checkpoint selection uses
# validation loss with generation off (train.py --generate-during-eval).
DOCUMENT_CAPS = {
    'prove_loop': {'train': 200, 'validation': 50, 'test': 20},
    'reduced': {'train': 20000, 'validation': 200, 'test': None},
    'full': {'train': None, 'validation': 1000, 'test': None},
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  downloading {url} -> {dest}')
    urllib.request.urlretrieve(url, dest)


def _ensure_split_files(split: str, cache: Path) -> Tuple[Path, Path]:
    """Fetch one D-Wikipedia split, extracting the 7z-compressed train split if needed."""
    stem = 'valid' if split == 'validation' else split
    src, tgt = cache / f'{stem}.src', cache / f'{stem}.tgt'
    if src.exists() and tgt.exists():
        return src, tgt

    if stem == 'train':
        # only train is distributed 7z-compressed
        import py7zr

        for name in ('train.src.7z', 'train.tgt.7z'):
            archive = cache / name
            _download(f'{D_WIKI_BASE}/{name}', archive)
            print(f'  extracting {archive}')
            with py7zr.SevenZipFile(archive, mode='r') as zf:
                zf.extractall(path=str(cache))
    else:
        _download(f'{D_WIKI_BASE}/{stem}.src', src)
        _download(f'{D_WIKI_BASE}/{stem}.tgt', tgt)
    return src, tgt


def load_d_wikipedia(scope: str = 'full', mode: str = 'strict') -> Dict[str, Dict]:
    """D-Wikipedia document pairs, cleaned with the same filter as the sentence corpus.

    The corpus is lowercased and PTB-pre-tokenized (verified: no uppercase in 500 sampled
    lines, no newlines inside any of 3,000 document bodies), hence the normalize/
    de-normalize pair in backend/document_text.py and the README's natural-case stretch stage.
    """
    caps = DOCUMENT_CAPS[scope]
    out: Dict[str, Dict] = {'stats': {}}
    for split in ('train', 'validation', 'test'):
        src_path, tgt_path = _ensure_split_files(split, D_WIKI_CACHE)
        sources = src_path.read_text(encoding='utf-8').splitlines()
        targets = tgt_path.read_text(encoding='utf-8').splitlines()
        if len(sources) != len(targets):
            raise SystemExit(
                f'D-Wikipedia {split}: {len(sources)} sources vs {len(targets)} targets. '
                f'The release is line-aligned, so a mismatch means a truncated download -- '
                f'delete {D_WIKI_CACHE} and rerun.'
            )
        cap = caps[split]
        if cap is not None:
            sources, targets = sources[:cap], targets[:cap]
        src, tgt, stats = clean_and_filter_split(sources, targets, split, mode)
        out[split] = {'source': src, 'target': tgt}
        out['stats'][split] = stats
    return out


def load_corpus(
    structure: str, scope: str = 'full', mode: str = 'strict'
) -> Dict[str, Dict]:
    if structure == 'sentence':
        return load_wikilarge(scope, mode)
    if structure == 'document':
        return load_d_wikipedia(scope, mode)
    raise ValueError(f'unknown structure {structure!r}')


# --report: quantify the filter divergence
def filter_report() -> Dict:
    """Apply both filters to the raw WikiLarge splits and record where they disagree.

    The artifact behind the §4.3 correction on which filter is canonical.
    """
    from datasets import load_dataset

    report: Dict = {'dataset': WIKILARGE_ID, 'splits': {}}
    for split in ('train', 'validation', 'test'):
        raw = load_dataset(
            WIKILARGE_ID, split=split, revision=HF_REVISIONS.get(WIKILARGE_ID)
        )
        sources, targets = raw['source'], raw['target']
        strict_only, targeted_only = [], []
        n_strict = n_targeted = n_both = 0
        for src, tgt in zip(sources, targets, strict=True):
            s = contains_mojibake(src, 'strict') or contains_mojibake(tgt, 'strict')
            t = contains_mojibake(src, 'targeted') or contains_mojibake(tgt, 'targeted')
            n_strict += s
            n_targeted += t
            n_both += s and t
            if s and not t and len(strict_only) < 12:
                strict_only.append(src[:200])
            if t and not s and len(targeted_only) < 12:
                targeted_only.append(src[:200])
        report['splits'][split] = {
            'raw_rows': len(sources),
            'dropped_strict': n_strict,
            'dropped_targeted': n_targeted,
            'dropped_by_both': n_both,
            'dropped_only_by_strict': n_strict - n_both,
            'dropped_only_by_targeted': n_targeted - n_both,
            'kept_strict': len(sources) - n_strict,
            'kept_targeted': len(sources) - n_targeted,
            'examples_dropped_only_by_strict': strict_only,
            'examples_dropped_only_by_targeted': targeted_only,
        }
    report['interpretation'] = (
        "prepare_data.py's strict filter produced the tracked data/stats.json and the data "
        "statement; the notebook's targeted filter is what ran during training. Rows dropped "
        "only by the strict filter are dominated by the 'Ã ' class (e.g. 'AmbÃ rieux-en-Dombes', "
        "'VendÃ e', 'PÃ ter BartÃ k'), which is genuine mojibake the targeted patterns do not "
        'match -- so the targeted filter under-drops rather than the strict filter '
        'over-dropping. Cite whichever filter the reported run used, and say so.'
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--report',
        action='store_true',
        help='write the filter-divergence artifact and exit',
    )
    ap.add_argument('--structure', choices=['sentence', 'document'], default='document')
    ap.add_argument(
        '--scope', choices=['prove_loop', 'reduced', 'full'], default='prove_loop'
    )
    ap.add_argument('--filter', dest='mode', choices=list(FILTERS), default='strict')
    args = ap.parse_args()

    if args.report:
        write_artifact(GPU_RESULTS / 'corpus_filter_divergence.json', filter_report())
        return

    data = load_corpus(args.structure, args.scope, args.mode)
    print(json.dumps(data['stats'], indent=2))


if __name__ == '__main__':
    main()
