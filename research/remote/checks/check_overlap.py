"""Measure how much of each evaluation set appears in the training data.

**Why this is not optional.** ASSET (Alva-Manchego et al., 2020) was built on TurkCorpus,
and TurkCorpus's source sentences were drawn from WikiLarge. This project trains on a
third-party WikiLarge mirror (`eilamc14/wikilarge-clean`), so nothing in the pipeline
guarantees that the sentences it is evaluated on were held out of what it learned from.
Anyone who works on sentence simplification knows that lineage. Reporting the overlap is a
threats-to-validity paragraph; being asked about it at a defence with no number to give is
something else.

Measured on the current data (this script's own output):

    ASSET test       ∩ WikiLarge train        5 / 359    (1.4%)
    ASSET validation ∩ WikiLarge train       42 / 1999   (2.1%)
    WikiLarge val    ∩ WikiLarge train       71 / 397    (17.9%)
    WikiLarge test   ∩ WikiLarge train        2 / 121     (1.7%)
    WikiLarge test   ⊂ ASSET test           121 / 121   (nested, not independent)
    few-shot demos   ∩ WikiLarge train        0 / 3      (clean)

Three of those matter, in descending order:

  - **17.9% of the WikiLarge validation split is in train.** That split is what
    `metric_for_best_model="loss"` selects checkpoints against, so absolute validation losses
    (the 0.4568 / 0.4590 pair quoted in §4.5 and RESULTS.md) are optimistically biased. The
    *relative* comparison between two checkpoints is not, since both face the same
    contaminated set -- so the selection is probably still right, but the numbers should not
    be presented as clean held-out losses.
  - **1.4% of the ASSET test split is in train.** Small enough not to move SARI +16.4
    meaningfully, and `--rescore-clean` quantifies exactly how little; large enough that it
    has to be stated.
  - **Every WikiLarge test source is in ASSET test.** Not a problem -- it is the useful
    explanation for why the S1/S2 (WikiLarge test) and S3-S5 (ASSET test) results move in the
    same direction. Nested sets, not independent replications, and worth saying so.

Matching is exact after Unicode NFKC normalisation, lowercasing, and collapsing everything
that is not alphanumeric to single spaces. That deliberately ignores the tokenisation and
punctuation differences between the corpora (WikiLarge is space-tokenised, ASSET is not) so
the same sentence is recognised across both, and it deliberately does *not* do fuzzy or
near-duplicate matching -- a near-duplicate is a different and much harder claim, and an
inflated overlap number would be as misleading as a missing one. The figures here are
therefore a floor, and the script says so in its own output.

    python remote/checks/check_overlap.py                # from research/
    python remote/checks/check_overlap.py --near-dupes    # add a token-overlap estimate
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import GPU_RESULTS, write_artifact  # noqa: E402
from dataio import WIKILARGE_ID, load_wikilarge  # noqa: E402

_NON_ALNUM = re.compile(r'[^a-z0-9]+')

# select_fewshot.py's demonstration indices (ASSET validation, seed 42); a demo seen in
# training would leak into the prompted conditions too.
FEWSHOT_INDICES = (285, 1516, 1116)


def norm(text: str) -> str:
    return _NON_ALNUM.sub(' ', unicodedata.normalize('NFKC', text).lower()).strip()


def norm_set(texts: Sequence[str]) -> Set[str]:
    return {norm(t) for t in texts}


def overlap(
    name_a: str, a: Sequence[str], name_b: str, b: Set[str], examples: int = 5
) -> Dict:
    """Exact-after-normalisation overlap of `a` against the normalised set `b`.

    Records every overlapping index (positions in `a` as loaded, so only meaningful with
    `left`), which a held-out-remainder rescore needs without re-implementing `norm`.
    """
    normalized = [norm(t) for t in a]
    matches = [
        (i, orig)
        for i, (orig, n) in enumerate(zip(a, normalized, strict=True))
        if n in b
    ]
    unique = len(set(normalized))
    return {
        'left': name_a,
        'right': name_b,
        'left_items': len(a),
        'left_unique_normalized': unique,
        'overlapping_items': len(matches),
        'overlap_pct_of_left': round(100.0 * len(matches) / max(len(a), 1), 2),
        'overlapping_indices': [i for i, _ in matches],
        'examples': [orig[:220] for _, orig in matches[:examples]],
    }


def token_jaccard_estimate(
    a: Sequence[str], b: Sequence[str], threshold: float = 0.9, cap: int = 400
) -> Dict:
    """Rough near-duplicate estimate on a capped sample -- O(n*m), so it is a probe, not a census.

    Never merged into the exact figures: near-duplication is a judgement call.
    """
    a_tokens = [set(norm(t).split()) for t in a[:cap]]
    b_tokens = [set(norm(t).split()) for t in b]
    near = 0
    for ta in a_tokens:
        if not ta:
            continue
        for tb in b_tokens:
            if not tb:
                continue
            inter = len(ta & tb)
            if inter and inter / len(ta | tb) >= threshold:
                near += 1
                break
    return {
        'sampled_left_items': len(a_tokens),
        'threshold_jaccard': threshold,
        'near_duplicate_items': near,
        'note': 'capped sample; an estimate of near-duplication, not an exact count',
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--near-dupes', action='store_true', help='add the token-overlap estimate'
    )
    ap.add_argument(
        '--filter',
        dest='mode',
        default='strict',
        help='mojibake filter used to build the train set being audited',
    )
    args = ap.parse_args()

    from datasets import load_dataset

    print('loading corpora')
    # audited as cleaned: what the model saw
    wl = load_wikilarge(scope='full', mode=args.mode)
    train_sources = wl['train']['source']
    train_set = norm_set(train_sources)

    asset_test = load_dataset('facebook/asset', 'simplification', split='test')
    asset_val = load_dataset('facebook/asset', 'simplification', split='validation')
    asset_test_sources: List[str] = list(asset_test['original'])
    asset_val_sources: List[str] = list(asset_val['original'])

    comparisons = [
        overlap(
            'asset_test.original',
            asset_test_sources,
            'wikilarge_train.source',
            train_set,
        ),
        overlap(
            'asset_validation.original',
            asset_val_sources,
            'wikilarge_train.source',
            train_set,
        ),
        overlap(
            'wikilarge_validation.source',
            wl['validation']['source'],
            'wikilarge_train.source',
            train_set,
        ),
        overlap(
            'wikilarge_test.source',
            wl['test']['source'],
            'wikilarge_train.source',
            train_set,
        ),
        overlap(
            'wikilarge_test.source',
            wl['test']['source'],
            'asset_test.original',
            norm_set(asset_test_sources),
        ),
    ]

    fewshot = []
    for idx in FEWSHOT_INDICES:
        source = asset_val_sources[idx]
        fewshot.append(
            {
                'asset_validation_index': idx,
                'in_wikilarge_train': norm(source) in train_set,
                'source': source[:200],
            }
        )

    payload: Dict = {
        'method': {
            'matching': 'exact after NFKC normalisation, lowercasing, and collapsing '
            'non-alphanumeric runs to single spaces',
            'rationale': 'ignores the tokenisation/punctuation difference between the '
            'corpora so the same sentence is recognised in both; no fuzzy '
            'matching, so every figure is a floor',
            'train_set': f'{WIKILARGE_ID} train, cleaned with the {args.mode!r} mojibake filter',
            'train_items': len(train_sources),
            'train_unique_normalized': len(train_set),
        },
        'comparisons': comparisons,
        'fewshot_demonstrations': fewshot,
        'interpretation': {
            'checkpoint_selection': (
                'The WikiLarge validation overlap is the one that affects a reported '
                "quantity: metric_for_best_model='loss' selects against that split, so "
                'absolute validation losses are optimistically biased. Relative comparison '
                'between checkpoints is unaffected -- every checkpoint faces the same set.'
            ),
            'asset_test': (
                'Small enough not to move SARI materially; state it and, if you want the '
                'number, rerun evaluate_sentence.py on the held-out remainder and report '
                'the delta.'
            ),
            'nesting': (
                'WikiLarge test is a subset of ASSET test, so S1/S2 and S3-S5 are nested '
                'rather than independent evaluations. This is the explanation for the '
                'direction replicating across them, and belongs in §6.6 as such.'
            ),
        },
    }

    if args.near_dupes:
        payload['near_duplicate_probe'] = {
            'asset_test_vs_wikilarge_train': token_jaccard_estimate(
                asset_test_sources, train_sources
            )
        }

    print()
    for c in comparisons:
        print(
            f"  {c['left']:<32} ∩ {c['right']:<28} "
            f"{c['overlapping_items']:>5} / {c['left_items']:<5} ({c['overlap_pct_of_left']}%)"
        )
    leaked = [f['asset_validation_index'] for f in fewshot if f['in_wikilarge_train']]
    print(f"  few-shot demonstrations in train: {leaked or 'none'}")

    # Filter mode in the filename, or the two runs §2e cites overwrite each other. `strict`
    # keeps the original name because that path is cited.
    name = (
        'overlap_audit.json'
        if args.mode == 'strict'
        else f'overlap_audit_{args.mode}.json'
    )
    write_artifact(GPU_RESULTS / name, payload)


if __name__ == '__main__':
    main()
