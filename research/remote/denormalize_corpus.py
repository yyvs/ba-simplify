"""Turn D-Wikipedia's lowercased, PTB-pre-tokenised text back into natural-case prose.

For the natural-case retraining experiment (§7.1's highest-value follow-up): the
normalize/de-normalize pair (§5.6) works around a train/inference mismatch, and §6.4
Finding 3 measured that mixed-case input makes the current document checkpoint
hallucinate ("northern Netherlands" -> "northern hemisphere").

Reuses the backend's own de-normaliser (research may import from the service, never the
reverse), so the transformation is exactly the serving path's.

Limits, to state if this run is reported:
  - Casing is recovered from the request's source text. Here that is the paired source
    document, itself lowercased, so a proper noun only in the target stays lowercase.
    Natural-*ish* case, not the original Wikipedia text.
  - The `u.s.` defect in `normalize_for_model` does not affect this script (only the
    de-normalising half runs), but it corrupts the serving path and metric normalisation;
    see preflight.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# research may import from the service; never the reverse.
BACKEND = Path(__file__).resolve().parent.parent.parent / 'backend'
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from document_text import denormalize_from_model  # noqa: E402


def denormalize_pair(source: str, target: str) -> tuple[str, str]:
    """De-normalise one (source, target) document pair.

    Source against itself; target against the *source*, as at inference time, where the
    request text is the only casing evidence. Restoring target casing from information the
    deployed system lacks would teach a capitalisation policy it can never reproduce.
    """
    src_natural = denormalize_from_model(source, source)
    tgt_natural = denormalize_from_model(target, source)
    return src_natural, tgt_natural


def denormalize_split(split: Dict[str, List[str]]) -> Dict[str, List[str]]:
    sources: List[str] = []
    targets: List[str] = []
    for src, tgt in zip(split['source'], split['target'], strict=True):
        s, t = denormalize_pair(src, tgt)
        sources.append(s)
        targets.append(t)
    return {'source': sources, 'target': targets}


if __name__ == '__main__':
    # eyeball a few pairs before spending GPU hours on them
    from dataio import load_d_wikipedia

    data = load_d_wikipedia(scope='prove_loop')
    for i in range(3):
        src, tgt = data['train']['source'][i], data['train']['target'][i]
        n_src, n_tgt = denormalize_pair(src, tgt)
        print(f'--- pair {i} ---')
        print(f'corpus source : {src[:220]}')
        print(f'natural source: {n_src[:220]}')
        print(f'corpus target : {tgt[:220]}')
        print(f'natural target: {n_tgt[:220]}\n')
