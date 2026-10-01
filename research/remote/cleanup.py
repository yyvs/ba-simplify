"""Reclaim disk on a shared machine, without deleting anything a result depends on.

The motd asks for this directly: "Remember to delete large files once you are done." It is
not just politeness -- these boxes have one disk between everyone using them, and the thing
this session generates most of is 1.6 GB intermediate checkpoints.

The distinction that matters, and the reason this is a script rather than an `rm -rf`:

  - **`best_checkpoint/` is the artifact.** ~535 MB of weights, and the only thing a reader,
    a re-evaluation or the Hub upload needs. Never removed here.
  - **`checkpoint-N/` are resume state.** ~1.6 GB each, because they carry optimiser,
    scheduler and RNG state alongside the weights. Useful only to someone continuing that
    exact run. Once training has finished and `best_checkpoint` exists, they are dead weight.
  - **Generations caches are expensive to recreate and cheap to keep.** A few MB of JSON that
    cost GPU hours; removing them would force regeneration to add a metric. Never touched.
  - **`training_history_*.json` replaces the need to keep checkpoints for their metrics.**
    train.py writes the full log history separately for exactly this reason, so pruning
    checkpoints never costs a training curve.

Default is a dry run. Nothing is deleted without you reading the list first.

    python remote/cleanup.py                 # what would go, and how much space
    python remote/cleanup.py --apply
    python remote/cleanup.py --apply --hf-cache   # also the HuggingFace model cache
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from common import SCRATCH, free_gb


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def gb(num_bytes: int) -> float:
    return num_bytes / 1024**3


def weights_fingerprint(checkpoint: Path) -> Optional[str]:
    """Cheap identity for a checkpoint's weights: size plus a hash of the head and tail.

    Avoids reading 532 MB per checkpoint on a dry run. Two BART checkpoints from different
    steps never agree on size plus 1 MB from each end.
    """
    weights = checkpoint / 'model.safetensors'
    if not weights.exists():
        return None
    size = weights.stat().st_size
    digest = hashlib.sha256()
    with weights.open('rb') as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 2 * 1024 * 1024:
            handle.seek(-1024 * 1024, os.SEEK_END)
            digest.update(handle.read())
    return f'{size}:{digest.hexdigest()[:16]}'


def find_prunable(include_hf: bool) -> List[Tuple[Path, int, str]]:
    targets: List[Tuple[Path, int, str]] = []

    for run_dir in sorted(SCRATCH.glob('simplification_results*')):
        if not run_dir.is_dir():
            continue
        best = run_dir / 'best_checkpoint'
        checkpoints = sorted(
            (
                p
                for p in run_dir.iterdir()
                if p.is_dir() and p.name.startswith('checkpoint-')
            ),
            key=lambda p: int(p.name.split('-')[1]),
        )
        if not checkpoints:
            continue
        if not best.exists():
            # without best_checkpoint, the newest intermediate one is the only copy of the weights
            keep = checkpoints[-1]
            for ckpt in checkpoints[:-1]:
                targets.append((ckpt, dir_size(ckpt), 'intermediate resume state'))
            print(
                f'  note: {run_dir.name} has no best_checkpoint/, so {keep.name} is kept as '
                f'the only copy of these weights. Finish or re-run the training, or copy the '
                f'weights out, before pruning it.'
            )
            continue

        # Only prune checkpoints whose weights best_checkpoint preserves. Real case: §6.4
        # attributes findings to "best_checkpoint, epoch 1.0, checkpoint-2500", but
        # best_checkpoint is byte-identical to checkpoint-5000 (epoch 2.0); resolving that
        # needs the epoch-1 weights.
        best_fingerprint = weights_fingerprint(best)
        for ckpt in checkpoints:
            fingerprint = weights_fingerprint(ckpt)
            if best_fingerprint and fingerprint and fingerprint != best_fingerprint:
                print(
                    f'  KEEPING {ckpt.relative_to(SCRATCH.parent)}: its weights differ from '
                    f'best_checkpoint/, so this is the only copy of them. If it is genuinely '
                    f'disposable, delete it by hand -- but check first whether any result '
                    f'cites it (§6.4 cites a document checkpoint by step number).'
                )
                continue
            targets.append(
                (
                    ckpt,
                    dir_size(ckpt),
                    'resume state; weights preserved in best_checkpoint',
                )
            )

    archives = list((SCRATCH / 'd_wikipedia_raw').glob('*.7z'))
    for archive in archives:
        # the loader reads the extracted .src/.tgt; the archive re-downloads in seconds
        if (SCRATCH / 'd_wikipedia_raw' / 'train.src').exists():
            targets.append(
                (archive, archive.stat().st_size, 'already extracted, re-downloadable')
            )

    if include_hf:
        hf_home = os.environ.get('HF_HOME')
        if hf_home:
            hub = Path(hf_home) / 'hub'
            if hub.exists():
                targets.append(
                    (hub, dir_size(hub), 'HuggingFace cache — re-downloadable')
                )

    return targets


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--apply', action='store_true', help='actually delete')
    ap.add_argument(
        '--hf-cache',
        action='store_true',
        help='include the HuggingFace model cache (re-downloadable, but slow)',
    )
    ap.add_argument(
        '--dry-run', action='store_true', help='explicit no-op; this is the default'
    )
    args = ap.parse_args()

    print(f'free before: {free_gb():.1f} GB\n')
    targets = find_prunable(args.hf_cache)

    if not targets:
        print('Nothing prunable found. Kept regardless, by design:')
        print('  best_checkpoint/            the artifact a reader needs')
        print('  results/**/generations_*    hours of GPU time as a few MB of JSON')
        print('  results/**/*.partial.json   resume state for an interrupted eval')
        print('  training_history_*.json     the metric trajectory')
        return

    total = 0
    print(f"{'size':>9}  path")
    print('-' * 78)
    for path, size, why in targets:
        total += size
        try:
            shown = path.relative_to(SCRATCH.parent)
        except ValueError:
            shown = path
        print(f'{gb(size):>7.2f}G  {shown}   ({why})')
    print('-' * 78)
    print(f'{gb(total):>7.2f}G  total\n')

    if not args.apply:
        print('Dry run. Re-run with --apply to delete. Never removed by this script:')
        print(
            '  best_checkpoint/, results/**/generations_*.json, *.partial.json, '
            'training_history_*.json'
        )
        return

    for path, _size, _why in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f'  removed {path}')
        except OSError as exc:
            print(f'  could not remove {path}: {exc}')

    print(f'\nfree after: {free_gb():.1f} GB')


if __name__ == '__main__':
    main()
