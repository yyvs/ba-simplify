"""Compute LENS over cached document generations, in an environment of its own.

**The problem this solves.** LENS (Maddela et al., ACL 2023) is the only metric in this
project trained directly on human simplification judgements, which makes it the one metric
that partially answers the objection §6.3 raises against all the others -- that they reward
the census-year substitution. It is missing from every reported row. Not for lack of trying:
`lens-metric` pins its own torch/transformers versions, which conflict with the versions the
FastAPI service needs, so installing it into the shared venv breaks the running backend.
§6.6 records the consequence: "in practice LENS can therefore only be obtained from a
disposable runtime, which means it is not a metric this project can report routinely."

On a machine where you control the environment, a second virtualenv is the entire fix. That
is what `setup.sh --lens` builds and what this script runs in.

**Why a separate script rather than `evaluate_document.py --rescore` in the LENS venv.**
`--rescore` would work and is the more elegant route, but it needs easse, sacrebleu and
bert-score importable in the conflicting environment too, so a version fight anywhere in
that stack takes the whole rescore with it. This reads the generations JSON, computes one
metric, writes one artifact, and lets `collect_results.py` merge it. Minimal dependency
surface is the point: the fewer packages have to coexist, the fewer ways this has of failing
at hour 20 of a reservation.

    # in the LENS venv
    ../venv-lens/bin/python remote/lens_only.py \
        --generations results/remote/document_eval \
        --outdir results/remote/document_eval

Reported descriptively as a corpus mean per system, never significance-tested -- each LENS
score is a full neural forward pass, so bootstrapping it at 1000 resamples would cost orders
of magnitude more than SARI/D-SARI's n-gram arithmetic for the same statistical payoff. That
matches how §2.6 already says LENS is treated.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from common import RESEARCH_DIR, human_seconds, rel, write_artifact

SYSTEMS = ('finetuned', 'baseline', 'llm')


def find_generations(where: Path) -> Optional[Path]:
    """Newest generations cache in a directory, or the file itself if one was named.

    `evaluate_document.py` names these `generations_n<N>_seed<S>.json`, so newest is safe
    enough; name an older file explicitly if needed.
    """
    path = Path(where)
    if path.is_file():
        return path
    candidates = sorted(
        glob.glob(str(path / 'generations_n*_seed*.json')),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    return Path(candidates[0]) if candidates else None


def load_sources_and_references(
    cache: Dict, limit: int, seed: int, data_dir: Optional[str] = None
) -> tuple[List[str], List[List[str]]]:
    """Sources and references for the cached run.

    Prefer the texts stored in the generations file. Otherwise re-derive the sample from the
    corpus and verify the recorded `sources_sha` (same pattern as
    `review_document_outputs.py`), rather than trusting seed and limit to reproduce it.
    """
    if cache.get('sources') and cache.get('references'):
        refs = cache['references']
        # references may be stored flat (one per document) or already nested
        nested = [r if isinstance(r, list) else [r] for r in refs]
        return list(cache['sources']), nested

    import sys

    sys.path.insert(0, str(RESEARCH_DIR))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        '_evaluate_document', RESEARCH_DIR / 'evaluate_document.py'
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load evaluate_document.py from {RESEARCH_DIR}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # load_dwikipedia_test's data_dir is required; default to evaluate_document.py's constant
    resolved = data_dir or getattr(
        module, 'DEFAULT_DATA_DIR', 'scratch/d_wikipedia_raw'
    )
    sources, references = module.load_dwikipedia_test(resolved, limit, seed)[:2]
    recorded = cache.get('sources_sha')
    if recorded:
        derived = module._sha16('\n'.join(sources))
        if derived != recorded:
            raise SystemExit(
                f'sources_sha mismatch: the generations were produced from a different '
                f'document set than --limit {limit} --seed {seed} re-derives '
                f'(recorded {recorded}, derived {derived}). Scoring these predictions '
                f'against these sources would pair the wrong documents. Point --limit/--seed '
                f'at the run that produced the cache.'
            )
    nested = [r if isinstance(r, list) else [r] for r in references]
    return list(sources), nested


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--generations',
        default='results/remote/document_eval',
        help='a generations_n*_seed*.json file, or a directory holding one',
    )
    ap.add_argument('--outdir', default='results/remote/document_eval')
    ap.add_argument(
        '--limit',
        type=int,
        default=None,
        help='documents the cache was produced from; derived from its '
        'filename when omitted, which is what you want',
    )
    ap.add_argument('--seed', type=int, default=None, help='likewise')
    ap.add_argument(
        '--data-dir',
        default=None,
        help="D-Wikipedia raw split; defaults to evaluate_document.py's own value",
    )
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument(
        '--checkpoint',
        default='davidheineman/lens',
        help="the authors' published LENS checkpoint",
    )
    args = ap.parse_args()

    cache_path = find_generations(Path(args.generations))
    if cache_path is None:
        raise SystemExit(
            f'no generations cache under {args.generations}. Run the eval_doc stage first -- '
            f'this script scores existing generations and never calls a simplification model.'
        )
    print(f'reading {cache_path}')
    cache = json.loads(cache_path.read_text())

    # The cache stores predictions only, so the document set is re-derived with the
    # generation run's limit and seed (else the length guard rejects everything), read from
    # the filename ("generations_n2000_seed42.json"). The pipeline's lens stage passes neither.
    from_name = re.search(r'generations_n(\d+)_seed(\d+)', cache_path.name)
    limit = (
        args.limit
        if args.limit is not None
        else (int(from_name.group(1)) if from_name else 0)
    )
    seed = (
        args.seed
        if args.seed is not None
        else (int(from_name.group(2)) if from_name else 42)
    )
    print(
        f'scoring against limit={limit} seed={seed}'
        f"{' (from the cache filename)' if from_name and args.limit is None else ''}"
    )

    sources, references = load_sources_and_references(cache, limit, seed, args.data_dir)

    try:
        # only resolvable inside venv-lens
        from lens import LENS, download_model  # type: ignore[import-not-found]
    except Exception as exc:
        raise SystemExit(
            f'lens-metric is not importable here ({exc}).\n'
            f'This script is meant to run in the dedicated venv, because lens-metric pins '
            f"torch/transformers versions that conflict with the service's:\n"
            f'    bash remote/setup.sh --lens\n'
            f'    ../venv-lens/bin/python remote/lens_only.py ...'
        ) from None

    print(f'loading LENS checkpoint {args.checkpoint}')
    # rescale=True: same 0-100 scale as the other metrics
    model = LENS(download_model(args.checkpoint), rescale=True)

    scores: Dict[str, Optional[float]] = {}
    per_system_seconds: Dict[str, float] = {}
    for system in SYSTEMS:
        predictions = cache.get(f'{system}_predictions')
        if not predictions:
            print(f'  {system}: absent from the cache, skipping')
            scores[system] = None
            continue
        if len(predictions) != len(sources):
            raise SystemExit(
                f'{system}: {len(predictions)} predictions against {len(sources)} sources. '
                f'A partial generation run cannot be scored as a complete one -- finish the '
                f'eval_doc stage first.'
            )
        print(f'  {system}: scoring {len(predictions)} documents')
        started = time.time()
        values = model.score(
            sources,
            list(predictions),
            references,
            batch_size=args.batch_size,
            devices=[0],
        )
        elapsed = time.time() - started
        scores[system] = float(sum(values) / len(values))
        per_system_seconds[system] = round(elapsed, 1)
        print(f'    LENS {scores[system]:.2f}  ({human_seconds(elapsed)})')

    n = len(sources)
    outdir = Path(args.outdir)
    write_artifact(
        outdir / f'lens_n{n}_seed{seed}.json',
        {
            'metric': 'LENS',
            'citation': 'Maddela, Dou, Heineman & Xu (ACL 2023)',
            'checkpoint': args.checkpoint,
            'rescaled_to_0_100': True,
            # rel(): relative_to() on a relative --generations path once crashed this write
            'generations_file': rel(cache_path),
            'sources_sha': cache.get('sources_sha'),
            'n_documents': n,
            'seed': seed,
            'batch_size': args.batch_size,
            'scores': scores,
            'seconds_per_system': per_system_seconds,
            'reporting': (
                'Descriptive corpus mean per system. Deliberately not significance-tested: '
                'each score is a neural forward pass, so 1000 bootstrap resamples would cost '
                "orders of magnitude more than SARI/D-SARI's n-gram arithmetic for the same "
                'statistical payoff (the thesis §2.6).'
            ),
        },
    )


if __name__ == '__main__':
    main()
