"""Refuse to start an expensive run while a known defect would corrupt its output.

This is the most valuable file in this directory, and the reason is arithmetic. The session
ahead is 12-20 GPU hours. Three defects currently in the repository would each silently
change numbers produced during it, and all three are already known -- two are recorded in
the project's own RESULTS.md as "not yet done". Discovering after the fact that the
document-level results were generated through a normaliser that manufactures `u . s .` from
`u.s.` means running the whole thing again, on a machine you have to reserve.

So this gate runs first, costs seconds, and exits non-zero if the session would produce
compromised output. `--warn-only` reports without blocking, for when you know better --
but the default is to block, because the failure mode being prevented is invisible.

    python preflight.py                     # gate the whole session
    python preflight.py --stage document    # only what the document stages need
    python preflight.py --warn-only         # report, never block

Checks marked BLOCKING would change a reported number. Checks marked ADVISORY concern
reproducibility or hygiene and are worth fixing but will not corrupt a result.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from common import (
    GPU_RESULTS,
    REPO_DIR,
    RESEARCH_DIR,
    free_gb,
    git_dirty,
    torch_info,
    write_artifact,
)

BACKEND = REPO_DIR / 'backend'


@dataclass
class Result:
    name: str
    ok: bool
    blocking: bool
    detail: str
    fix: str = ''
    stages: List[str] = field(default_factory=lambda: ['all'])


def _import_backend(module: str):
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    return importlib.import_module(module)


# BLOCKING: things that change a number
def check_normalizer_abbreviations() -> Result:
    """`normalize_for_model` must not split abbreviation periods.

    D-Wikipedia test has `u.s.` 531 times and `u . s .` zero times. `backend/main.py` applies
    the same function to document input, so production is affected too, not just scoring.
    """
    try:
        dt = _import_backend('document_text')
        got = dt.normalize_for_model('The U.S. Army in 1990.')
    except Exception as exc:
        return Result(
            'normalizer: abbreviations',
            False,
            True,
            f'could not import/run document_text: {exc}',
            'check backend/document_text.py imports cleanly',
            ['document'],
        )
    bad = 'u . s .' in got
    return Result(
        'normalizer: abbreviations',
        not bad,
        True,
        f"normalize_for_model('The U.S. Army in 1990.') -> {got!r}",
        fix=(
            'Fix the period rule in backend/document_text.py so a period inside an '
            'abbreviation is not split off. `_PERIOD_NOT_IN_NUMBER` currently splits any '
            'period not adjacent to a digit; it also needs to leave alone a period whose '
            'preceding character is a single letter (the `u.s.`, `e.g.`, `a.m.` shape). Add '
            'a case to backend/tests/test_document_text_normalization.py asserting '
            "'u.s.' survives, then rerun. Fixing this repairs the serving path and the "
            'metric-normalisation step at once.'
        ),
        stages=['document'],
    )


def check_prompting_document_constant() -> Result:
    """`evaluate_document.py --llm` needs whatever granularity constant it references.

    The rename `prompting.DOCUMENT` -> `prompting.WHOLE_SECTIONS` left four old call sites in
    evaluate_document.py (480, 506, 533, 1000), all inside `if args.llm`: the three-way run
    raises AttributeError partway in.
    """
    try:
        prompting = _import_backend('prompting')
    except Exception as exc:
        return Result(
            'prompting: document granularity constant',
            False,
            True,
            f'could not import prompting: {exc}',
            '',
            ['document_llm'],
        )

    src = (RESEARCH_DIR / 'evaluate_document.py').read_text()
    referenced = sorted(
        {
            name
            for name in (
                'DOCUMENT',
                'WHOLE_SECTIONS',
                'SENTENCE',
                'SENTENCE_BY_SENTENCE',
            )
            if f'prompting.{name}' in src
        }
    )
    missing = [name for name in referenced if not hasattr(prompting, name)]
    return Result(
        'prompting: document granularity constant',
        not missing,
        True,
        f"evaluate_document.py references {referenced or ['none']}; missing from prompting: "
        f"{missing or ['none']}",
        fix=(
            'Either re-export the old name in backend/prompting.py (`DOCUMENT = '
            'WHOLE_SECTIONS`) or update the four call sites in research/evaluate_document.py '
            'to the new constant. Prefer updating the call sites -- an alias keeps the '
            'retired vocabulary alive, which is what the rename was for. Then add '
            "backend/tests/test_vocabulary.py, which vocabulary.py's own docstring already "
            'promises exists.'
        ),
        stages=['document_llm'],
    )


def check_d_sari_reference() -> Result:
    """The D-SARI port must reproduce the reference implementation's own test cases.

    D-SARI carries the document-level claim (+21.10 over the un-fine-tuned baseline,
    p < 0.001). checks/test_d_sari.py tests the port against upstream.
    """
    test = Path(__file__).parent / 'checks' / 'test_d_sari.py'
    if not test.exists():
        return Result(
            'd_sari: reference test', False, True, f'{test} missing', '', ['document']
        )
    proc = subprocess.run(
        [sys.executable, str(test)],
        capture_output=True,
        text=True,
        cwd=str(RESEARCH_DIR),
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return Result(
        'd_sari: reference test',
        proc.returncode == 0,
        True,
        tail[-1] if tail else f'exit {proc.returncode}',
        fix=(
            'Run `python remote/checks/test_d_sari.py` from research/ and read the diff. If '
            'the port has drifted, fix it before generating any D-SARI number; if the '
            'expected values in the test are what drifted, update them only against the '
            'upstream RLSNLP/Document-level-text-simplification implementation, never '
            "against this port's own output."
        ),
        stages=['document'],
    )


def check_corpus_filter_decided() -> Result:
    """The filter divergence has to be a recorded decision before training on either side.

    prepare_data.py drops 6,206 train rows, the notebook's targeted filter 2,205 (~4,000 rows
    of disagreement). The data statement describes the former; the latter trained.
    """
    artifact = GPU_RESULTS / 'corpus_filter_divergence.json'
    return Result(
        'corpus: filter divergence recorded',
        artifact.exists(),
        True,
        f"{'present' if artifact.exists() else 'missing'}: {artifact}",
        fix='python remote/dataio.py --report   (run from research/; ~1 min, CPU only)',
        stages=['train'],
    )


def check_overlap_audited() -> Result:
    """Train/eval overlap has to be measured before results are reported off it.

    ASSET descends from TurkCorpus (built from WikiLarge sources), and the WikiLarge copy is a
    third-party mirror, so held-out status is not guaranteed. Measured: 5/359 ASSET test and
    71/397 WikiLarge validation sources appear in train; the latter selects the checkpoint.
    """
    artifact = GPU_RESULTS / 'overlap_audit.json'
    return Result(
        'corpus: train/eval overlap audited',
        artifact.exists(),
        True,
        f"{'present' if artifact.exists() else 'missing'}: {artifact}",
        fix='python remote/checks/check_overlap.py   (run from research/; ~2 min, CPU only)',
        stages=['all'],
    )


# BLOCKING: environment
def check_gpu() -> Result:
    info = torch_info()
    ok = bool(info.get('cuda_available'))
    detail = (
        f"torch {info.get('torch')} (cuda build {info.get('torch_cuda_build')}), "
        f"device={info.get('device_name')}, vram={info.get('vram_gb')}GB, "
        f"cc={info.get('compute_capability')}, bf16={info.get('bf16_supported')}"
        if ok
        else f"no CUDA device: {info.get('error', 'torch.cuda.is_available() is False')}"
    )
    return Result(
        'environment: CUDA visible to torch',
        ok,
        True,
        detail,
        fix=(
            "Check `nvidia-smi` first. If it reports 'Failed to initialize NVML: "
            "Driver/library version mismatch', the machine needs a reboot -- that is the "
            'documented fix in the gpus-at-cl-hhu README and you will need an admin (David '
            'Arps or Kilian Evang). If nvidia-smi works but torch does not see the device, '
            "the installed torch wheel does not match the driver's CUDA version: reinstall "
            "from the index matching `nvidia-smi`'s CUDA version (see requirements-gpu.txt)."
        ),
    )


def check_metric_deps() -> Result:
    missing = []
    for module in (
        'easse',
        'sacrebleu',
        'bert_score',
        'datasets',
        'transformers',
        'torch',
    ):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return Result(
        'environment: metric packages importable',
        not missing,
        True,
        f"missing: {missing or ['none']}",
        fix='bash remote/setup.sh   (creates the venvs and installs both requirement sets)',
    )


def check_easse_pinned() -> Result:
    """easse must not be installed from a moving branch.

    Every SARI and FKGL number comes from it, and requirements.txt installs
    `git+...easse.git@master`.
    """
    text = (RESEARCH_DIR / 'requirements.txt').read_text()
    hhu = (Path(__file__).parent / 'requirements-gpu.txt').read_text()
    moving = 'easse.git@master' in text and 'easse.git@master' in hhu
    return Result(
        'environment: easse pinned to a commit',
        not moving,
        False,
        (
            'requirements-gpu.txt still installs easse from @master'
            if moving
            else 'pinned'
        ),
        fix=(
            "Capture the commit you are actually running and pin it, from pip's own\n"
            'install metadata (PEP 610):\n'
            '  python -c "import json,importlib.metadata as m;print(json.loads(\n'
            "      m.distribution('easse').read_text('direct_url.json'))['vcs_info']['commit_id'])\"\n"
            'NOT `git -C <site-packages>/.. rev-parse HEAD`: the venvs sit inside this\n'
            "repository, so git walks up and reports the PROJECT's HEAD -- a plausible-looking\n"
            'sha that pins the wrong code.\n'
            'then replace `@master` with `@<sha>` in remote/requirements-gpu.txt.'
        ),
    )


# ADVISORY
def check_tree_clean() -> Result:
    dirty = git_dirty()
    return Result(
        'repo: working tree clean',
        dirty is False,
        False,
        'uncommitted changes present' if dirty else 'clean',
        fix=(
            "Commit before a long run. Otherwise every artifact's `git_commit` names a "
            'revision that is not the code that produced it, and the manifests will carry '
            '`git_dirty: true` -- which is honest but not citable.'
        ),
    )


def check_disk() -> Result:
    free = free_gb()
    return Result(
        'environment: disk headroom',
        free >= 40,
        False,
        f'{free:.1f} GB free at research/scratch',
        fix=(
            'A document run keeps ~3 checkpoints at ~1.6 GB plus the HF cache. The motd asks '
            'you to delete large files when done: `python remote/cleanup.py --dry-run`.'
        ),
    )


def check_hf_home() -> Result:
    import os

    hf = os.environ.get('HF_HOME')
    return Result(
        'environment: HF_HOME set outside $HOME',
        bool(hf),
        False,
        hf or 'unset (models will cache into ~/.cache/huggingface)',
        fix=(
            'Home directories on these machines are small (50 GB soft quota on the HPC '
            'cluster). Point the cache at working storage:\n'
            '  source remote/profiles/beet.env'
        ),
    )


CHECKS: List[Callable[[], Result]] = [
    check_gpu,
    check_metric_deps,
    check_normalizer_abbreviations,
    check_prompting_document_constant,
    check_d_sari_reference,
    check_corpus_filter_decided,
    check_overlap_audited,
    check_easse_pinned,
    check_tree_clean,
    check_disk,
    check_hf_home,
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--stage',
        default='all',
        help='only checks relevant to this stage (train, document, document_llm, all)',
    )
    ap.add_argument('--warn-only', action='store_true', help='report without blocking')
    ap.add_argument('--no-artifact', action='store_true')
    args = ap.parse_args()

    results: List[Result] = []
    for check in CHECKS:
        try:
            results.append(check())
        except (
            Exception
        ) as exc:  # a check must never be the thing that breaks the session
            results.append(
                Result(check.__name__, False, False, f'check errored: {exc}')
            )

    relevant = [
        r
        for r in results
        if args.stage == 'all' or 'all' in r.stages or args.stage in r.stages
    ]

    width = max(len(r.name) for r in relevant) + 2
    print('\nPREFLIGHT\n' + '=' * (width + 30))
    for r in relevant:
        tag = 'ok  ' if r.ok else ('FAIL' if r.blocking else 'warn')
        print(f'[{tag}] {r.name.ljust(width)} {r.detail}')

    failures = [r for r in relevant if not r.ok and r.blocking]
    warnings = [r for r in relevant if not r.ok and not r.blocking]

    for r in failures + warnings:
        if r.fix:
            print(f'\n--- {r.name} ---\n{r.fix}')

    if not args.no_artifact:
        write_artifact(
            GPU_RESULTS / 'preflight.json',
            {
                'stage': args.stage,
                'passed': not failures,
                'checks': [
                    {
                        'name': r.name,
                        'ok': r.ok,
                        'blocking': r.blocking,
                        'detail': r.detail,
                    }
                    for r in relevant
                ],
            },
            quiet=True,
        )

    print()
    if failures and not args.warn_only:
        print(
            f'{len(failures)} blocking check(s) failed. Each one would change a number this '
            f'session produces, so fix them before spending GPU hours -- or rerun with '
            f'--warn-only if you have decided otherwise deliberately.'
        )
        raise SystemExit(1)
    if failures:
        print(
            f'{len(failures)} blocking check(s) failed; continuing because --warn-only.'
        )
    else:
        print(f'All blocking checks passed. {len(warnings)} advisory warning(s).')


if __name__ == '__main__':
    main()
