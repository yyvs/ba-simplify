"""Staged, resumable orchestrator for the whole GPU session.

**What this is for.** The session's real goal is not "train a model" -- it is to produce, on
one machine with one code version, the complete set of numbers the thesis reports. Right now
those numbers come from four different hosts with different filters, different precision and
different decoding, and the resulting comparability caveats take up a large part of
the thesis §6.6. Most of them can simply be retired by regenerating the reported set
here.

**Why stages rather than one script.** These machines are reserved through a shared calendar
(https://calendar.online/5afa284259e0c92ff6a2), so a run can lose its host at a scheduled
boundary. Every stage is therefore idempotent, records its own completion, and can be
re-entered: `--from` picks up where the reservation ended, and a finished stage is skipped
unless `--force`.

**Order is a dependency graph, not a preference.** The cheap CPU audits run first because two
of them gate correctness of everything after (the corpus filter decision must be settled
before training, or the new checkpoint inherits the same ambiguity as the old one). The smoke
stage runs before any multi-hour stage because the project has already been burned once by
16 passing mocked tests over a non-functional path (§4.9). The short ASSET stage runs before
the long document stage so that a lost reservation still leaves one completed result.

    python remote/pipeline.py --list                  # stages, estimates, current status
    python remote/pipeline.py --dry-run               # print the exact commands
    python remote/pipeline.py --only audits           # one group
    python remote/pipeline.py --from train_doc        # resume after a lost reservation
    python remote/pipeline.py --stretch               # include the natural-case experiment

Run from `research/`, inside tmux, with the GPU reserved. Every stage's stdout is teed to
results/remote/logs/<stage>.log so a dropped SSH costs nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from common import (
    GPU_LOGS,
    GPU_RESULTS,
    RESEARCH_DIR,
    ensure_dirs,
    human_seconds,
    load_state,
    mark_stage,
    rel,
    stage_status,
)

PY = sys.executable
HHU = 'remote'


@dataclass
class Stage:
    name: str
    group: str
    about: str = ''
    cmd: List[str] = field(default_factory=list)
    estimate: str = ''
    gpu: bool = False
    needs_ollama: bool = False
    venv: Optional[str] = None
    stretch: bool = False
    produces: List[str] = field(default_factory=list)


def build_stages(args: argparse.Namespace) -> List[Stage]:
    """The plan. Every command is one the repository already supports, except train.py.

    Deliberately absent: retraining the sentence checkpoint. It has artifact-backed results
    (RESULTS.md S4/S5), is on the Hub, and retraining would change the weights the thesis
    cites; the sentence branch only needs re-evaluation on one device (`asset`).
    """
    doc_ckpt = 'scratch/simplification_results_document/best_checkpoint'
    nat_ckpt = 'scratch/simplification_results_document_naturalcase/best_checkpoint'

    stages = [
        # audits: cheap, CPU-only; two of them gate everything after
        Stage(
            name='filters',
            group='audits',
            about='Quantify the mojibake-filter divergence and settle which filter is '
            'canonical. Must precede training: the tracked data statement describes '
            'one filter and the notebook trained with the other, and they disagree on '
            '~4,000 rows.',
            cmd=[PY, f'{HHU}/dataio.py', '--report'],
            estimate='~2 min',
            produces=['results/remote/corpus_filter_divergence.json'],
        ),
        Stage(
            name='overlap',
            group='audits',
            about='Measure train/eval contamination across all five split pairs. Produces '
            'the threats-to-validity numbers §6.6 currently has none of.',
            cmd=[PY, f'{HHU}/checks/check_overlap.py'],
            estimate='~5 min',
            produces=['results/remote/overlap_audit.json'],
        ),
        Stage(
            name='dsari',
            group='audits',
            about="Differential-test the D-SARI port against the paper's own implementation. "
            'Gates every document number: D-SARI carries the whole document claim and '
            'its verification is currently not reproducible from the repository.',
            cmd=[PY, f'{HHU}/checks/test_d_sari.py', '--n', '60'],
            estimate='~5 min',
        ),
        # smoke: prove the chain on this host first
        Stage(
            name='smoke',
            group='smoke',
            about='One prove_loop training run and a 20-document evaluation, end to end on '
            'this machine. Catches a wrong torch wheel, a missing tokenizer download, '
            'a full disk or a broken Ollama tag in 15 minutes instead of at hour six.',
            cmd=[
                PY,
                f'{HHU}/train.py',
                '--structure',
                'document',
                '--scope',
                'prove_loop',
                '--filter',
                args.filter,
                # Not train.py's default dir: it omits the scope, so train_doc's `--resume`
                # would continue the smoke checkpoint under a "full scope" manifest.
                '--output-dir',
                'scratch/smoke_document',
            ],
            estimate='~10 min',
            gpu=True,
        ),
        Stage(
            name='smoke_eval',
            group='smoke',
            about='Score the smoke checkpoint on 20 documents, including the prompted LLM '
            'condition, so the three-way path is known to work before the real run.',
            cmd=[
                PY,
                'evaluate_document.py',
                '--model',
                'scratch/smoke_document/best_checkpoint',
                '--limit',
                '20',
                '--seed',
                '42',
                '--device',
                'cuda',
                '--batch-size',
                str(args.doc_batch),
                '--llm',
                args.llm_tag,
                '--skip-lens',
                '--outdir',
                'results/remote/smoke',
            ],
            estimate='~10 min',
            gpu=True,
            needs_ollama=True,
        ),
        # sentence branch
        Stage(
            name='asset',
            group='sentence',
            about='The four-condition ASSET comparison at three seeds -- the run that was '
            'abandoned locally after one condition because it projected to ~3 hours at '
            'degraded throughput. On a GPU the seq2seq conditions are seconds and the '
            "prompted ones are minutes. This is what turns RQ1's 'compared with an "
            "off-the-shelf baseline' into a table with a *strong* baseline in it.",
            cmd=[
                PY,
                'evaluate_sentence.py',
                '--conditions',
                'base,online,local,llm_7b,llm_3b',
                '--seeds',
                '3',
                '--decoding',
                'beam',
                '--significance-baseline',
                'base',
                '--device',
                'cuda',
                '--allow-llm-generate',
                '--outdir',
                'results/remote/sentence_eval',
            ],
            estimate='1-2 h',
            gpu=True,
            needs_ollama=True,
            produces=['results/remote/sentence_eval/summary.json'],
        ),
        # document branch
        Stage(
            name='train_doc',
            group='document',
            about='Full-scope document training at 5 epochs. The existing document '
            'checkpoint is reduced-scope (20k documents, 2 epochs) and the 2-epoch cap '
            "is explicitly an artefact of Colab's 5h20m session ceiling, not a "
            'modelling choice. Removing that ceiling is the single biggest unfinished '
            'research item in the project.',
            cmd=[
                PY,
                f'{HHU}/train.py',
                '--structure',
                'document',
                '--scope',
                'full',
                '--filter',
                args.filter,
                *(['--batch-size', str(args.train_batch)] if args.train_batch else []),
                '--resume',
            ],
            estimate='5-8 h',
            gpu=True,
            produces=['results/remote/train_document_full.json'],
        ),
        # Two document evaluations: the prompted LLM takes ~2-4 s per document (~130 tokens)
        # even with GPU offload, so a three-way run on all 8,000 would take 5-9 h; seq2seq is
        # ~20x faster. Three-way at n=2000 (4x D4, identical documents), two-system RQ1 on the
        # full split, so no table mixes sample sizes.
        Stage(
            name='eval_doc',
            group='document',
            about="Three-way document evaluation at n=2000 -- four times D4's sample, with "
            "the prompted LLM on identical documents. Tests whether D4's ranking and "
            'its length-profile reversal hold at 4x the sample.',
            cmd=[
                PY,
                'evaluate_document.py',
                '--model',
                doc_ckpt,
                '--limit',
                '2000',
                '--seed',
                '42',
                '--device',
                'cuda',
                '--batch-size',
                str(args.doc_batch),
                '--llm',
                args.llm_tag,
                '--llm-seed',
                '1',
                '--resamples',
                '1000',
                '--skip-lens',
                '--outdir',
                'results/remote/document_eval',
            ],
            estimate='2-4 h',
            gpu=True,
            needs_ollama=True,
        ),
        Stage(
            name='eval_doc_full',
            group='document',
            about='Fine-tuned vs zero-shot baseline on the FULL 8,000-document test split '
            '(--limit 0). This is the primary document-level RQ1 claim, and running it '
            "unsampled retires §6.6's '(b) a seeded n=500 sample, not the full split' "
            'caveat outright rather than shrinking it. No LLM condition here -- see the '
            'comment above for why it would cost more than the training did.',
            cmd=[
                PY,
                'evaluate_document.py',
                '--model',
                doc_ckpt,
                '--baseline',
                'facebook/bart-base',
                '--limit',
                '0',
                '--seed',
                '42',
                '--device',
                'cuda',
                '--batch-size',
                str(args.doc_batch),
                '--resamples',
                '1000',
                '--skip-lens',
                '--outdir',
                'results/remote/document_eval_full',
            ],
            estimate='1-2 h',
            gpu=True,
        ),
        Stage(
            name='lens',
            group='document',
            about='LENS over the cached document generations, in its own virtualenv. LENS is '
            'the one metric in the project trained on human simplification judgements '
            'and it is currently missing from every single row, because lens-metric '
            "pins torch/transformers versions that conflict with the service's. A "
            'second venv is the whole fix.',
            cmd=[
                PY,
                f'{HHU}/lens_only.py',
                '--generations',
                'results/remote/document_eval',
                '--outdir',
                'results/remote/document_eval',
            ],
            estimate='5 min',
            gpu=True,
            venv='lens',
        ),
        Stage(
            name='review',
            group='document',
            about='Re-run the qualitative review against the new full-split generations, so '
            "§6.3's error taxonomy (the census-year class, over-deletion rates) "
            'describes the checkpoint the thesis reports rather than the reduced-scope '
            'one. Reads cached generations only -- no model, no GPU.',
            # filename carries the post-cleaning document count; {GLOB:...} expands at run time
            cmd=[
                PY,
                'review_document_outputs.py',
                '--generations',
                '{GLOB:results/remote/document_eval_full/generations_n*_seed42.json}',
                '--partial',
                '{GLOB:results/remote/document_eval_full/partial_finetuned_n*_seed42.json}',
                '--limit',
                '0',
                '--seed',
                '42',
                '--out',
                'results/remote/document_eval_full/qualitative_review_full.json',
            ],
            estimate='~10 min',
        ),
        # stretch: §7.1's highest-value follow-up
        Stage(
            name='train_doc_natural',
            group='stretch',
            about='Retrain the document model on natural-case, normally-punctuated text. '
            "§7.1: 'training on the convention the extension actually encounters would "
            "remove both the workaround and its residual failure modes entirely.' This "
            'is a genuinely new contribution rather than a completed measurement -- it '
            "removes document_text.py's normalize/de-normalize layer from the deployed "
            "path, and §6.4's Finding 3 already measured that layer's absence causing a "
            'factual hallucination.',
            cmd=[
                PY,
                f'{HHU}/train.py',
                '--structure',
                'document',
                '--scope',
                'full',
                '--filter',
                args.filter,
                '--natural-case',
                '--resume',
            ],
            estimate='5-8 h',
            gpu=True,
            stretch=True,
        ),
        Stage(
            name='eval_doc_natural',
            group='stretch',
            about='Score the natural-case checkpoint against the same test documents, '
            'de-normalised the same way. The comparison to report is against the '
            'corpus-convention checkpoint on identical text -- that is the number that '
            'says whether the workaround cost anything.',
            cmd=[
                PY,
                'evaluate_document.py',
                '--model',
                nat_ckpt,
                '--limit',
                '0',
                '--seed',
                '42',
                '--device',
                'cuda',
                '--batch-size',
                str(args.doc_batch),
                '--skip-lens',
                '--outdir',
                'results/remote/document_eval_naturalcase',
            ],
            estimate='1-2 h',
            gpu=True,
            stretch=True,
        ),
        # collect
        Stage(
            name='collect',
            group='collect',
            about='Assemble every artifact this session produced into RESULTS_GPU.md with '
            'provenance, ready to diff against the committed RESULTS.md.',
            cmd=[PY, f'{HHU}/collect_results.py'],
            estimate='~1 min',
            produces=['results/remote/RESULTS_GPU.md'],
        ),
    ]
    for stage in stages:
        stage.cmd = [part for part in stage.cmd if part != '']
    return stages


def check_ollama(tag: str) -> Optional[str]:
    """Confirm a served tag is present before a stage that needs it.

    Without the tag, `evaluate_document.py --llm` fails partway through generation (cf. §4.9:
    HTTP 400 on every prompted request while 16 mocked tests passed).
    """
    # Over HTTP, as evaluate_*.py uses it. The CLI may not be on PATH: a rootless install
    # lands in ~/.local/bin, which no login profile here adds.
    host = os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')
    if not host.startswith('http'):
        host = f'http://{host}'
    try:
        with urllib.request.urlopen(f'{host}/api/tags', timeout=10) as response:
            served = {m.get('model', '') for m in json.load(response).get('models', [])}
    # Narrow on purpose: a bug here must crash, not silently skip stages (a broad except once
    # turned a NameError into "no Ollama server" and the run reported success).
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        hint = '' if shutil.which('ollama') else ' (the binary is not on PATH either)'
        return (
            f'no Ollama server answering at {host}: {exc}{hint}. Start one:\n'
            f'    ollama serve &            # ~/.local/bin/ollama if installed without root\n'
            f'    ollama pull {tag}'
        )
    if tag not in served and tag.split(':')[0] not in {s.split(':')[0] for s in served}:
        return (
            f"{tag} is not pulled; {host} serves {sorted(served) or 'nothing'}. Run:\n"
            f'    ollama pull {tag}'
        )
    return None


def expand_globs(cmd: List[str]) -> Optional[List[str]]:
    """Resolve any `{GLOB:pattern}` argument against the filesystem, now.

    Some filenames encode a count only the previous stage knows
    (`generations_n<documents>_seed<seed>.json`). Returns None if a pattern matches nothing,
    i.e. the upstream stage has not run.
    """
    resolved: List[str] = []
    for part in cmd:
        if not (part.startswith('{GLOB:') and part.endswith('}')):
            resolved.append(part)
            continue
        pattern = part[len('{GLOB:') : -1]
        matches = sorted(
            (RESEARCH_DIR / p for p in glob.glob(pattern, root_dir=str(RESEARCH_DIR))),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not matches:
            # show directory contents: a mistyped pattern once silently skipped `review`
            print(f'  unresolved: no file matches {pattern}')
            parent = RESEARCH_DIR / Path(pattern).parent
            if parent.is_dir():
                names = sorted(q.name for q in parent.iterdir())
                print(f"    {parent.name}/ holds: {', '.join(names) or '(empty)'}")
            else:
                print(f'    {parent} does not exist')
            return None
        resolved.append(rel(matches[0]))
    return resolved


def run_stage(stage: Stage, args: argparse.Namespace) -> bool:
    log = GPU_LOGS / f'{stage.name}.log'
    env = os.environ.copy()

    if stage.venv:
        # second interpreter (LENS): its pins conflict with the service's
        interpreter = RESEARCH_DIR / f'venv-{stage.venv}' / 'bin' / 'python'
        if not interpreter.exists():
            print(f'  SKIP {stage.name}: {interpreter} missing (run setup.sh --lens)')
            mark_stage(stage.name, 'skipped', reason='venv missing')
            return True
        cmd = [str(interpreter)] + stage.cmd[1:]
    else:
        cmd = stage.cmd

    if stage.needs_ollama:
        problem = check_ollama(args.llm_tag)
        if problem:
            print(f'  SKIP {stage.name}: {problem}')
            mark_stage(stage.name, 'skipped', reason='ollama unavailable')
            return True

    if args.dry_run:
        print(
            f"\n=== {stage.name} ({stage.estimate}) ===\n  {' '.join(cmd)}\n  log: {log}"
        )
        return True

    expanded = expand_globs(cmd)
    if expanded is None:
        print(
            f'  SKIP {stage.name}: an input it depends on does not exist yet. Run the stage '
            f'that produces it first, then `--from {stage.name}`.'
        )
        mark_stage(stage.name, 'skipped', reason='unresolved input')
        return True
    cmd = expanded

    printable = ' '.join(cmd)
    print(f'\n=== {stage.name} ({stage.estimate}) ===\n  {printable}\n  log: {log}')

    mark_stage(stage.name, 'running', command=printable)
    started = time.time()
    with log.open('a', encoding='utf-8') as handle:
        handle.write(
            f"\n\n===== {stage.name} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
        )
        handle.write(printable + '\n\n')
        handle.flush()
        # tee: the log survives a dropped SSH
        process = subprocess.Popen(
            cmd,
            cwd=str(RESEARCH_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:  # stdout=PIPE above guarantees otherwise
            raise RuntimeError('subprocess was started without a stdout pipe')
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        code = process.wait()

    elapsed = time.time() - started
    if code == 0:
        mark_stage(
            stage.name, 'done', elapsed_seconds=round(elapsed, 1), command=printable
        )
        print(f'  {stage.name} done in {human_seconds(elapsed)}')
        return True

    mark_stage(stage.name, 'failed', exit_code=code, elapsed_seconds=round(elapsed, 1))
    print(
        f'  {stage.name} FAILED (exit {code}) after {human_seconds(elapsed)}; see {log}'
    )
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--list', action='store_true', help='show stages and current status'
    )
    ap.add_argument(
        '--dry-run', action='store_true', help='print commands without running'
    )
    ap.add_argument(
        '--only', metavar='NAME_OR_GROUP', help='run one stage or one group'
    )
    ap.add_argument('--from', dest='start', metavar='NAME', help='start at this stage')
    ap.add_argument('--skip', default='', help='comma-separated stage names to skip')
    ap.add_argument(
        '--force', action='store_true', help='rerun stages already marked done'
    )
    ap.add_argument(
        '--stretch', action='store_true', help='include the natural-case experiment'
    )
    ap.add_argument(
        '--keep-going', action='store_true', help='continue past a failed stage'
    )
    ap.add_argument(
        '--filter',
        default='strict',
        choices=['strict', 'targeted', 'none'],
        help='mojibake filter for training (recorded in every manifest)',
    )
    ap.add_argument('--llm-tag', default='qwen2.5:7b-instruct-q4_K_M')
    ap.add_argument(
        '--train-batch',
        type=int,
        default=None,
        help="override train.py's card-sized default",
    )
    ap.add_argument(
        '--doc-batch',
        type=int,
        default=16,
        help='generation batch size for evaluate_document.py',
    )
    ap.add_argument(
        '--no-preflight',
        action='store_true',
        help='skip the correctness gate (not recommended; see preflight.py)',
    )
    args = ap.parse_args()

    ensure_dirs()
    stages = build_stages(args)
    if not args.stretch:
        stages = [s for s in stages if not s.stretch]

    if args.list:
        state = load_state()['stages']
        print(f"\n{'stage':<20} {'group':<10} {'est':<10} {'status':<12} about")
        print('-' * 110)
        for s in stages:
            status = (state.get(s.name) or {}).get('status', '-')
            print(
                f'{s.name:<20} {s.group:<10} {s.estimate:<10} {status:<12} {s.about[:60]}...'
            )
        gpu_estimate = [s.estimate for s in stages if s.gpu]
        print(f'\n{len(stages)} stages, {len(gpu_estimate)} of them GPU-bound.')
        print('Full session including the stretch experiment: roughly 15-25 GPU hours.')
        return

    selected = stages
    if args.only:
        selected = [s for s in stages if s.name == args.only or s.group == args.only]
        if not selected:
            raise SystemExit(f'no stage or group named {args.only!r}')
    if args.start:
        names = [s.name for s in stages]
        if args.start not in names:
            raise SystemExit(f'no stage named {args.start!r}')
        selected = stages[names.index(args.start) :]
    skip = {n.strip() for n in args.skip.split(',') if n.strip()}
    selected = [s for s in selected if s.name not in skip]

    if not args.no_preflight and not args.dry_run:
        print('running preflight gate')
        gate = subprocess.run(
            [PY, f'{HHU}/preflight.py', '--stage', 'all'], cwd=str(RESEARCH_DIR)
        )
        if gate.returncode != 0:
            raise SystemExit(
                '\nPreflight blocked the session. Each failing check would change a number '
                'produced here. Fix them, or pass --no-preflight if you have decided '
                'otherwise deliberately.'
            )

    failed: List[str] = []
    for stage in selected:
        if not args.force and stage_status(stage.name) == 'done':
            print(f'  skip {stage.name} (already done; --force to rerun)')
            continue
        if not run_stage(stage, args):
            failed.append(stage.name)
            if not args.keep_going:
                raise SystemExit(
                    f'\nStopped at {stage.name}. Fix it, then resume with:\n'
                    f'    python {HHU}/pipeline.py --from {stage.name}'
                )

    print('\n' + '=' * 60)
    if failed:
        print(f"finished with failures: {', '.join(failed)}")
    else:
        print('all selected stages completed')
    print(f'artifacts: {GPU_RESULTS.relative_to(RESEARCH_DIR.parent)}')
    print("Remember: the motd asks you to delete large files when you're done ->")
    print(f'    python {HHU}/cleanup.py --dry-run')


if __name__ == '__main__':
    main()
