"""Assemble every artifact this session produced into one reviewable file.

Writes `results/remote/RESULTS_GPU.md` in the same shape as the committed
`research/results/RESULTS.md`, so the two can be diffed and the new rows folded in by hand
rather than by a script -- deciding which number supersedes which is an editorial judgement
and RESULTS.md's own history shows why it should stay one (a row was retired there because
its BLEU pair turned out not to reproduce, which no merge tool would have caught).

Every row carries its artifact path and the host that produced it. That is the whole point of
the session: the existing table mixes an Apple Silicon MPS run, a Colab T4, and CPU
inference on an M3, and several §6.6 caveats exist only because of that mixing.

    python remote/collect_results.py            # from research/
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import GPU_RESULTS, load_state, read_json, rel


def newest(pattern: str) -> Optional[Path]:
    matches = sorted(
        glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True
    )
    return Path(matches[0]) if matches else None


def host_line(artifact: Optional[Dict[str, Any]]) -> str:
    if not artifact:
        return '—'
    prov = artifact.get('provenance') or {}
    host = prov.get('host') or {}
    # evaluate_document.py predates `provenance` and records under `run` instead
    run = artifact.get('run') or {}
    bits = [
        host.get('hostname'),
        host.get('gpu_name') or host.get('device_name'),
        f"torch {host.get('torch')}" if host.get('torch') else None,
        f"driver {host.get('nvidia_driver')}" if host.get('nvidia_driver') else None,
        (
            f"commit {prov.get('git_commit') or run.get('git_commit')}"
            if (prov.get('git_commit') or run.get('git_commit'))
            else None
        ),
        'DIRTY TREE' if prov.get('git_dirty') else None,
    ]
    return ' · '.join(b for b in bits if b) or '—'


def metric(scores: Dict[str, Any], *names: str) -> Any:
    """First present key among `names`.

    Artifacts disagree on spelling (evaluate_document.py: "D-SARI", sentence summary:
    "sari"). Explicit None test so a legitimate 0.0 does not fall through.
    """
    for name in names:
        if scores.get(name) is not None:
            return scores[name]
    return None


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return '—'
    if isinstance(value, (int, float)):
        return f'{value:.{digits}f}'
    return str(value)


def sentence_section() -> List[str]:
    summary_path = GPU_RESULTS / 'sentence_eval' / 'summary.json'
    data = read_json(summary_path)
    # counted, not hard-coded: the run grew a fifth condition
    n_conditions = len((data or {}).get('results') or {})
    heading = '## 1. Sentence level — ASSET test'
    lines = [f'{heading} ({n_conditions} conditions)' if n_conditions else heading, '']
    if not data:
        lines += ['_Not produced this session (`--only sentence` to run it)._', '']
        return lines

    run = data.get('run', {})
    lines += [
        f"Benchmark: {data.get('dataset')} · n={data.get('n_sentences')} · "
        f"{data.get('references_per_sentence')} references · "
        f"decoding {(data.get('seq2seq_decoding') or {}).get('strategy')} · "
        f"few-shot {data.get('fewshot_n')} · "
        f"{data.get('seeds_per_sampled_condition')} seeds per sampled condition · "
        f"significance baseline `{data.get('significance_baseline')}`",
        f"Host: {host_line(data)} · device {run.get('device')} · "
        f"{run.get('duration_seconds', 0) / 60:.0f} min",
        f'Artifact: `{rel(summary_path)}`',
        '',
        '| Condition | Model | SARI ↑ | BLEU | FKGL ↓ | BERTScore | unchanged | mean words | SARI sd (seeds) |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for key, block in (data.get('results') or {}).items():
        m = block.get('metrics', {})
        bert = m.get('bertscore_f1')
        # evaluate_sentence.py stores BERTScore on 0-1, evaluate_document.py on 0-100
        # (flagged in RESULTS.md)
        if isinstance(bert, float) and bert <= 1.0:
            bert *= 100
        lines.append(
            f"| `{key}` | {block.get('model_id')} | **{fmt(m.get('sari'))}** | "
            f"{fmt(m.get('bleu'))} | {fmt(m.get('fkgl'))} | {fmt(bert)} | "
            f"{fmt(m.get('unchanged_rate'), 3)} | {fmt(m.get('mean_words'), 1)} | "
            f"{fmt(m.get('sari_std'), 3)} |"
        )
    lines.append('')

    # evaluate_sentence.py records the bootstrap per condition under `vs_baseline` and
    # leaves top-level `significance` null
    sig = data.get('significance') or {
        key: block['vs_baseline']
        for key, block in (data.get('results') or {}).items()
        if isinstance(block, dict) and block.get('vs_baseline')
    }
    if sig:
        lines += [
            '**Paired bootstrap over sentence indices** (SARI, vs the significance baseline):',
            '',
        ]
        for key, block in sig.items():
            lines.append(
                f"- `{key}`: ΔSARI {fmt(block.get('mean_delta_sari'))} "
                f"95% CI [{fmt(block.get('ci_low'))}, {fmt(block.get('ci_high'))}], "
                f"p={block.get('p_value')} over {metric(block, 'n_resamples', 'resamples')} resamples"
            )
        lines += [
            '',
            '> Report the CI, not the p-value, as the substantive claim: 1000 resamples '
            'cannot resolve below 1/1000, so a p of 0.0 is `p < 0.001`.',
            '',
        ]

    reasons = {
        key: block.get('reason_counts')
        for key, block in (data.get('results') or {}).items()
        if block.get('reason_counts')
    }
    if reasons:
        lines += [
            '**Output-guard rejections per condition** — the first measured guard rates the '
            'project has; §6.4 currently states that no filter has one.',
            '',
            '```json',
            json.dumps(reasons, indent=2),
            '```',
            '',
        ]
    return lines


def document_section(label: str, subdir: str) -> List[str]:
    directory = GPU_RESULTS / subdir
    step = newest(str(directory / 'step4b_dwikipedia_*.json'))
    lines = [f'## {label}', '']
    if not step:
        lines += ['_Not produced this session._', '']
        return lines

    data = read_json(step) or {}
    dataset = data.get('dataset', {})
    lines += [
        f"Benchmark: D-Wikipedia test · n={dataset.get('n_evaluated')} · "
        f"{dataset.get('references_per_document')} reference/document · seed "
        f"{(dataset.get('sample_seed') if dataset.get('sampled') else 'full split (unsampled)')}",
        f'Host: {host_line(data)}',
        f'Artifact: `{rel(step)}`',
        '',
        '| System | D-SARI ↑ | SARI ↑ | BLEU | FKGL ↓ | BERTScore | LENS |',
        '|---|---|---|---|---|---|---|',
    ]

    lens_artifact = newest(str(directory / 'lens_n*_seed*.json'))
    lens_scores = (
        ((read_json(lens_artifact) or {}).get('scores') or {}) if lens_artifact else {}
    )

    systems = data.get('scores') or data.get('systems') or data.get('results') or {}
    for name, scores in systems.items():
        if not isinstance(scores, dict):
            continue
        # LENS: prefer the lens stage's artifact; the eval artifact may hold a null
        lens_value = lens_scores.get(name)
        if lens_value is None:
            lens_value = metric(scores, 'LENS', 'lens')
        lines.append(
            f"| {name} | **{fmt(metric(scores, 'D-SARI', 'D_SARI', 'd_sari'))}** | "
            f"{fmt(metric(scores, 'SARI', 'sari'))} | "
            f"{fmt(metric(scores, 'BLEU', 'bleu'))} | "
            f"{fmt(metric(scores, 'FKGL', 'fkgl'))} | "
            f"{fmt(metric(scores, 'BERTScore', 'bertscore'))} | "
            f'{fmt(lens_value)} |'
        )
    lines.append('')

    if lens_artifact:
        lines += [
            f'LENS artifact: `{rel(lens_artifact)}` — descriptive corpus means, not '
            f"significance-tested. **This is the project's first LENS column.**",
            '',
        ]
    else:
        lines += [
            '> LENS absent. It is the only metric here trained on human judgements, and the '
            'only reason it has never been reported is a dependency conflict a second venv '
            'solves: `bash remote/setup.sh --lens`, then the `lens` stage.',
            '',
        ]

    # evaluate_document.py: `resamples`; sentence summary: `n_resamples`
    def resamples(block: Dict[str, Any]) -> Any:
        return metric(block, 'n_resamples', 'resamples')

    for block in (
        data.get('significance', [])
        if isinstance(data.get('significance'), list)
        else []
    ):
        lines.append(
            f"- {block.get('comparison', 'significance')}: p={block.get('p_value')} "
            f"over {resamples(block)} resamples ({block.get('metric')})"
        )
    if isinstance(data.get('significance'), dict):
        sig = data['significance']
        lines.append(
            f"- finetuned vs baseline ({sig.get('metric', 'd_sari')}, {sig.get('test')}): "
            f"p={sig.get('p_value')} over {resamples(sig)} resamples"
        )
    # prompted-LLM comparisons live in their own block
    sig_llm = data.get('significance_llm')
    if isinstance(sig_llm, dict):
        lines.append(
            f"- llm vs baseline ({sig_llm.get('metric', 'd_sari')}): "
            f"p={sig_llm.get('llm_vs_baseline_p')} · llm vs finetuned: "
            f"p={sig_llm.get('llm_vs_finetuned_p')} over {resamples(sig_llm)} resamples"
        )
    lines.append('')

    review = newest(str(directory / 'qualitative_review*.json'))
    if review:
        lines += [
            f'Qualitative review: `{rel(review)}` — rerun against these generations, so '
            f"§6.3's taxonomy describes the checkpoint this table reports.",
            '',
        ]
    return lines


def audits_section() -> List[str]:
    lines = ['## Audits and threats to validity', '']

    overlap = read_json(GPU_RESULTS / 'overlap_audit.json')
    if overlap:
        lines += [
            '### Train/eval overlap',
            '',
            f"Artifact: `{rel(GPU_RESULTS / 'overlap_audit.json')}` · "
            f"method: {(overlap.get('method') or {}).get('matching')}",
            '',
            '| Evaluation set | Training set | Overlap | % |',
            '|---|---|---|---|',
        ]
        for c in overlap.get('comparisons', []):
            lines.append(
                f"| `{c['left']}` | `{c['right']}` | {c['overlapping_items']} / "
                f"{c['left_items']} | {c['overlap_pct_of_left']}% |"
            )
        leaked = [
            f['asset_validation_index']
            for f in overlap.get('fewshot_demonstrations', [])
            if f.get('in_wikilarge_train')
        ]
        lines += [
            '',
            f"Few-shot demonstrations found in training data: **{leaked or 'none'}**",
            '',
            '> Every figure is a floor: exact matching after normalisation, no fuzzy '
            'matching. The validation-split overlap is the one that touches a reported '
            'quantity — checkpoint selection runs on validation loss.',
            '',
        ]

    filters = read_json(GPU_RESULTS / 'corpus_filter_divergence.json')
    if filters:
        lines += [
            '### Corpus cleaning: which filter, and what it costs',
            '',
            f"Artifact: `{rel(GPU_RESULTS / 'corpus_filter_divergence.json')}`",
            '',
            '| Split | Raw | Dropped (strict) | Dropped (targeted) | Only strict | Kept (strict) |',
            '|---|---|---|---|---|---|',
        ]
        for split, s in (filters.get('splits') or {}).items():
            lines.append(
                f"| {split} | {s['raw_rows']} | {s['dropped_strict']} | "
                f"{s['dropped_targeted']} | {s['dropped_only_by_strict']} | {s['kept_strict']} |"
            )
        lines += [
            '',
            '> The tracked `data/stats.json` and the data statement describe the strict '
            'filter; the notebook trained with the targeted one. Quote whichever the '
            'reported checkpoint used — every `train_*.json` manifest records it.',
            '',
        ]

    preflight = read_json(GPU_RESULTS / 'preflight.json')
    if preflight:
        failed = [
            c['name']
            for c in preflight.get('checks', [])
            if not c['ok'] and c['blocking']
        ]
        lines += [
            '### Preflight',
            '',
            f"Last run: {'passed' if preflight.get('passed') else 'BLOCKED'}"
            + (f" — failing: {', '.join(failed)}" if failed else ''),
            '',
        ]
    return lines


def training_section() -> List[str]:
    lines = ['## Training runs', '']
    manifests = sorted(glob.glob(str(GPU_RESULTS / 'train_*.json')))
    if not manifests:
        return lines + ['_None this session._', '']

    lines += [
        '| Run | Corpus rows | Filter | Epochs | Batch | Precision | Best val loss | Wall clock |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for path in manifests:
        data = read_json(Path(path)) or {}
        run = data.get('run', {})
        corpus = data.get('corpus', {})
        hp = data.get('hyperparameters', {})
        sel = data.get('selection', {})
        train_rows = ((corpus.get('split_stats') or {}).get('train') or {}).get(
            'kept_rows'
        )
        lines.append(
            f"| `{run.get('tag')}` | {train_rows} | {corpus.get('mojibake_filter')} | "
            f"{hp.get('epochs')} | {hp.get('effective_batch_size')} | {hp.get('precision')} | "
            f"{fmt(sel.get('best_metric'), 4)} | {run.get('elapsed_human')} |"
        )
    lines += [
        '',
        '> `Corpus rows` is what the checkpoint actually trained on — cite this, not '
        '`data/stats.json`, which was produced by a different script and possibly a '
        'different filter.',
        '',
    ]
    for path in manifests:
        data = read_json(Path(path)) or {}
        run, sel = data.get('run', {}), data.get('selection', {})
        lines.append(
            f"- `{run.get('tag')}`: {host_line(data)} · selected "
            f"`{sel.get('best_model_checkpoint')}` at step {sel.get('global_step')} "
            f"(epoch {fmt(sel.get('epoch'), 2)}) · manifest `{rel(Path(path))}`"
        )
    lines.append('')
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    state = load_state()['stages']
    done = [name for name, block in state.items() if block.get('status') == 'done']
    failed = [name for name, block in state.items() if block.get('status') == 'failed']

    lines = [
        '# HHU GPU session results',
        '',
        'Produced by `research/remote/pipeline.py`. Every table below names the host, the '
        'driver, the commit and the artifact that produced it.',
        '',
        '**Read this against `research/results/RESULTS.md` and fold rows in by hand.** '
        "Deciding which number supersedes which is editorial — RESULTS.md's own history "
        'includes retiring a row whose BLEU pair turned out not to reproduce, and no '
        'automatic merge would have caught that.',
        '',
        f"Stages completed: {', '.join(sorted(done)) or 'none'}"
        + (f" · **failed: {', '.join(sorted(failed))}**" if failed else ''),
        '',
        '---',
        '',
    ]
    lines += sentence_section()
    lines += ['---', '']
    # the full-split run (n=8000) is the headline document result
    lines += document_section(
        '2. Document level — D-Wikipedia test, FULL split (n=8000)',
        'document_eval_full',
    )
    lines += ['---', '']
    lines += document_section(
        '2b. Document level — three-way comparison incl. prompted LLM (n=2000)',
        'document_eval',
    )
    lines += ['---', '']
    lines += document_section(
        '2c. Normaliser isolation — the previous checkpoint, re-scored under the fixed '
        'normaliser (n=500)',
        'normaliser_isolation',
    )
    lines += ['---', '']
    lines += document_section(
        '2d. Document level, natural-case checkpoint (stretch experiment)',
        'document_eval_naturalcase',
    )
    lines += ['---', '']
    lines += training_section()
    lines += ['---', '']
    lines += audits_section()
    lines += [
        '---',
        '',
        '## What still is not measured',
        '',
        'Kept explicit so the gap does not close by implication:',
        '',
        "- **RQ2's site-sample study.** No GPU is needed for it and no GPU run substitutes "
        'for it: filter false-positive/false-negative rates over a real page sample remain '
        'the single largest evidence gap in the thesis.',
        '- **Human evaluation** — scoped out to Future Work (§7.2), not pending.',
        '- **Cohesion measurement** — scoped out to Future Work (§7.6). If a spare hour '
        'appears, the paired document outputs on disk are enough to compute TAACO-style '
        'indices with no generation and no GPU.',
        '- **Additional seeds for the document LLM condition.** BLESS aggregates three; the '
        'document table still runs one, so one row of it carries sampling variance the '
        'deterministic beam-search rows do not.',
        '',
    ]

    out = Path(args.out) if args.out else GPU_RESULTS / 'RESULTS_GPU.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines))
    print(f'wrote {rel(out)}')


if __name__ == '__main__':
    main()
