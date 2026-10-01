"""Phase 5: evaluate every simplification method on the same ASSET test split.

    python evaluate_sentence.py --limit 10                  # smoke run
    python evaluate_sentence.py                             # full run, all conditions
    python evaluate_sentence.py --conditions llm_7b --seeds 3

    # Row S3 in results/RESULTS.md -- "does fine-tuning beat the un-fine-tuned
    # checkpoint?", at the beam-search decoding every existing row was measured with:
    python evaluate_sentence.py --conditions base,local --decoding beam \
        --significance-baseline base --outdir results/sentence_eval

    # The held-out-remainder check (results/RESULTS.md 2e): drop the ASSET test sentences
    # an overlap audit found in the training data and re-score the remainder from the
    # cached generations of a full-split run. Reads caches, never generates.
    python evaluate_sentence.py --cache-dir results/remote/sentence_eval \
        --outdir results/asset_heldout --decoding beam \
        --exclude-indices 60,135,137,141,144,211,228,357

Produces, per condition: SARI (simplicity), BLEU, FKGL (readability), BERTScore
(meaning preservation), a copy of every system output, and -- for the prompted-LLM
conditions -- a count of each rejection reason code, which is the raw material for the
error categorisation in the thesis §6.3.

Design notes:

- **One script, all methods.** The point of Phase 5 is a like-for-like comparison, so
  every condition goes through the same loader, the same metric calls, and the same
  ASSET references. Scoring code that differs per method is scoring that can't be
  compared.
- **Generation is cached to disk** (`--outdir`). Re-scoring, adding a metric, or
  running the significance tests never re-runs a model; a 7B over 359 sentences is
  ~11 minutes per seed and shouldn't be repeated to add a column.
- **BERTScore is not optional.** SARI alone cannot separate "simplified well" from
  "deleted half the sentence", and the simplicity/meaning trade-off is BLESS's central
  finding.
- **FKGL is reported but never headline.** BLESS §5 found the top-5 models *by FKGL*
  had a 36% hallucination rate -- it is trivially gamed by degeneration.
- **Significance via paired bootstrap** over sentence indices, recomputing corpus SARI
  on each resample, matching the test already used for the WikiLarge comparison.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

# prompting.py lives in ../backend because the BLESS templates are used at serve time
# too; imported rather than copied so the prompted-LLM conditions are scored with
# byte-identical prompts. research/ -> backend/ is the only allowed direction: the
# service must never import from here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))
import prompting  # noqa: E402
from hf_revisions import HF_REVISIONS  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')

SCRIPT_VERSION = '1.1'

# Decoding presets answer different questions:
#   "checkpoint_default": what backend/main.py serves (it passes only max_length), i.e.
#   what an extension user gets.
#   "beam": what every previously reported number used (row S3 in results/RESULTS.md and
#   all document-level results), so comparable with the existing table.
# The preset used is recorded in the output: an unlabelled SARI is not citable.
#
# The preset formerly called "greedy" was never greedy (found 2026-08-24): an empty dict
# defers to the checkpoint's `generation_config`, and every BART checkpoint here ships
# `num_beams=4`. Verified: `{}` is byte-identical to explicit `num_beams=4` and differs
# from `num_beams=1`. So the old "greedy" rows are 4-beam rows, and S4 and S5 (0.11 SARI
# apart) differ only by S4's two penalty terms. The name is retired, not redefined:
# `--decoding greedy` fails instead of meaning something other than the S5 run.

# model_id -> decoding parameters actually in force, filled by generate_seq2seq as each
# checkpoint loads. Empty on a fully cached run (stamped "unknown", not guessed).
EFFECTIVE_DECODING: Dict[str, Dict[str, Any]] = {}

DECODING_PRESETS: Dict[str, Dict[str, object]] = {
    # Deployed configuration (main.py passes only max_length): 4-beam for these
    # checkpoints, inherited rather than chosen.
    'checkpoint_default': {},
    'beam': {'num_beams': 4, 'no_repeat_ngram_size': 3, 'repetition_penalty': 1.2},
    # Actually greedy: one beam, no penalties.
    'true_greedy': {'num_beams': 1},
}

# The four conditions from docs/roadmap-open-llm.md Phase 5, plus the fast LLM.
# kind: "seq2seq" -> a local/HF checkpoint;  "ollama" -> a prompted open LLM.
CONDITIONS: Dict[str, Dict[str, str]] = {
    'base': {
        'kind': 'seq2seq',
        'model_id': 'facebook/bart-base',
        'label': 'BART-base zero-shot (RQ1 baseline)',
    },
    'online': {
        'kind': 'seq2seq',
        'model_id': 'eilamc14/bart-large-text-simplification',
        'label': 'BART off-the-shelf (baseline)',
    },
    # Named "local" to match result files under results/sentence_eval/ (filenames, JSON
    # keys). The backend key was renamed "local" -> "finetuned" on 2026-08-17; renaming
    # this id would orphan those artifacts and the results/RESULTS.md rows citing them.
    'local': {
        'kind': 'seq2seq',
        # Same default as main.py's MODEL_ENV_CONFIG; a local path such as
        # scratch/simplification_results/best_checkpoint scores a local training run.
        'model_id': 'yunvs/bart-base-wikilarge-simplification',
        'label': 'BART fine-tuned (this project)',
    },
    'llm_7b': {
        'kind': 'ollama',
        'model_id': 'qwen2.5:7b-instruct-q4_K_M',
        'label': 'Qwen2.5-7B + BLESS Prompt 2',
    },
    'llm_3b': {
        'kind': 'ollama',
        'model_id': 'qwen2.5:3b-instruct-q4_K_M',
        'label': 'Qwen2.5-3B + BLESS Prompt 2',
    },
    # Two off-the-shelf baselines, not to be blurred (thesis §6.2): `base` is the
    # un-fine-tuned starting checkpoint, needed for RQ1's "does fine-tuning help?" and
    # used for row S3 in results/RESULTS.md; `online` is the larger third-party
    # fine-tune the extension ships as default, a harder, separate comparison.
    #
    # The "document" checkpoint is deliberately absent: ASSET has sentence-level
    # references, and that model deletes, merges and reorders across sentence
    # boundaries. It needs a document-level benchmark (D-Wikipedia's test split) and its
    # own table.
}


# --- data ---


def load_asset_test(limit: Optional[int] = None) -> Tuple[List[str], List[List[str]]]:
    """ASSET test split: 359 sentences, 10 human references each.

    Returns (sources, references) with references shaped (n_samples, n_refs) --
    the metric calls below transpose where they need the other orientation.
    """
    from datasets import load_dataset

    # cast: load_dataset's return type is a union over (split, streaming), and
    # Dataset.__iter__ is inferred as yielding lists too. With `split=` this is one
    # Dataset of dict rows.
    ds = cast(
        Sequence[Dict[str, Any]],
        load_dataset(
            'facebook/asset',
            'simplification',
            split='test',
            revision=HF_REVISIONS.get('facebook/asset'),
        ),
    )
    sources = [row['original'] for row in ds]
    references = [list(row['simplifications']) for row in ds]
    if limit:
        sources, references = sources[:limit], references[:limit]
    logger.info(
        'Loaded ASSET test: %d sentences, %d refs each',
        len(sources),
        len(references[0]),
    )
    return sources, references


# --- generation ---


def resolve_device(requested: str = 'auto') -> str:
    """Pick the torch device for the seq2seq conditions.

    Defaults to CPU rather than auto-detecting Metal, which is the slower choice and
    deliberately so -- see the warning in generate_seq2seq. Pass --device mps to opt in.
    """
    if requested != 'auto':
        return requested
    return 'cpu'


def generate_seq2seq(
    model_id: str,
    sources: Sequence[str],
    batch_size: int = 16,
    device: str = 'cpu',
    decoding: str = 'checkpoint_default',
) -> List[str]:
    """Run a BART-style checkpoint over the sources.

    max_length=64 matches SEQ2SEQ_MAX_LENGTH["sentence"] in main.py, the length the
    sentence checkpoints were fine-tuned at, so evaluation matches serving.

    `decoding` selects a preset from DECODING_PRESETS: "checkpoint_default" (what the
    backend serves; 4-beam here, inherited from the checkpoint's generation_config),
    "beam" (that plus two penalty terms, used by the table's S3/S4 rows) or
    "true_greedy" (num_beams=1). What it resolved to is recorded in the output.

    Device: CPU by default, markedly slower than Metal on Apple Silicon (bart-large over
    ASSET's 359 sentences: 1526 s, 4.25 s/sentence, in the 2026-08-15 run). `--device mps`
    is much faster, but MPS and CPU are not bit-identical, so generations can differ and
    a table mixing devices is not internally comparable. If you switch device, delete the
    cached generations and regenerate *every* condition.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if decoding not in DECODING_PRESETS:
        raise ValueError(
            f'unknown decoding {decoding!r}; expected one of {list(DECODING_PRESETS)}'
        )
    gen_kwargs = {'max_length': 64, **DECODING_PRESETS[decoding]}

    logger.info(
        'Loading seq2seq model %s on %s (%s decoding)', model_id, device, decoding
    )
    revision = HF_REVISIONS.get(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    model.eval()
    model.to(device)

    # Actual decoding: preset kwargs layered over the checkpoint's generation_config.
    # Recorded per model because it depends on the checkpoint, not only the flag.
    effective = {
        k: getattr(model.generation_config, k, None)
        for k in (
            'num_beams',
            'do_sample',
            'no_repeat_ngram_size',
            'repetition_penalty',
        )
    }
    effective.update(gen_kwargs)
    EFFECTIVE_DECODING[model_id] = effective
    logger.info('  effective decoding for %s: %s', model_id, effective)

    outputs: List[str] = []
    for start in range(0, len(sources), batch_size):
        batch = list(sources[start : start + batch_size])
        inputs = tokenizer(
            batch, return_tensors='pt', truncation=True, padding=True, max_length=64
        ).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        outputs.extend(tokenizer.decode(g, skip_special_tokens=True) for g in generated)
        logger.info('  %d/%d', min(start + batch_size, len(sources)), len(sources))

    # Free weights before the next condition loads: on a 16 GB machine with Ollama
    # holding ~4.7 GB, two resident large checkpoints got the process OOM-killed
    # (observed 2026-08-15).
    del model
    if device == 'mps':
        torch.mps.empty_cache()

    return [prompting.tidy_punctuation(o.strip()) for o in outputs]


def generate_ollama(
    model_id: str, sources: Sequence[str], seed: Optional[int] = None
) -> Tuple[List[str], Dict[str, int]]:
    """Prompt an Ollama-served model once per sentence with BLESS Prompt 2.

    Uses EVALUATION_DECODING (nucleus p=0.9, temperature 1.0, 100 tokens -- BLESS
    §3.4), *not* the greedy config the API serves with, so these numbers are
    comparable to the paper rather than to the extension.

    Returns (outputs, reason_counts). Rejected generations fall back to the source,
    exactly as the live backend does, so the score reflects what a user would see.
    """
    import urllib.request

    options = dict(prompting.EVALUATION_DECODING)
    if seed is not None:
        options['seed'] = seed

    outputs: List[str] = []
    reasons: Dict[str, int] = {}
    started = time.monotonic()

    for i, source in enumerate(sources):
        payload = {
            'model': model_id,
            'prompt': prompting.build_prompt(source),
            'stream': False,
            'options': options,
            # int, not "-1": a string keep_alive is parsed as a Go duration and 400s.
            'keep_alive': -1,
        }
        request = urllib.request.Request(
            f'{OLLAMA_URL}/api/generate',
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = json.loads(response.read()).get('response', '')

        final, reason, _model_result = prompting.sanitize(source, raw)
        outputs.append(final)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1

        if (i + 1) % 25 == 0 or i + 1 == len(sources):
            rate = (time.monotonic() - started) / (i + 1)
            logger.info('  %d/%d  (%.2fs/sentence)', i + 1, len(sources), rate)

    return outputs, reasons


# --- metrics ---


def compute_metrics(
    sources: Sequence[str],
    outputs: Sequence[str],
    references: Sequence[Sequence[str]],
    with_bertscore: bool = True,
) -> Dict[str, float]:
    """SARI / BLEU / FKGL / BERTScore for one condition.

    easse wants references as (n_references, n_samples), ASSET gives (n_samples,
    n_references), hence the transpose. Backwards, it silently yields plausible but
    meaningless numbers (a real bug in an earlier harness; thesis §4.7).
    """
    from easse.bleu import corpus_bleu
    from easse.fkgl import corpus_fkgl
    from easse.sari import corpus_sari

    refs_transposed = [list(r) for r in zip(*references, strict=True)]

    metrics = {
        'sari': float(
            corpus_sari(
                orig_sents=list(sources),
                sys_sents=list(outputs),
                refs_sents=refs_transposed,
            )
        ),
        'bleu': float(corpus_bleu(sys_sents=list(outputs), refs_sents=refs_transposed)),
        'fkgl': float(corpus_fkgl(sentences=list(outputs))),
    }

    if with_bertscore:
        import bert_score

        # multi-reference: bert_score takes the best-matching reference per candidate
        # With return_hash=False score() returns (P, R, F), but its declared return
        # type also admits (triple, hash); cast to pin the unpack.
        precision, recall, f1 = cast(
            Tuple,
            bert_score.score(
                list(outputs),
                [list(r) for r in references],
                lang='en',
                verbose=False,
            ),
        )
        metrics['bertscore_f1'] = float(f1.mean())

    # Descriptive, not scored: how much the system actually changed the input.
    unchanged = sum(
        1 for s, o in zip(sources, outputs, strict=True) if s.strip() == o.strip()
    )
    metrics['unchanged_rate'] = unchanged / len(sources)
    metrics['mean_words'] = float(np.mean([len(o.split()) for o in outputs]))
    return metrics


def paired_bootstrap_sari(
    sources: Sequence[str],
    outputs_a: Sequence[str],
    outputs_b: Sequence[str],
    references: Sequence[Sequence[str]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Is A's corpus SARI genuinely above B's, or is the gap noise?

    Resamples sentence indices with replacement and recomputes corpus SARI for both
    systems on the same resample -- paired, so the two systems always see identical
    sentences. p is the fraction of resamples where A does not beat B.

    Do not "optimise" this into resampling precomputed per-sentence SARI.
    evaluate_document.py does that for D-SARI, where it is exact because corpus D-SARI is
    the mean of per-document scores. Corpus SARI is not: easse aggregates n-gram
    statistics before scoring. Measured on 40 ASSET sentences of the fine-tuned outputs:
    corpus 40.7587 vs. mean-of-per-sentence 38.6801 (2.08 points). Recomputing per
    resample is why this is the slow part of a run.
    """
    from easse.sari import corpus_sari

    rng = random.Random(seed)
    n = len(sources)
    wins = 0
    diffs = []

    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        src = [sources[i] for i in idx]
        refs = [list(r) for r in zip(*[references[i] for i in idx], strict=True)]
        sari_a = corpus_sari(
            orig_sents=src, sys_sents=[outputs_a[i] for i in idx], refs_sents=refs
        )
        sari_b = corpus_sari(
            orig_sents=src, sys_sents=[outputs_b[i] for i in idx], refs_sents=refs
        )
        diffs.append(sari_a - sari_b)
        if sari_a > sari_b:
            wins += 1

    return {
        'mean_delta_sari': float(np.mean(diffs)),
        'ci_low': float(np.percentile(diffs, 2.5)),
        'ci_high': float(np.percentile(diffs, 97.5)),
        'p_value': 1.0 - wins / n_resamples,
        'n_resamples': n_resamples,
    }


# --- driver ---


def _provenance_helpers():
    """Borrow evaluate_document.py's provenance helpers instead of keeping a second copy.

    Two copies would drift. Imported lazily so this module stays importable on its own.
    """
    from evaluate_document import git_commit, resolve_model_provenance

    return git_commit, resolve_model_provenance


def cache_name(name: str, seed: int, decoding: str) -> str:
    """Cache filename for one (condition, seed) pair.

    The decoding strategy is part of the key for seq2seq conditions: beam and greedy
    produce different output from the same weights, and a cache that ignored the
    difference would serve one run's generations to the other. The prompted-LLM
    conditions don't take these presets, so their filenames are unchanged.
    """
    if CONDITIONS[name]['kind'] == 'seq2seq':
        return f'{name}_{decoding}_seed{seed}.json'
    return f'{name}_seed{seed}.json'


def run_condition(
    name: str,
    sources: Sequence[str],
    seeds: int,
    outdir: Path,
    device: str = 'cpu',
    decoding: str = 'checkpoint_default',
    allow_llm_generate: bool = False,
    require_cache: bool = False,
) -> List[Dict]:
    """Generate (or reload cached) outputs for one condition, once per seed.

    Prompted-LLM generations are **not** produced here unless `allow_llm_generate`: a
    missing seed would turn a scoring run into hours of slow single-stream inference,
    and `--seeds` defaults to 3 (BLESS §3.4) while `evaluate_llm.py --seeds` defaults
    to 1, so the mismatch is easy to hit. Missing caches raise with the exact
    `evaluate_llm.py` command that fills them (resumable, rate-limit tolerant,
    optionally remote).
    """
    config = CONDITIONS[name]
    runs = []
    # `require_cache` (set by --exclude-indices) must never generate: cache names
    # (condition, seed, decoding) don't record sentence coverage, so a subset run would
    # write a short file under the canonical name that later full-split rescores read as
    # complete. Caches stay whole-split; subsetting happens after.

    # seq2seq decoding is deterministic (do_sample=False), so extra seeds would copy the
    # first; only the sampled LLM conditions repeat.
    n_runs = seeds if config['kind'] == 'ollama' else 1

    for seed in range(1, n_runs + 1):
        cache = outdir / cache_name(name, seed, decoding)
        if cache.exists():
            logger.info('Reusing cached generations: %s', cache)
            runs.append(json.loads(cache.read_text()))
            continue

        if require_cache:
            raise SystemExit(
                f"\nMissing generations for '{name}' seed {seed}: {cache.name}\n\n"
                f'--exclude-indices scores a subset of a *cached* full-split run and will '
                f'not generate: a partial cache written under this name would be '
                f'indistinguishable from a complete one.\n'
                f'Score the full split first (or point --cache-dir at the run that did), '
                f'then re-run with the exclusions.\n'
            )

        if config['kind'] == 'ollama' and not allow_llm_generate:
            raise SystemExit(
                f"\nMissing generations for '{name}' seed {seed}: {cache.name}\n\n"
                f'Generate them first (resumable, and can run against a remote endpoint):\n'
                f'  python evaluate_llm.py --condition {name} '
                f"--model {config['model_id']} --seeds {seeds} --outdir {outdir}\n\n"
                f'Then re-run this command. Or pass --allow-llm-generate to generate '
                f'inline here instead, which is single-stream, not resumable, and takes '
                f'roughly 12-60 min per seed per model on this hardware.\n'
                f'If you only meant to score one seed, pass --seeds 1.\n'
            )

        logger.info('=== %s (seed %d) ===', config['label'], seed)
        started = time.monotonic()
        if config['kind'] == 'seq2seq':
            outputs = generate_seq2seq(
                config['model_id'], sources, device=device, decoding=decoding
            )
            reasons = {}
        else:
            outputs, reasons = generate_ollama(config['model_id'], sources, seed=seed)
        elapsed = time.monotonic() - started

        run = {
            'condition': name,
            'label': config['label'],
            'model_id': config['model_id'],
            'decoding': decoding if config['kind'] == 'seq2seq' else 'sampling',
            'seed': seed,
            'outputs': outputs,
            'reason_counts': reasons,
            'seconds_total': elapsed,
            'seconds_per_sentence': elapsed / len(sources),
        }
        cache.write_text(json.dumps(run, indent=2))
        runs.append(run)

    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--conditions',
        default=','.join(CONDITIONS),
        help='comma-separated subset of: ' + ', '.join(CONDITIONS),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='only the first N ASSET sentences (smoke runs)',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=3,
        help='generation runs per sampled condition (BLESS uses 3)',
    )
    parser.add_argument('--outdir', default='scratch/eval_methods')
    parser.add_argument(
        '--cache-dir',
        default=None,
        metavar='DIR',
        help='read cached generations from here instead of --outdir. Lets a '
        'derived scoring pass (see --exclude-indices) write its result '
        'somewhere new without copying the generations it reuses, and '
        "without overwriting the source run's summary.json.",
    )
    parser.add_argument(
        '--exclude-indices',
        default=None,
        metavar='I,J,...',
        help='0-based positions in the ASSET test split to drop before '
        'scoring, e.g. the sentences an overlap audit found in the '
        'training data (`overlapping_indices` in '
        'results/remote/overlap_audit*.json). Scores the remainder from '
        'the cached full-split generations and never generates; the '
        'indices are recorded in the output so the subset is '
        'reproducible.',
    )
    parser.add_argument(
        '--exclusion-note',
        default=None,
        metavar='TEXT',
        help='recorded beside --exclude-indices: where the list came from. '
        'An unexplained index list is not a citable exclusion.',
    )
    parser.add_argument(
        '--no-bertscore',
        action='store_true',
        help='skip BERTScore (downloads ~1.4GB on first use)',
    )
    parser.add_argument(
        '--bootstrap',
        type=int,
        default=1000,
        help='paired bootstrap resamples; 0 to skip',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='torch device for seq2seq conditions: auto (=cpu) | cpu | mps. '
        'Switching device invalidates cached generations -- see '
        "generate_seq2seq's docstring.",
    )
    parser.add_argument(
        '--decoding',
        default='checkpoint_default',
        choices=sorted(DECODING_PRESETS),
        help='seq2seq decoding. checkpoint_default passes nothing and lets the '
        "checkpoint's generation_config decide, which is what "
        'backend/main.py serves and which is 4-beam for every checkpoint '
        "here (it was misnamed 'greedy' until 2026-08-24); beam is that "
        "plus the two penalty terms RESULTS.md's S3/S4 rows used; "
        'true_greedy is actually greedy, num_beams=1',
    )
    parser.add_argument(
        '--allow-llm-generate',
        action='store_true',
        help='generate missing prompted-LLM outputs inline instead of '
        'erroring. Off by default: inline generation is '
        'single-stream and not resumable -- prefer evaluate_llm.py, '
        'which is both and can also run remotely.',
    )
    parser.add_argument(
        '--significance-baseline',
        default='online',
        metavar='CONDITION',
        help="condition the others are bootstrapped against. Use 'base' for "
        "RQ1's does-fine-tuning-help comparison (row S3); 'online' for "
        'the harder comparison against the shipped default model.',
    )
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cachedir = Path(args.cache_dir) if args.cache_dir else outdir
    if not cachedir.is_dir():
        parser.error(f'--cache-dir {cachedir} does not exist')
    device = resolve_device(args.device)

    # Always the full split: caches are keyed by whole-split position, so indices must
    # mean the same as in the overlap audit that produced them.
    full_sources, full_references = load_asset_test(args.limit)
    excluded: List[int] = []
    if args.exclude_indices:
        try:
            excluded = sorted(
                {int(x) for x in args.exclude_indices.split(',') if x.strip()}
            )
        except ValueError:
            parser.error('--exclude-indices takes comma-separated integers')
        bad = [i for i in excluded if not 0 <= i < len(full_sources)]
        if bad:
            parser.error(
                f'--exclude-indices out of range for a {len(full_sources)}-sentence split: {bad}'
            )
    keep = [i for i in range(len(full_sources)) if i not in set(excluded)]
    sources = [full_sources[i] for i in keep]
    references = [full_references[i] for i in keep]
    if excluded:
        logger.info(
            'Excluding %d of %d sentences; scoring the %d-sentence remainder. Indices: %s',
            len(excluded),
            len(full_sources),
            len(sources),
            excluded,
        )
    names = [n.strip() for n in args.conditions.split(',') if n.strip()]
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        parser.error(f"unknown condition(s): {', '.join(unknown)}")
    if args.significance_baseline not in CONDITIONS:
        parser.error(f'unknown --significance-baseline: {args.significance_baseline}')

    results = {}
    for name in names:
        runs = run_condition(
            name,
            full_sources,
            args.seeds,
            cachedir,
            device=device,
            decoding=args.decoding,
            allow_llm_generate=args.allow_llm_generate,
            require_cache=bool(excluded),
        )

        # Cached runs cover the whole split; slice at the same positions for every
        # condition and seed so the remainder is comparable.
        for r in runs:
            if len(r['outputs']) != len(full_sources):
                raise SystemExit(
                    f"\nCached generations for '{name}' seed {r['seed']} have "
                    f"{len(r['outputs'])} outputs, expected {len(full_sources)}: "
                    f"{cache_name(name, r['seed'], args.decoding)}\n"
                    f'A cache that does not cover the split cannot be sliced by split '
                    f'position. Regenerate it, or drop --limit/--exclude-indices.\n'
                )
        outputs_by_run = [[r['outputs'][i] for i in keep] for r in runs]

        per_run = [
            compute_metrics(sources, outputs, references, not args.no_bertscore)
            for outputs in outputs_by_run
        ]
        # BLESS aggregates across seeds; with one run this is just that run.
        aggregated = {
            key: float(np.mean([m[key] for m in per_run])) for key in per_run[0]
        }
        if len(per_run) > 1:
            aggregated['sari_std'] = float(np.std([m['sari'] for m in per_run]))

        merged_reasons: Dict[str, int] = {}
        for r in runs:
            for reason, count in (r.get('reason_counts') or {}).items():
                merged_reasons[reason] = merged_reasons.get(reason, 0) + count

        results[name] = {
            'label': CONDITIONS[name]['label'],
            'model_id': CONDITIONS[name]['model_id'],
            'n_runs': len(runs),
            'metrics': aggregated,
            'reason_counts': merged_reasons,
            'seconds_per_sentence': float(
                np.mean([r['seconds_per_sentence'] for r in runs])
            ),
            # first run's outputs, for the qualitative pass in §6.3
            'sample_outputs': outputs_by_run[0][:10],
        }
        logger.info('%s -> %s', name, {k: round(v, 3) for k, v in aggregated.items()})

    # Significance vs. one named baseline, a research decision: 'base' tests "does
    # fine-tuning help?" (RQ1, row S3), 'online' tests "does it beat the shipped model?".
    # The output records which.
    ref_name = args.significance_baseline
    if args.bootstrap and ref_name in results:
        baseline_full = json.loads(
            (cachedir / cache_name(ref_name, 1, args.decoding)).read_text()
        )['outputs']
        baseline = [baseline_full[i] for i in keep]
        for name in names:
            if name == ref_name:
                continue
            first_full = json.loads(
                (cachedir / cache_name(name, 1, args.decoding)).read_text()
            )['outputs']
            first = [first_full[i] for i in keep]
            logger.info('Bootstrapping %s vs %s...', name, ref_name)
            results[name]['vs_baseline'] = {
                'baseline_condition': ref_name,
                'baseline_model_id': CONDITIONS[ref_name]['model_id'],
                **paired_bootstrap_sari(
                    sources, first, baseline, references, n_resamples=args.bootstrap
                ),
            }
    elif args.bootstrap:
        logger.warning(
            'Skipping significance tests: --significance-baseline %r is not among the '
            'conditions being run (%s)',
            ref_name,
            ', '.join(names),
        )

    # Provenance (as in evaluate_document.py): a SARI without date, commit and decoding
    # is not citable. Numbers in results/RESULTS.md predating this block are recorded as
    # "measured but not reproducible" (§4.8 there).
    git_commit, resolve_model_provenance = _provenance_helpers()
    finished_at = datetime.now(timezone.utc)
    summary = {
        'run': {
            'script': 'evaluate_sentence.py',
            'script_version': SCRIPT_VERSION,
            'step': '4/5',
            'started_at': started_at.isoformat(),
            'finished_at': finished_at.isoformat(),
            'duration_seconds': round((finished_at - started_at).total_seconds(), 1),
            'git_commit': git_commit(),
            'device': device,
        },
        'dataset': 'facebook/asset test',
        'n_sentences': len(sources),
        'excluded_indices': excluded,
        'n_excluded': len(excluded),
        'exclusion_note': args.exclusion_note,
        'generations_read_from': str(cachedir),
        'references_per_sentence': len(references[0]) if references else 0,
        'seeds_per_sampled_condition': args.seeds,
        # "preset", not "strategy": unset parameters defer to the checkpoint's
        # generation_config (how a 4-beam run was recorded as "greedy" for two months).
        # `effective_per_model` holds what generate_seq2seq actually ran.
        'seq2seq_decoding': {
            'preset': args.decoding,
            **DECODING_PRESETS[args.decoding],
            'max_length': 64,
            'effective_per_model': EFFECTIVE_DECODING or None,
        },
        'decoding_eval': prompting.EVALUATION_DECODING,
        'fewshot_n': prompting.FEWSHOT_N,
        'significance_baseline': ref_name,
        'models': {
            name: resolve_model_provenance(CONDITIONS[name]['model_id'])
            for name in names
        },
        'results': results,
    }

    stamp = started_at.strftime('%Y%m%dT%H%M%SZ')
    out_path = outdir / f'step4_asset_{stamp}_{args.decoding}_n{len(sources)}.json'
    out_path.write_text(json.dumps(summary, indent=2))
    # summary.json is the stable "latest run" path some notes refer to; cite the
    # timestamped file. Subset runs leave it alone so it keeps the full-split result.
    if excluded:
        logger.info('Subset run: leaving summary.json alone; cite %s', out_path.name)
    else:
        (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))

    print('\n' + '=' * 100)
    excl_note = f'  |  excluded {len(excluded)}: {excluded}' if excluded else ''
    print(
        f'ASSET test, n={len(sources)}  |  seq2seq decoding: {args.decoding}{excl_note}'
    )
    print(
        f"{'condition':<34} {'SARI':>7} {'BLEU':>7} {'FKGL':>7} {'BERTSc':>7} "
        f"{'unchg':>7} {'s/sent':>7}"
    )
    print('-' * 100)
    for name in names:
        m = results[name]['metrics']
        print(
            f"{results[name]['label']:<34} {m['sari']:>7.2f} {m['bleu']:>7.2f} "
            f"{m['fkgl']:>7.2f} {m.get('bertscore_f1', float('nan')):>7.3f} "
            f"{m['unchanged_rate']:>7.1%} {results[name]['seconds_per_sentence']:>7.2f}"
        )
    print('=' * 100)
    for name in names:
        boot = results[name].get('vs_baseline')
        if boot:
            print(
                f"{name} vs {boot['baseline_condition']}: "
                f"ΔSARI {boot['mean_delta_sari']:+.2f} "
                f"[{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]  p={boot['p_value']:.4f}"
            )
        if results[name]['reason_counts']:
            print(f"{name} rejections: {results[name]['reason_counts']}")
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
