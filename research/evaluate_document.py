"""Step 4B: document-level evaluation on D-Wikipedia's own test split.

    python evaluate_document.py --limit 20                 # smoke run
    python evaluate_document.py --limit 500                # the reportable run
    python evaluate_document.py --limit 500                # rerun: resumes where it stopped
    python evaluate_document.py --limit 500 --rescore      # re-score cached generations

Step 4 (evaluate_sentence.py) scores against ASSET, which is a *sentence*-level
benchmark. Running the document checkpoint there measures how a document-trained model
handles isolated single sentences -- not the task it was fine-tuned on, and not an
answer to RQ1's document-level branch. This script is the document-appropriate
counterpart: D-Wikipedia's own held-out test split, scored with D-SARI (designed for
document-length simplification), LENS and BERTScore, alongside the same BLEU/SARI/FKGL
Step 4 reports for continuity.

Extracted from notebooks/train_sentence_and_document_pipeline.ipynb's Step 4B cell so it can run against
an existing checkpoint without re-running training, and so a result carries provenance
(timestamp, model id, commit, sample seed) instead of living in a notebook's scroll
buffer. The metric implementations are ported unchanged from that notebook.

Four deliberate differences from the notebook cell, all of them about making the run
finish -- and, for the last one, about the result outliving the session that produced it:

- **Generation is batched.** The notebook generates one document at a time; at 8,000
  test documents x 2 systems x beam search that is many hours. Batching is a pure
  throughput change -- padding is left-independent for encoder-decoder generation and
  the decoded strings are identical.
- **The test split is sampled**, seeded and recorded, via --limit. The full split stays
  available (--limit 0), but a fixed 500-document sample makes the numbers reachable in
  minutes. n and the seed go in the output so the sample is reproducible.
- **The D-SARI bootstrap resamples precomputed per-document scores** rather than
  recomputing D-SARI from scratch on every resample. This is exact, not an
  approximation: corpus D-SARI is the plain mean of per-document scores, so resampling
  documents and re-averaging is the same arithmetic. The notebook's version recomputes
  n-gram counters ~1M times for n=500, which dominates the whole run.
- **Every stage is checkpointed to disk as it completes**, and a rerun resumes from
  wherever the last one stopped. Generation flushes partial predictions every
  --checkpoint-every batches; the result JSON is rewritten after each system is scored,
  carrying a run.status field so an interrupted file is recognisable as one. The
  motivating failure is mundane and repeatable: this runs on Colab, Colab kills the
  runtime when the browser disconnects, and everything held only in that runtime's
  filesystem goes with it. Point --outdir at a mounted Drive folder and an interrupted
  run costs minutes instead of the whole thing.

LENS is optional. It is declared in requirements.txt (lens-metric) but pulls its own
pinned torch/transformers, which conflicts with the versions the backend needs; it is
skipped with a warning when not importable rather than being a hard dependency of
getting a document-level number at all.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

# Prompt templates and D-Wikipedia text conventions live in ../backend and are used at
# serve time too; imported, not copied, so scoring and serving cannot diverge.
# One-way dependency (as in evaluate_sentence.py): research/ may import backend/, never
# the reverse.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))
import prompting  # noqa: E402
import vocabulary  # noqa: E402
from document_text import normalize_for_model  # noqa: E402
from hf_revisions import HF_REVISIONS  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('evaluate_document')

SCRIPT_VERSION = '1.1'

# Local output of the last training run; a Hub id also works, e.g.
# --model yunvs/bart-base-dwikipedia-simplification for the published copy.
DEFAULT_MODEL = 'scratch/simplification_results_document/best_checkpoint'
# Zero-shot, un-fine-tuned baseline (same as Step 4): measures what fine-tuning adds.
DEFAULT_BASELINE = 'facebook/bart-base'

# D-Wikipedia as prepared by the notebook's loader: one document per line, lowercased
# and PTB-pre-tokenized (see document_text.py).
DEFAULT_DATA_DIR = 'scratch/d_wikipedia_raw'

# Matches training. Output ceiling well above the measured target-document mean
# (~98 tokens); the notebook sandbox's default of 128 truncates to ~2 sentences.
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 512

# Prompted-LLM document condition (--llm). Optional: absent unless a tag is passed.
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
# Per-document timeout so a stalled request can't hold the run; generous because a
# 1024-token completion on consumer hardware is slow.
OLLAMA_TIMEOUT = float(os.environ.get('OLLAMA_TIMEOUT', '600'))


# --- provenance ---


def git_commit() -> Optional[str]:
    """Short commit hash, so a result can be tied to the code that produced it."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def resolve_model_provenance(model_id: str) -> Dict[str, Any]:
    """Record *what* was scored precisely enough to find it again.

    A bare "best_checkpoint" in a results file is not identifying -- it gets
    overwritten by the next training run. For a local path this also records the
    resolved absolute path and the weights' mtime and size, which together distinguish
    one run's checkpoint from another's.
    """
    info: Dict[str, Any] = {'model_id': model_id}
    path = Path(model_id)
    if path.exists():
        weights = path / 'model.safetensors'
        info['resolved_path'] = str(path.resolve())
        info['source'] = 'local_path'
        if weights.exists():
            stat = weights.stat()
            info['weights_bytes'] = stat.st_size
            info['weights_mtime'] = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
    else:
        info['source'] = 'huggingface_hub'
    return info


# --- data ---


# Mojibake signatures, ported from the notebook's contains_mojibake(). Training drops
# these rows from every split, so filtering here keeps n comparable to a notebook run
# (rather than the raw 8,000).
_MOJIBAKE_PATTERNS = [
    r"â\s*''",  # broken en-dash
    r'â\s*€\s*™',  # broken single curly quote
    r'Ã\s*©',  # broken e-acute
    r'Ã\s*¹',  # broken u-grave
    r'Ã\s*¶',  # broken o-umlaut
    r'â\s*€\s*œ',  # broken opening double quote
    r'â\s*€\s*',  # broken closing double quote
]


def contains_mojibake(text: str) -> bool:
    """Ported unchanged from the notebook. Clean accents ("São Tomé") do not match."""
    return bool(text) and any(re.search(p, text) for p in _MOJIBAKE_PATTERNS)


def load_dwikipedia_test(
    data_dir: str, limit: Optional[int], seed: int
) -> Tuple[List[str], List[List[str]], Dict[str, object]]:
    """D-Wikipedia test split: one document per line, one reference per document.

    Applies the same whitespace-collapse + mojibake filter the training pipeline applies
    to every split, *before* sampling -- so a --limit sample is drawn from the same
    cleaned population the notebook's Step 4B would evaluate, not from the raw file.

    Returns (sources, references, sampling_info) with references shaped
    (n_samples, n_refs) -- the same orientation the metric functions below expect.
    """
    src_path = Path(data_dir) / 'test.src'
    tgt_path = Path(data_dir) / 'test.tgt'
    for p in (src_path, tgt_path):
        if not p.exists():
            raise FileNotFoundError(
                f'{p} not found. The D-Wikipedia raw split lives under {data_dir}/ '
                f'(gitignored); regenerate it via notebooks/train_sentence_and_document_pipeline.ipynb Step 1, '
                f'or download it directly from the RLSNLP/Document-level-text-simplification repo.'
            )

    raw_sources = src_path.read_text(encoding='utf-8').splitlines()
    raw_targets = tgt_path.read_text(encoding='utf-8').splitlines()
    if len(raw_sources) != len(raw_targets):
        raise ValueError(
            f'Split mismatch: {len(raw_sources)} sources vs {len(raw_targets)} targets'
        )

    raw_total = len(raw_sources)
    sources, targets = [], []
    for src, tgt in zip(raw_sources, raw_targets, strict=True):
        src, tgt = collapse_whitespace(src), collapse_whitespace(tgt)
        if contains_mojibake(src) or contains_mojibake(tgt):
            continue
        sources.append(src)
        targets.append(tgt)

    dropped = raw_total - len(sources)
    logger.info(
        'Cleaning: dropped %d/%d documents containing mojibake (%.2f%%)',
        dropped,
        raw_total,
        (dropped / raw_total * 100) if raw_total else 0.0,
    )

    total = len(sources)
    sampling: Dict[str, object] = {
        'split_total_raw': raw_total,
        'split_total_cleaned': total,
        'mojibake_dropped': dropped,
        'sampled': False,
        'sample_seed': None,
    }

    # Random indices, not a head slice: D-Wikipedia's file order is not documented as
    # shuffled. The seed is recorded so the subset is reproducible.
    if limit and limit < total:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), limit))
        sources = [sources[i] for i in indices]
        targets = [targets[i] for i in indices]
        sampling.update(
            sampled=True,
            sample_seed=seed,
            sample_indices_sha=_indices_fingerprint(indices),
        )

    references = [[t] for t in targets]
    logger.info(
        'Loaded D-Wikipedia test: %d of %d documents%s',
        len(sources),
        total,
        f' (seeded sample, seed={seed})' if sampling['sampled'] else '',
    )
    return sources, references, sampling


def _sha16(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _indices_fingerprint(indices: Sequence[int]) -> str:
    return _sha16(','.join(map(str, indices)))


# --- generation ---


def collapse_whitespace(text: str) -> str:
    """Ported from the notebook's preprocessing cell."""
    return re.sub(r'\s+', ' ', text).strip()


def pick_device(requested: str = 'auto') -> str:
    import torch

    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def generate(
    model_id: str,
    sources: List[str],
    device: str,
    batch_size: int,
    label: str,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 5,
    sources_sha: Optional[str] = None,
) -> Tuple[List[str], float]:
    """Batched beam-search generation. Returns (predictions, elapsed_seconds).

    Decoding matches the notebook sandbox's "beam_search" strategy (num_beams=4,
    no_repeat_ngram_size=3, repetition_penalty=1.2), used for every other reported number.

    Checkpointed to `checkpoint_path` every `checkpoint_every` batches and resumed on a
    rerun, for durability: measured on a T4, ~1.0 s/doc fine-tuned and ~2.3 s/doc for the
    baseline (no learned stopping point, runs to max_length), so the full 8,000-document
    split takes ~7 h on a Colab runtime that is killed on disconnect.

    The checkpoint stores a fingerprint of the source list; a partial file from a
    different sample, --limit or cleaning pass is discarded. The fingerprint covers the
    sources, not the decoding, so the sampled --llm condition also keys its cache filename
    by `--llm-seed` (fixed 2026-08-23); otherwise runs differing only in seed would
    silently return the first seed's text.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    predictions: List[str] = []
    prior_seconds = 0.0
    if checkpoint_path and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text())
        saved_predictions = saved.get('predictions', [])
        if saved.get('sources_sha') == sources_sha and len(saved_predictions) <= len(
            sources
        ):
            predictions = saved_predictions
            prior_seconds = saved.get('elapsed_seconds', 0.0)
            logger.info(
                '[%s] resuming from %s -- %d/%d documents already generated',
                label,
                checkpoint_path,
                len(predictions),
                len(sources),
            )
        else:
            logger.warning(
                '[%s] ignoring %s: it was written for a different source list '
                '(fingerprint mismatch) and cannot be resumed from',
                label,
                checkpoint_path,
            )

    if len(predictions) == len(sources):
        logger.info('[%s] generation already complete; skipping the model load', label)
        return predictions, prior_seconds

    started = time.time()

    def flush() -> None:
        if not checkpoint_path:
            return
        checkpoint_path.write_text(
            json.dumps(
                {
                    'label': label,
                    'model_id': model_id,
                    'sources_sha': sources_sha,
                    'n_total': len(sources),
                    'n_done': len(predictions),
                    'complete': len(predictions) == len(sources),
                    'elapsed_seconds': round(
                        prior_seconds + (time.time() - started), 1
                    ),
                    'predictions': predictions,
                },
                indent=2,
            )
        )

    logger.info('[%s] loading %s', label, model_id)
    revision = HF_REVISIONS.get(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision)
    model.to(device)
    model.eval()

    remaining = sources[len(predictions) :]
    for batch_number, start in enumerate(range(0, len(remaining), batch_size), start=1):
        batch = remaining[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_OUTPUT_TOKENS,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                do_sample=False,
            )
        predictions.extend(
            collapse_whitespace(t)
            for t in tokenizer.batch_decode(outputs, skip_special_tokens=True)
        )
        if batch_number % checkpoint_every == 0:
            flush()
        done_this_session = min(start + batch_size, len(remaining))
        elapsed = time.time() - started
        logger.info(
            '[%s] %d/%d documents (%.1fs this session, %.2fs/doc, eta %.0fs)',
            label,
            len(predictions),
            len(sources),
            elapsed,
            elapsed / done_this_session,
            (elapsed / done_this_session) * (len(remaining) - done_this_session),
        )

    flush()

    del model
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()

    return predictions, prior_seconds + (time.time() - started)


def preflight_ollama(tag: str) -> None:
    """Confirm the sidecar is up and the tag is pulled, *before* any generation starts.

    The LLM runs last, so a bad tag would otherwise surface only after both seq2seq
    systems had generated (tens of minutes at n=500). Observed: a trailing hyphen on an
    otherwise-correct tag.
    """
    try:
        with urllib.request.urlopen(f'{OLLAMA_URL}/api/tags', timeout=10) as response:
            available = [
                m.get('name', '') for m in json.loads(response.read()).get('models', [])
            ]
    except urllib.error.URLError as exc:
        raise SystemExit(
            f'--llm was given but Ollama is not reachable at {OLLAMA_URL}: {exc}\n'
            f'Start it with `ollama serve` (or `brew services start ollama`).'
        ) from exc

    if tag not in available:
        listing = '\n  '.join(available) or '(none pulled)'
        raise SystemExit(
            f'--llm {tag!r} is not available in Ollama.\n\nPulled tags:\n  {listing}\n\n'
            f'Check for a typo, or pull it with `ollama pull {tag}`.'
        )
    logger.info('[llm] %s is pulled and Ollama is reachable', tag)


def check_llm_cache_identity(
    cache_path: Path, tag: Optional[str], llm_seed: int, rescoring: bool
) -> None:
    """Refuse to read or overwrite a merged cache that holds a *different* LLM run.

    The merged cache is named by the *sampling* seed (`generations_n<N>_seed<S>.json`),
    but its `llm_predictions` depend on the Ollama tag and `--llm-seed`. Unchecked, a
    second `--llm-seed` in the same --outdir would silently overwrite the first one's
    generations, and `--rescore` would return the first seed's text as the second's.

    Enforced here rather than fixed by renaming the file: renaming would orphan caches
    cited by path in results/RESULTS.md and stop a no-LLM rescore from reusing the seq2seq
    half of an LLM-bearing cache. Rule: one --outdir per LLM run.

    Called before any model is loaded (at n=2000 it would otherwise fail ~70 min in).

    Caches written before 2026-08-24 record no LLM identity; they are all `--llm-seed 1`
    and pass with a warning rather than an error.
    """
    if not cache_path.exists():
        return
    try:
        cached = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return  # a corrupt cache is the resume logic's problem, not this check's
    if not cached.get('llm_predictions'):
        return

    recorded_tag, recorded_seed = cached.get('llm_model'), cached.get('llm_seed')
    if recorded_seed is None:
        logger.warning(
            '[llm] %s predates LLM-identity recording; assuming it is --llm-seed 1 of the '
            'tag this run names. If it is not, point --outdir somewhere else.',
            cache_path.name,
        )
        return

    if (tag, llm_seed) == (recorded_tag, recorded_seed):
        return

    # Allowed: a rescore naming no LLM still scores the cached LLM row ("add a metric to a
    # finished run"), writes nothing, and attributes the row from the recorded identity.
    if rescoring and tag is None:
        return

    asked = f'{tag!r} seed {llm_seed}' if tag else 'no LLM condition'
    raise SystemExit(
        f'\n{cache_path} holds the generations of a different LLM run: '
        f'{recorded_tag!r} seed {recorded_seed}, and this run asks for {asked}.\n\n'
        + (
            "Re-scoring it would report that run's text under this run's provenance.\n"
            if rescoring
            else "Generating over it would overwrite that run's text under a filename that "
            'records no difference between them.\n'
        )
        + f'Use a separate --outdir per LLM run, e.g. {cache_path.parent}_llmseed{llm_seed}/ '
        f'(see results/RESULTS.md \u00a74, item 15).\n'
    )


def generate_ollama_documents(
    tag: str,
    sources: List[str],
    label: str,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 5,
    sources_sha: Optional[str] = None,
    seed: int = 1,
) -> Tuple[List[str], float]:
    """Document-scope generation with a prompted open LLM served by Ollama.

    Mirrors `generate()`'s contract (checkpoint/resume, source fingerprint, return shape)
    so downstream code treats both alike. Separate because almost nothing is shared: no
    tokenizer, no batching (llama.cpp serves one stream), no beam search.

    Uses `prompting` at DOCUMENT granularity, so template, stop sequences, token budget
    and output sanitiser are byte-identical to what the backend serves. Sampling
    parameters come from `evaluation_decoding(DOCUMENT)`.

    Checkpointing matters more here: each document is one ~1024-token completion, so a
    few hundred documents take hours on this hardware.

    Two caveats, both recorded in the run's notes:

    1. D-Wikipedia's sources are lowercased and PTB-pre-tokenized (see
       `document_text.py`), unlike the web prose the LLM sees in the extension. This
       almost certainly *understates* its quality, and restoring capitalisation would
       mean inventing data: the natural-cased original is not in the corpus.
    2. Its natural-prose output is scored against lowercased, pre-tokenized references.
       See `normalize_predictions_for_metrics`.
    """
    predictions: List[str] = []
    prior_seconds = 0.0
    if checkpoint_path and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text())
        saved_predictions = saved.get('predictions', [])
        if saved.get('sources_sha') == sources_sha and len(saved_predictions) <= len(
            sources
        ):
            predictions = saved_predictions
            prior_seconds = saved.get('elapsed_seconds', 0.0)
            logger.info(
                '[%s] resuming from %s -- %d/%d documents already generated',
                label,
                checkpoint_path,
                len(predictions),
                len(sources),
            )
        else:
            logger.warning(
                '[%s] ignoring %s: written for a different source list (fingerprint '
                'mismatch) and cannot be resumed from',
                label,
                checkpoint_path,
            )

    if len(predictions) == len(sources):
        logger.info('[%s] generation already complete; skipping', label)
        return predictions, prior_seconds

    options = prompting.evaluation_decoding(prompting.WHOLE_SECTIONS)
    options['seed'] = seed
    started = time.time()
    rejected: Dict[str, int] = {}

    def flush() -> None:
        if not checkpoint_path:
            return
        checkpoint_path.write_text(
            json.dumps(
                {
                    'label': label,
                    'model_id': tag,
                    'sources_sha': sources_sha,
                    'n_total': len(sources),
                    'n_done': len(predictions),
                    'complete': len(predictions) == len(sources),
                    'elapsed_seconds': round(
                        prior_seconds + (time.time() - started), 1
                    ),
                    'rejected': rejected,
                    'predictions': predictions,
                },
                indent=2,
            )
        )

    logger.info(
        '[%s] prompting %s at document granularity (%d to go)',
        label,
        tag,
        len(sources) - len(predictions),
    )

    for offset, source in enumerate(sources[len(predictions) :], start=1):
        payload = {
            'model': tag,
            'prompt': prompting.build_prompt(
                source, granularity=prompting.WHOLE_SECTIONS
            ),
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
        try:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
                raw = json.loads(response.read()).get('response', '')
        except urllib.error.HTTPError as exc:
            # Keep the body: it carries the reason (dropping it once cost hours; see
            # backend/main.py's Ollama handling).
            raise SystemExit(
                f'Ollama rejected the request (HTTP {exc.code}): '
                f"{exc.read().decode(errors='replace')[:400]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SystemExit(
                f'Cannot reach Ollama at {OLLAMA_URL}: {exc}. Start it with '
                f'`ollama serve` (or `brew services start ollama`) and pull {tag}.'
            ) from exc

        final, reason, _model_result = prompting.sanitize(
            source, raw, prompting.WHOLE_SECTIONS
        )
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
        predictions.append(collapse_whitespace(final))

        if offset % checkpoint_every == 0:
            flush()
        if offset % 10 == 0 or len(predictions) == len(sources):
            elapsed = time.time() - started
            logger.info(
                '[%s] %d/%d documents (%.2fs/doc, eta %.0fs)',
                label,
                len(predictions),
                len(sources),
                elapsed / offset,
                (elapsed / offset) * (len(sources) - len(predictions)),
            )

    flush()
    if rejected:
        # Not a failure: rejected generations fall back to the source, as in the backend,
        # so the score reflects what a user would see.
        logger.warning(
            '[%s] %d/%d generations rejected by the output guards: %s',
            label,
            sum(rejected.values()),
            len(sources),
            rejected,
        )
    return predictions, prior_seconds + (time.time() - started)


def normalize_predictions_for_metrics(predictions: List[str], label: str) -> List[str]:
    """Put every system's output into D-Wikipedia's own text convention before scoring.

    D-SARI, SARI and BLEU are n-gram metrics, and this corpus (sources and references) is
    lowercased and PTB-pre-tokenized. Scored raw, a prompted LLM's prose ("Achtkarspelen
    is a town in Friesland, ...") would count `Achtkarspelen`/`achtkarspelen` or
    `Friesland,`/`friesland ,` as different tokens -- an artifact of text conventions,
    not of simplification quality.

    Applied to every system, not only the LLM, so all get the same treatment.

    NOT a no-op for the seq2seq systems. Measured at n=20: 4/20 fine-tuned and 12/20
    baseline predictions changed (LLM: 20/20). Two effects in opposite directions:

    - Fixes: `"actor.he was born"` -> `"actor . he was born"` (missing space after a
      period, a real seq2seq artifact).
    - Corruption: `"u.s. state"` -> `"u . s . state"`. The test split has `u.s.` 531
      times and `u . s .` zero times, so this creates a token the corpus never contains,
      from output that was already correct. `normalize_for_model`'s period rule splits
      any period not adjacent to a digit, which is wrong for abbreviations.

    Consequences:

    1. Scores are not strictly comparable to a run scored without this step (D2 in
       results/RESULTS.md); the two must not share a table without a note.
    2. The abbreviation bug is in the shared serving path: the backend applies
       `normalize_for_model` to document input, so the document checkpoint sees
       `u . s .` in production too. Fixing it there fixes both.
    3. The corruption mildly penalises the seq2seq systems. It does not explain the
       LLM's D-SARI gap, which is far larger than this effect.
    """
    normalized = [normalize_for_model(p) for p in predictions]
    changed = sum(
        1
        for before, after in zip(predictions, normalized, strict=True)
        if before != after
    )
    logger.info(
        '[%s] metric normalization changed %d/%d predictions%s',
        label,
        changed,
        len(predictions),
        ' (no-op, as expected for corpus-convention output)' if changed == 0 else '',
    )
    return normalized


# --- metrics ---
# Ported unchanged from notebooks/train_sentence_and_document_pipeline.ipynb.


def _ngrams_upto_4(tokens: List[str]) -> Tuple[List[str], List[str], List[str]]:
    bigrams, trigrams, fourgrams = [], [], []
    for i in range(len(tokens) - 1):
        bigrams.append(' '.join(tokens[i : i + 2]))
        if i < len(tokens) - 2:
            trigrams.append(' '.join(tokens[i : i + 3]))
        if i < len(tokens) - 3:
            fourgrams.append(' '.join(tokens[i : i + 4]))
    return bigrams, trigrams, fourgrams


def _d_sari_ngram(
    sgrams: List[str], cgrams: List[str], rgramslist: List[List[str]], numref: int
) -> Tuple[float, float, float]:
    """Keep/delete/add at a single n. Retains the reference implementation's quirk of
    returning delete *precision* rather than an F1, so scores stay comparable to
    published D-SARI results."""
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter({g: c * numref for g, c in sgramcounter.items()})
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter({g: c * numref for g, c in cgramcounter.items()})

    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = sum(
        keepgramcountergood_rep[g] / keepgramcounter_rep[g]
        for g in keepgramcountergood_rep
    )
    keeptmpscore2 = sum(
        keepgramcountergood_rep[g] / keepgramcounterall_rep[g]
        for g in keepgramcountergood_rep
    )

    keepscore_precision = (
        keeptmpscore1 / len(keepgramcounter_rep) if keepgramcounter_rep else 0
    )
    keepscore_recall = (
        keeptmpscore2 / len(keepgramcounterall_rep) if keepgramcounterall_rep else 0
    )
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = (
            2
            * keepscore_precision
            * keepscore_recall
            / (keepscore_precision + keepscore_recall)
        )

    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    deltmpscore1 = sum(
        delgramcountergood_rep[g] / delgramcounter_rep[g]
        for g in delgramcountergood_rep
    )
    delscore_precision = (
        deltmpscore1 / len(delgramcounter_rep) if delgramcounter_rep else 0
    )

    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = addgramcounter & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = len(addgramcountergood)
    addscore_precision = addtmpscore / len(addgramcounter) if addgramcounter else 0
    addscore_recall = addtmpscore / len(addgramcounterall) if addgramcounterall else 0
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = (
            2
            * addscore_precision
            * addscore_recall
            / (addscore_precision + addscore_recall)
        )

    return keepscore, delscore_precision, addscore


def d_sari_document(source: str, candidate: str, references: List[str]) -> float:
    """Per-document D-SARI in [0, 1] -- compute_d_sari rescales to 0-100."""
    import nltk

    numref = len(references)

    s1grams = source.lower().split(' ')
    c1grams = candidate.lower().split(' ')
    s2grams, s3grams, s4grams = _ngrams_upto_4(s1grams)
    c2grams, c3grams, c4grams = _ngrams_upto_4(c1grams)

    r1gramslist, r2gramslist, r3gramslist, r4gramslist = [], [], [], []
    for reference in references:
        r1grams = reference.lower().split(' ')
        r2grams, r3grams, r4grams = _ngrams_upto_4(r1grams)
        r1gramslist.append(r1grams)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    keep_scores, del_scores, add_scores = [], [], []
    for sgrams, cgrams, rgramslist in (
        (s1grams, c1grams, r1gramslist),
        (s2grams, c2grams, r2gramslist),
        (s3grams, c3grams, r3gramslist),
        (s4grams, c4grams, r4gramslist),
    ):
        keep, delete, add = _d_sari_ngram(sgrams, cgrams, rgramslist, numref)
        keep_scores.append(keep)
        del_scores.append(delete)
        add_scores.append(add)

    avg_keep = sum(keep_scores) / 4
    avg_del = sum(del_scores) / 4
    avg_add = sum(add_scores) / 4

    input_length = len(source.split(' '))
    output_length = len(candidate.split(' '))
    reference_length = int(sum(len(r.split(' ')) for r in references) / numref)

    output_sentence_count = len(nltk.sent_tokenize(candidate))
    reference_sentence_count = int(
        sum(len(nltk.sent_tokenize(r)) for r in references) / numref
    )

    lp_1 = (
        1.0
        if output_length >= reference_length
        else math.exp((output_length - reference_length) / max(output_length, 1))
    )
    lp_2 = 1.0
    if output_length > reference_length:
        lp_2 = math.exp(
            (reference_length - output_length) / max(input_length - reference_length, 1)
        )

    slp = math.exp(
        -abs(reference_sentence_count - output_sentence_count)
        / max(reference_sentence_count, output_sentence_count, 1)
    )

    avg_keep *= lp_2 * slp
    avg_add *= lp_1
    avg_del *= lp_2

    return (avg_keep + avg_del + avg_add) / 3


def d_sari_per_document(
    predictions: List[str], references: List[List[str]], sources: List[str]
) -> List[float]:
    """Per-document D-SARI scores, in [0, 1]. Kept separately from the corpus mean so
    the bootstrap can resample these instead of recomputing n-gram counters per
    resample -- corpus D-SARI is their plain mean, so this is exact."""
    return [
        d_sari_document(src, pred, refs)
        for src, pred, refs in zip(sources, predictions, references, strict=True)
    ]


def compute_evaluation_metrics(
    predictions: List[str], references: List[List[str]], sources: List[str]
) -> Dict[str, float]:
    """SARI/BLEU/FKGL, for continuity with Step 4's numbers.

    SARI is reported here because Step 4 reports it, but D-SARI is the metric that
    actually fits this task -- plain SARI has no notion of sentence deletion, merging or
    reordering, which is precisely what a document model does.
    """
    import sacrebleu
    from easse.fkgl import corpus_fkgl
    from easse.sari import corpus_sari

    # sacrebleu and easse want (n_references, n_samples); `references` is per-sample, so
    # transpose rather than assume a single reference.
    formatted_references = [list(refs) for refs in zip(*references, strict=True)]
    bleu = sacrebleu.corpus_bleu(
        predictions, formatted_references, tokenize='none', force=True
    ).score
    sari = corpus_sari(
        orig_sents=sources, sys_sents=predictions, refs_sents=formatted_references
    )
    fkgl = corpus_fkgl(sentences=predictions)
    return {'BLEU': bleu, 'SARI': sari, 'FKGL': fkgl}


def compute_bertscore(predictions: List[str], references: List[List[str]]) -> float:
    """Corpus-mean BERTScore F1, rescaled to 0-100.

    Caveat: BERTScore silently truncates past ~510 tokens, a real limitation for
    document-length output.
    """
    from bert_score import score

    # With return_hash=False score() returns (P, R, F), but its declared return type
    # also admits (triple, hash); cast to pin the unpack.
    _, _, f1 = cast(
        Tuple,
        score(predictions, references, lang='en', batch_size=16, verbose=False),
    )
    return f1.mean().item() * 100


def compute_lens(
    sources: List[str], predictions: List[str], references: List[List[str]]
) -> Optional[float]:
    """Corpus-mean LENS (0-100), or None when lens-metric isn't importable."""
    try:
        # lens-metric pins its own torch/transformers, so it lives in a separate venv;
        # the except branch is the normal path here.
        from lens import LENS, download_model  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            'LENS unavailable (lens-metric not importable) -- skipping. Install it in a '
            "SEPARATE venv from the backend's: it pins its own torch/transformers."
        )
        return None
    logger.info('Loading the LENS checkpoint (davidheineman/lens)...')
    model = LENS(download_model('davidheineman/lens'), rescale=True)
    scores = model.score(
        complex=sources, simplified=predictions, references=references, batch_size=16
    )
    return sum(scores) / len(scores)


def paired_bootstrap_d_sari(
    baseline_scores: List[float],
    finetuned_scores: List[float],
    num_resamples: int,
    seed: int,
) -> float:
    """p-value for the D-SARI improvement, by paired bootstrap over documents.

    Resamples the precomputed per-document scores. Corpus D-SARI is their mean, so this
    is arithmetically identical to recomputing D-SARI on each resample -- just ~1000x
    cheaper. Paired: both systems are resampled at the *same* indices, so the comparison
    controls for which documents happened to be drawn.
    """
    n = len(baseline_scores)
    if n == 0:
        return 1.0
    rng = random.Random(seed)
    better = 0
    for _ in range(num_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        base = sum(baseline_scores[i] for i in indices) / n
        fine = sum(finetuned_scores[i] for i in indices) / n
        if fine > base:
            better += 1
    return 1.0 - (better / num_resamples)


# --- run ---


def score_system(
    label: str,
    predictions: List[str],
    references: List[List[str]],
    sources: List[str],
    skip_lens: bool,
) -> Tuple[Dict[str, Optional[float]], List[float]]:
    logger.info('[%s] scoring BLEU/SARI/FKGL...', label)
    scores: Dict[str, Optional[float]] = dict(
        compute_evaluation_metrics(predictions, references, sources)
    )

    logger.info('[%s] scoring D-SARI...', label)
    per_doc = d_sari_per_document(predictions, references, sources)
    scores['D-SARI'] = (sum(per_doc) / len(per_doc)) * 100 if per_doc else 0.0

    logger.info('[%s] scoring BERTScore...', label)
    scores['BERTScore'] = compute_bertscore(predictions, references)

    scores['LENS'] = (
        None if skip_lens else compute_lens(sources, predictions, references)
    )
    logger.info('[%s] scores: %s', label, scores)
    return scores, per_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help='document checkpoint (local path or Hub id)',
    )
    parser.add_argument(
        '--baseline', default=DEFAULT_BASELINE, help='zero-shot comparison model'
    )
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument(
        '--limit', type=int, default=500, help='documents to sample; 0 = the full 8,000'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='sampling + bootstrap seed'
    )
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='auto-detects; measured ~8.4s/doc on MPS vs. far faster on a CUDA T4',
    )
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--outdir', default='results/document_eval')
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=5,
        help='flush partial generations every N batches (default 5); a rerun resumes from them',
    )
    parser.add_argument(
        '--skip-lens', action='store_true', help='skip LENS even if importable'
    )
    parser.add_argument(
        '--rescore',
        action='store_true',
        help='reuse cached generations, recompute metrics',
    )
    parser.add_argument(
        '--llm',
        default=None,
        metavar='OLLAMA_TAG',
        help='also score a prompted open LLM at document granularity, e.g. '
        'qwen2.5:7b-instruct-q4_K_M. Needs `ollama serve` and the tag pulled. '
        'Adds a third system; the fine-tuned-vs-baseline comparison is unchanged.',
    )
    parser.add_argument(
        '--llm-seed',
        type=int,
        default=1,
        help='sampling seed for the --llm condition (its decoding samples; the seq2seq '
        "systems use beam search and are deterministic). Also keys the condition's "
        'generation cache filename: a second seed must not resume from the first '
        "seed's finished generations, which is what a shared filename would cause.",
    )
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cached_llm_identity: Tuple[Optional[str], Optional[int]] = (None, None)
    limit = args.limit or None
    sources, references, sampling = load_dwikipedia_test(
        args.data_dir, limit, args.seed
    )
    device = pick_device(args.device)
    logger.info('Device: %s', device)

    # Fingerprint of the cleaned, sampled source list; per-system checkpoints resume only
    # on a match.
    sources_sha = _sha16('\n'.join(sources))
    stamp = started_at.strftime('%Y%m%dT%H%M%SZ')
    out_path = outdir / f'step4b_dwikipedia_{stamp}_n{len(sources)}.json'

    # Cached so adding a metric or rerunning the significance test never re-runs a model.
    cache_path = outdir / f'generations_n{len(sources)}_seed{args.seed}.json'

    # Before anything expensive, in this order: a wrong --outdir needs no sidecar to
    # detect, and a bad tag is caught before the seq2seq systems generate.
    check_llm_cache_identity(
        cache_path, args.llm, args.llm_seed, rescoring=args.rescore
    )
    if args.llm:
        preflight_ollama(args.llm)
    if args.rescore and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        finetuned_predictions = cached['finetuned_predictions']
        baseline_predictions = cached['baseline_predictions']
        # absent in caches predating --llm or generated without it: no LLM condition
        llm_predictions = cached.get('llm_predictions')
        # So an LLM row rescored without --llm is still attributed to its model.
        cached_llm_identity = (cached.get('llm_model'), cached.get('llm_seed'))
        gen_seconds = cached.get('generation_seconds', {})
        logger.info('Reusing cached generations from %s', cache_path)
        if args.llm and not llm_predictions:
            # Common case: seq2seq output (expensive, deterministic) is cached and only
            # the LLM is new; generate just that and extend the cache.
            logger.info(
                'Cache has no LLM generations; generating only that and reusing the '
                'cached seq2seq output'
            )
            llm_predictions, llm_secs = generate_ollama_documents(
                args.llm,
                sources,
                'llm',
                checkpoint_path=outdir
                / f'partial_llm_n{len(sources)}_seed{args.seed}_llmseed{args.llm_seed}.json',
                checkpoint_every=args.checkpoint_every,
                sources_sha=sources_sha,
                seed=args.llm_seed,
            )
            gen_seconds = {**gen_seconds, 'llm': round(llm_secs, 1)}
            cached['llm_predictions'] = llm_predictions
            cached['llm_model'] = args.llm
            cached['llm_seed'] = args.llm_seed
            cached['generation_seconds'] = gen_seconds
            cache_path.write_text(json.dumps(cached, indent=2))
            logger.info('Extended %s with the LLM generations', cache_path)
    else:
        finetuned_predictions, ft_secs = generate(
            args.model,
            sources,
            device,
            args.batch_size,
            'fine-tuned',
            checkpoint_path=outdir
            / f'partial_finetuned_n{len(sources)}_seed{args.seed}.json',
            checkpoint_every=args.checkpoint_every,
            sources_sha=sources_sha,
        )
        baseline_predictions, bl_secs = generate(
            args.baseline,
            sources,
            device,
            args.batch_size,
            'baseline',
            checkpoint_path=outdir
            / f'partial_baseline_n{len(sources)}_seed{args.seed}.json',
            checkpoint_every=args.checkpoint_every,
            sources_sha=sources_sha,
        )
        gen_seconds = {'finetuned': round(ft_secs, 1), 'baseline': round(bl_secs, 1)}

        llm_predictions = None
        if args.llm:
            llm_predictions, llm_secs = generate_ollama_documents(
                args.llm,
                sources,
                'llm',
                checkpoint_path=outdir
                / f'partial_llm_n{len(sources)}_seed{args.seed}_llmseed{args.llm_seed}.json',
                checkpoint_every=args.checkpoint_every,
                sources_sha=sources_sha,
                seed=args.llm_seed,
            )
            gen_seconds['llm'] = round(llm_secs, 1)

        cache_path.write_text(
            json.dumps(
                {
                    'finetuned_predictions': finetuned_predictions,
                    'baseline_predictions': baseline_predictions,
                    # The filename can't carry tag and seed (see
                    # check_llm_cache_identity), so the file does.
                    **(
                        {
                            'llm_predictions': llm_predictions,
                            'llm_model': args.llm,
                            'llm_seed': args.llm_seed,
                        }
                        if llm_predictions
                        else {}
                    ),
                    'generation_seconds': gen_seconds,
                },
                indent=2,
            )
        )
        logger.info('Cached generations to %s', cache_path)

    result = {
        'run': {
            'script': 'evaluate_document.py',
            'script_version': SCRIPT_VERSION,
            'step': '4B',
            'status': 'generated',
            'started_at': started_at.isoformat(),
            'finished_at': None,
            'duration_seconds': None,
            'git_commit': git_commit(),
            'device': device,
        },
        'models': {
            'finetuned': resolve_model_provenance(args.model),
            'baseline': resolve_model_provenance(args.baseline),
            **(
                {
                    'llm': {
                        'model_id': args.llm,
                        'kind': 'ollama',
                        # No digest: Ollama tags are mutable and this script can't read
                        # one back. main.py's MODEL_ENV_CONFIG has the pinned digests.
                        'granularity': vocabulary.GRANULARITY_WHOLE_SECTIONS,
                        'prompt': "document instruction (adaptation of BLESS Prompt 2, not the paper's)",
                        'fewshot_n': 0,
                        'decoding': prompting.evaluation_decoding(
                            prompting.WHOLE_SECTIONS
                        ),
                        'seed': args.llm_seed,
                    }
                }
                if args.llm
                else {}
            ),
            # --rescore without --llm still scores a cached LLM condition; its identity
            # then comes from the cache.
            **(
                {
                    'llm': {
                        'model_id': cached_llm_identity[0],
                        'kind': 'ollama',
                        'seed': cached_llm_identity[1],
                        'source': 'recorded by the run that generated this cache; --llm was not '
                        'passed to the run that scored it',
                    }
                }
                if (not args.llm and llm_predictions and cached_llm_identity[0])
                else {}
            ),
        },
        'dataset': {
            'name': 'D-Wikipedia',
            'citation': 'Sun, Jin & Wan (EMNLP 2021)',
            'split': 'test',
            'references_per_document': 1,
            'n_evaluated': len(sources),
            **sampling,
        },
        'decoding': {
            'strategy': 'beam_search',
            'num_beams': 4,
            'no_repeat_ngram_size': 3,
            'repetition_penalty': 1.2,
            'max_input_tokens': MAX_INPUT_TOKENS,
            'max_output_tokens': MAX_OUTPUT_TOKENS,
            'batch_size': args.batch_size,
        },
        'scores': {},
        'significance': None,
        'generation_seconds': gen_seconds,
        'notes': [
            'D-SARI is the headline metric here; plain SARI is reported only for '
            'continuity with the ASSET-based Step 4 numbers and does not model '
            'sentence deletion/merging/reordering.',
            'BERTScore truncates inputs past ~510 tokens.',
            "Every system's predictions are normalized into D-Wikipedia's own text "
            'convention (lowercased, PTB-pre-tokenized) before scoring, because the '
            "corpus's sources and references both use it and D-SARI/SARI/BLEU are "
            'n-gram metrics. It is NOT a no-op for the seq2seq systems -- it changed '
            '~25% of fine-tuned and ~65% of baseline predictions at n=500 -- and it '
            'moves D-SARI by up to ~1 point; see the per-system counts logged above.',
        ]
        + (
            [
                "The --llm condition is prompted with the corpus's lowercased, "
                'pre-tokenized source text, which is nothing like the web prose it '
                'processes in the extension. This likely understates its quality and is '
                'not fixable here: the natural-cased original is not in the corpus.',
                'The --llm condition samples (temperature 1.0, top_p 0.9) while the seq2seq '
                'systems use deterministic beam search, so only the former carries '
                'run-to-run variance. One seed was run; BLESS aggregates three.',
            ]
            if args.llm
            else []
        ),
    }

    def save(status: str) -> None:
        """Rewrite the result file after every stage (scoring isn't free: BERTScore
        pulls roberta-large). `status` records how far the run got, so a file from an
        interrupted run is recognisable as one.
        """
        finished = datetime.now(timezone.utc)
        result['run']['status'] = status
        result['run']['finished_at'] = finished.isoformat()
        result['run']['duration_seconds'] = round(
            (finished - started_at).total_seconds(), 1
        )
        out_path.write_text(json.dumps(result, indent=2))
        logger.info('Wrote %s (status=%s)', out_path, status)

    save('generated')

    # Identical treatment for every system -- see normalize_predictions_for_metrics.
    finetuned_predictions = normalize_predictions_for_metrics(
        finetuned_predictions, 'fine-tuned'
    )
    baseline_predictions = normalize_predictions_for_metrics(
        baseline_predictions, 'baseline'
    )
    if llm_predictions:
        llm_predictions = normalize_predictions_for_metrics(llm_predictions, 'llm')

    finetuned_scores, ft_per_doc = score_system(
        'fine-tuned', finetuned_predictions, references, sources, args.skip_lens
    )
    result['scores']['finetuned'] = finetuned_scores
    if finetuned_scores.get('LENS') is None:
        result['notes'].append(
            'LENS not computed (lens-metric unavailable in this environment).'
        )
    save('scored_finetuned')

    baseline_scores, bl_per_doc = score_system(
        'baseline', baseline_predictions, references, sources, args.skip_lens
    )
    result['scores']['baseline'] = baseline_scores
    save('scored_baseline')

    llm_per_doc: Optional[List[float]] = None
    if llm_predictions:
        llm_scores, llm_per_doc = score_system(
            'llm', llm_predictions, references, sources, args.skip_lens
        )
        result['scores']['llm'] = llm_scores
        save('scored_llm')

    p_value = paired_bootstrap_d_sari(bl_per_doc, ft_per_doc, args.resamples, args.seed)
    logger.info(
        'D-SARI improvement significance: p=%.5f (%d resamples)',
        p_value,
        args.resamples,
    )
    result['significance'] = {
        'metric': 'd_sari',
        'test': 'paired bootstrap over documents',
        'resamples': args.resamples,
        'seed': args.seed,
        'p_value': p_value,
    }
    if llm_per_doc is not None:
        # Two questions, two tests (not one three-way comparison): fine-tuned vs.
        # un-fine-tuned (RQ1, above) and prompted LLM vs. each (Method 3). Same
        # per-document D-SARI scores and seed.
        result['significance_llm'] = {
            'metric': 'd_sari',
            'test': 'paired bootstrap over documents',
            'resamples': args.resamples,
            'seed': args.seed,
            'llm_vs_baseline_p': paired_bootstrap_d_sari(
                bl_per_doc, llm_per_doc, args.resamples, args.seed
            ),
            'llm_vs_finetuned_p': paired_bootstrap_d_sari(
                ft_per_doc, llm_per_doc, args.resamples, args.seed
            ),
        }
    save('complete')

    print(json.dumps(result['scores'], indent=2))
    print(
        f'\np(fine-tuned D-SARI improvement over baseline due to noise) = {p_value:.5f}'
    )
    if result.get('significance_llm'):
        sig = result['significance_llm']
        print(f"p(LLM over baseline due to noise)   = {sig['llm_vs_baseline_p']:.5f}")
        print(f"p(LLM over fine-tuned due to noise) = {sig['llm_vs_finetuned_p']:.5f}")
    print(f'Full result with provenance: {out_path}')


if __name__ == '__main__':
    main()
