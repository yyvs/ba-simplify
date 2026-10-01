"""Generate prompted-LLM simplifications for the ASSET benchmark, locally or remotely.

    # local Ollama, same as evaluate_sentence.py does inline
    python evaluate_llm.py --condition llm_7b --backend ollama

    # remote OpenAI-compatible endpoint (Together / Groq / OpenRouter / vLLM / TGI)
    export LLM_API_KEY=...
    python evaluate_llm.py --condition llm_7b_remote --backend openai \
        --base-url https://api.together.xyz/v1 \
        --model Qwen/Qwen2.5-7B-Instruct-Turbo --concurrency 8

    # then score whatever exists, with the harness that owns the metrics
    python evaluate_sentence.py --conditions online,local,llm_7b --bootstrap 1000

This deliberately does **not** compute metrics. `evaluate_sentence.py` owns SARI/BLEU/
FKGL/BERTScore and the bootstrap; this script only produces generations and writes them
in that script's cache format (`{condition}_seed{n}.json`), so the two never disagree
about how a number was computed. Duplicated scoring code is how a comparison stops
being a comparison.

--------------------------------------------------------------------------------
Why a remote backend is acceptable *here* and not in the extension
--------------------------------------------------------------------------------

The browser extension must run the LLM locally: it processes whatever page the user is
reading, and sending that to a third party would void the local-processing property the
system is built around (the thesis §3.6). That argument does not transfer to this
script. ASSET is a public, published benchmark -- there is no user data in it, nothing
private to leak, and the only thing that matters is wall-clock. Measured on the
project's own hardware, a 4-bit 7B manages 3.4-17.3 tok/s depending on machine load,
which puts a four-condition, three-seed run in the several-hours range; the same weights
on a rented GPU run 1-2 orders of magnitude faster.

So: local for deployment because privacy is a design constraint, remote for offline
benchmarking because it is public data. Both use the same prompt, the same decoding
parameters, and the same output sanitiser.

--------------------------------------------------------------------------------
What using a hosted endpoint costs, recorded rather than hidden
--------------------------------------------------------------------------------

1. **Reproducibility.** A local Ollama tag can be pinned to a digest. A hosted endpoint
   cannot -- providers silently change quantisation, serving stack and model revision.
   Every run therefore writes a provenance block with `reproducible: false` for remote
   backends, and the thesis should report the endpoint and date alongside any number
   produced this way (§6.6).
2. **Chat templating.** BLESS Prompt 2 is a *completion* prompt ending in a dangling
   `Simple:`. Ollama's /api/generate consumes it raw. An OpenAI-compatible **chat**
   endpoint wraps it in a chat template, which is not the same input. Use
   `--api-style completions` where the provider supports it; where it doesn't, the
   remote and local numbers are close but not strictly identical, and that belongs in
   the write-up rather than in a footnote nobody reads.
3. **Rate limits.** Free tiers throttle aggressively, which is exactly why this script
   appends to a JSONL as it goes and can resume: a 429 storm, an expired session or a
   closed laptop costs you the current sentence, not the run.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, cast

# The real prompt module, not a copy, so serving and evaluation prompts can't diverge.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))
import prompting  # noqa: E402
from hf_revisions import HF_REVISIONS  # noqa: E402
from prompting import Audience  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Retry schedule for throttling and transient server errors; generous because on a
# free tier waiting is nearly always cheaper than failing the run.
RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
BASE_BACKOFF_S = 2.0


class RateLimited(Exception):
    """Raised for a retryable HTTP status, carrying the server's Retry-After if given."""

    def __init__(self, status: int, retry_after: Optional[float] = None):
        super().__init__(f'HTTP {status}')
        self.status = status
        self.retry_after = retry_after


def _post_json(
    url: str, payload: dict, headers: Dict[str, str], timeout: float
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in RETRY_STATUSES:
            retry_after = exc.headers.get('Retry-After') if exc.headers else None
            try:
                retry_after = float(retry_after) if retry_after else None
            except ValueError:
                retry_after = None
            raise RateLimited(exc.code, retry_after) from exc
        # Non-retryable: surface the body, where providers put the actual reason (bad
        # model name, missing credit, malformed field). See main.py's Ollama handling.
        body = exc.read().decode(errors='replace')[:500]
        raise RuntimeError(f'HTTP {exc.code} from {url}: {body}') from exc


def _with_retries(fn, label: str) -> dict:
    """Run `fn`, retrying retryable statuses with exponential backoff + jitter."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except RateLimited as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = (
                exc.retry_after
                if exc.retry_after
                else BASE_BACKOFF_S * (2 ** (attempt - 1))
            )
            # jitter so parallel workers don't retry in lockstep and re-trip the limit
            wait += random.uniform(0, 0.5 * wait)
            logger.warning(
                '%s: %s, retrying in %.1fs (attempt %d/%d)',
                label,
                exc,
                wait,
                attempt,
                MAX_ATTEMPTS,
            )
            time.sleep(wait)
    # Unreachable; keeps the declared return type valid on every path.
    raise AssertionError(f'{label}: retry loop exhausted without returning or raising')


# --- backends ---


class OllamaBackend:
    """Local sidecar. Raw completion prompt, one request at a time.

    llama.cpp serves a single stream per model, so concurrency above 1 only queues
    inside Ollama while making failures harder to attribute.
    """

    reproducible = True
    default_concurrency = 1

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, options: dict) -> str:
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': options,
            # int, not "-1": a string keep_alive is parsed as a Go duration and 400s.
            'keep_alive': -1,
        }
        body = _with_retries(
            lambda: _post_json(
                f'{self.base_url}/api/generate', payload, {}, self.timeout
            ),
            'ollama',
        )
        return body.get('response', '')


class OpenAICompatBackend:
    """Any OpenAI-compatible endpoint: Together, Groq, OpenRouter, vLLM, TGI, ...

    Two API styles, and the difference matters for comparability:

    - ``completions`` (/v1/completions) takes the raw prompt, so it is what BLESS
      Prompt 2 was designed for and what the local backend sends. Prefer it.
    - ``chat`` (/v1/chat/completions) wraps the prompt in the model's chat template.
      Universally supported, but no longer byte-identical input.
    """

    reproducible = False
    default_concurrency = 8

    def __init__(
        self, base_url: str, model: str, timeout: float, api_key: str, api_style: str
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self.api_style = api_style

    def generate(self, prompt: str, options: dict) -> str:
        # Map Ollama option names to the OpenAI dialect so both backends use the single
        # EVALUATION_DECODING definition.
        shared = {
            'model': self.model,
            'temperature': options.get('temperature'),
            'top_p': options.get('top_p'),
            'max_tokens': options.get('num_predict'),
            'stop': options.get('stop'),
        }
        if options.get('seed') is not None:
            shared['seed'] = options['seed']
        shared = {k: v for k, v in shared.items() if v is not None}

        if self.api_style == 'completions':
            url, payload = f'{self.base_url}/completions', {**shared, 'prompt': prompt}
            body = _with_retries(
                lambda: _post_json(url, payload, self.headers, self.timeout), 'remote'
            )
            return body['choices'][0].get('text', '')

        url = f'{self.base_url}/chat/completions'
        payload = {**shared, 'messages': [{'role': 'user', 'content': prompt}]}
        body = _with_retries(
            lambda: _post_json(url, payload, self.headers, self.timeout), 'remote'
        )
        return body['choices'][0]['message'].get('content', '')


class Backend(Protocol):
    """What main() needs of a backend, and all it needs.

    `reproducible` is False for hosted endpoints, whose served revision and
    quantisation cannot be pinned -- see the module docstring and thesis §6.6.
    """

    reproducible: bool
    default_concurrency: int

    def generate(self, prompt: str, options: Dict) -> str: ...


def build_backend(args) -> Backend:
    if args.backend == 'ollama':
        return OllamaBackend(
            args.base_url or 'http://127.0.0.1:11434', args.model, args.timeout
        )

    api_key = os.environ.get(args.api_key_env, '')
    if not api_key:
        # A warning, not an error: a local vLLM or TGI needs no key.
        logger.warning(
            '%s is unset -- sending unauthenticated requests', args.api_key_env
        )
    if not args.base_url:
        raise SystemExit('--base-url is required for --backend openai')
    return OpenAICompatBackend(
        args.base_url, args.model, args.timeout, api_key, args.api_style
    )


# --- data + resumable output ---


def load_asset_test(limit: Optional[int] = None) -> List[str]:
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
    if limit:
        sources = sources[:limit]
    logger.info('Loaded ASSET test: %d sentences', len(sources))
    return sources


def read_partial(path: Path) -> Dict[int, dict]:
    """Load whatever a previous (possibly interrupted) run already produced.

    Kept in a single JSON object keyed by sentence index, so a resumed run skips exactly
    what is done and nothing else. The older *.partial.jsonl format is still accepted for
    compatibility with earlier runs.
    """
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        # Legacy format: one JSON record per line.
        done: Dict[int, dict] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done[record['index']] = record
            except (json.JSONDecodeError, KeyError):
                logger.warning(
                    'Dropping malformed line in %s (interrupted write?)', path.name
                )
        return done

    if isinstance(payload, dict):
        # New partial format: {"0": {...}, "1": {...}}.
        done: Dict[int, dict] = {}
        for key, record in payload.items():
            try:
                done[int(key)] = record
            except (TypeError, ValueError):
                logger.warning('Ignoring non-index key in %s: %r', path.name, key)
        return done

    if isinstance(payload, list):
        done = {}
        for record in payload:
            if isinstance(record, dict) and 'index' in record:
                done[int(record['index'])] = record
        return done

    return {}


def generate_all(
    backend,
    sources: Sequence[str],
    audience: Audience,
    options: dict,
    partial_path: Path,
    concurrency: int,
) -> List[dict]:
    """Generate for every source, writing each result back into `partial_path` as it lands."""
    done = read_partial(partial_path)
    if done:
        logger.info('Resuming: %d/%d already generated', len(done), len(sources))

    todo = [i for i in range(len(sources)) if i not in done]
    if not todo:
        logger.info('Nothing to do -- all %d sentences already generated', len(sources))
        return [done[i] for i in range(len(sources))]

    started = time.monotonic()
    completed = 0

    def one(index: int) -> dict:
        source = sources[index]
        raw = backend.generate(prompting.build_prompt(source, audience), options)
        final, reason, _model_result = prompting.sanitize(source, raw)
        return {
            'index': index,
            'source': source,
            'raw': raw,
            'output': final,
            'reason': reason,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for record in pool.map(one, todo):
            done[record['index']] = record
            partial_path.write_text(
                json.dumps({str(i): done[i] for i in sorted(done)}, indent=2)
            )
            completed += 1
            if completed % 25 == 0 or completed == len(todo):
                rate = (time.monotonic() - started) / completed
                remaining = (len(todo) - completed) * rate
                logger.info(
                    '  %d/%d  (%.2fs/sentence, ~%.0fs left)',
                    len(done),
                    len(sources),
                    rate,
                    remaining,
                )

    return [done[i] for i in range(len(sources))]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--condition',
        required=True,
        help='cache key to write, e.g. llm_7b or llm_7b_remote. Must match '
        'the condition name evaluate_sentence.py will score.',
    )
    parser.add_argument('--backend', choices=['ollama', 'openai'], default='ollama')
    parser.add_argument(
        '--base-url',
        default=None,
        help='ollama: defaults to http://127.0.0.1:11434. openai: required, '
        'e.g. https://api.together.xyz/v1',
    )
    parser.add_argument(
        '--model', required=True, help='Ollama tag or provider model id'
    )
    parser.add_argument(
        '--api-style',
        choices=['completions', 'chat'],
        default='completions',
        help="openai backend only. 'completions' matches the local backend's "
        "raw prompt; 'chat' applies the model's chat template and is "
        'therefore not identical input (see module docstring).',
    )
    parser.add_argument(
        '--api-key-env',
        default='LLM_API_KEY',
        help='env var holding the key. The key is never logged or written to '
        'any output file.',
    )
    parser.add_argument(
        '--audience',
        default=prompting.DEFAULT_AUDIENCE.value,
        choices=[a.value for a in Audience],
        help='only the default has ASSET references, so anything else is '
        'reference-free/exploratory (the thesis §6.6)',
    )
    parser.add_argument(
        '--seeds', type=int, default=1, help='generation runs; BLESS §3.4 aggregates 3'
    )
    parser.add_argument(
        '--limit', type=int, default=None, help='first N sentences (smoke runs)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=None,
        help='parallel requests. Defaults to 1 for ollama (single-stream) '
        'and 8 for remote. Lower it if a free tier throttles.',
    )
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument(
        '--outdir',
        default='scratch/eval_methods',
        help='must be the same --outdir evaluate_sentence.py scores from',
    )
    args = parser.parse_args()

    backend = build_backend(args)
    concurrency = args.concurrency or backend.default_concurrency
    audience = Audience(args.audience)
    sources = load_asset_test(args.limit)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not backend.reproducible:
        logger.warning(
            'Hosted endpoint: the served model revision/quantisation cannot be pinned. '
            'Report the endpoint and date with any number from this run (§6.6).'
        )
    if args.backend == 'openai' and args.api_style == 'chat':
        logger.warning(
            'Chat API style wraps BLESS Prompt 2 in a chat template -- not byte-identical '
            'to the raw completion the local backend sends. Prefer --api-style completions.'
        )

    for seed in range(1, args.seeds + 1):
        options = dict(prompting.EVALUATION_DECODING)
        options['seed'] = seed

        partial = outdir / f'{args.condition}_seed{seed}.partial.json'
        final_path = outdir / f'{args.condition}_seed{seed}.json'
        if final_path.exists():
            logger.info('%s already complete, skipping', final_path.name)
            continue

        logger.info(
            '=== %s seed %d: %s via %s (concurrency %d) ===',
            args.condition,
            seed,
            args.model,
            args.backend,
            concurrency,
        )
        started = time.monotonic()
        records = generate_all(
            backend, sources, audience, options, partial, concurrency
        )
        elapsed = time.monotonic() - started

        reasons: Dict[str, int] = {}
        for record in records:
            if record.get('reason'):
                reasons[record['reason']] = reasons.get(record['reason'], 0) + 1

        # Same shape evaluate_sentence.py's run_condition() writes, plus a provenance
        # block. That script reads `outputs` and ignores the rest.
        payload = {
            'condition': args.condition,
            'label': f'{args.model} + BLESS Prompt 2',
            'model_id': args.model,
            'seed': seed,
            'outputs': [r['output'] for r in records],
            'reason_counts': reasons,
            'seconds_total': elapsed,
            'seconds_per_sentence': elapsed / max(len(records), 1),
            'provenance': {
                'backend': args.backend,
                'base_url': args.base_url or 'http://127.0.0.1:11434',
                'api_style': args.api_style if args.backend == 'openai' else 'native',
                'audience': audience.value,
                'decoding': options,
                'fewshot_n': prompting.FEWSHOT_N,
                'n_sentences': len(records),
                # False for hosted endpoints: no digest to pin. See module docstring.
                'reproducible': backend.reproducible,
            },
        }
        final_path.write_text(json.dumps(payload, indent=2))
        # keep the partial: its raw pre-sanitising generations are what an analysis of
        # why a guard fired works from (§6.3)
        logger.info(
            'Wrote %s (%.2fs/sentence, rejected %d)',
            final_path.name,
            payload['seconds_per_sentence'],
            sum(reasons.values()),
        )

    print(
        f'\nDone. Score with:\n  python evaluate_sentence.py --conditions {args.condition} --outdir {args.outdir}'
    )


if __name__ == '__main__':
    main()
