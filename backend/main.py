import asyncio
import logging
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, List, Optional, Protocol, Set, Tuple

import change_guard
import httpx
import pysbd
from bs4 import BeautifulSoup
from document_text import denormalize_from_model, normalize_for_model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prompting import (
    AUDIENCE_LABELS,
    DEFAULT_AUDIENCE,
    Audience,
    build_prompt,
    sanitize,
    serving_decoding,
    tidy_punctuation,
)
from pydantic import BaseModel, Field
from vocabulary import (
    GRANULARITY_SENTENCE_BY_SENTENCE,
    GRANULARITY_WHOLE_SECTIONS,
    METHOD_FINE_TUNED_SEQ2SEQ,
    METHOD_PROMPTED_LLM,
)

BATCH_SIZE = 8
BATCH_TIMEOUT = 0.1  # seconds to wait for batch to fill
CACHE_MAX = 1024  # LRU cache size limit

# Ollama sidecar for the "llm_*" keys. Out of process: a 7B in fp16 is ~14 GB and does
# not fit beside the two BART models on a 16 GB machine; as a 4-bit GGUF under Ollama
# it is ~5 GB and stays warm. See docs/roadmap-open-llm.md.
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434')
# Per-request ceiling. ~1-3 s/sentence for a 7B on Apple Silicon, but long input or a
# cold model can be far slower; without a cap one slow sentence holds a fetch open.
OLLAMA_TIMEOUT = float(os.environ.get('OLLAMA_TIMEOUT', '120'))


def _parse_keep_alive(raw: str) -> Any:
    """Coerce a keep_alive env value into the JSON type Ollama expects.

    Ollama accepts ``keep_alive`` either as a Go duration *string* ("10m", "1h30m") or
    as a *number* of seconds, where -1 means "never unload". The two are not
    interchangeable: a string is always parsed as a duration, so sending the string
    "-1" fails with ``time: missing unit in duration "-1"`` -- an HTTP 400 on every
    single generation.

    Measured 2026-08-14: passing the env var through as a string broke the whole
    prompted-LLM path. Fake-transport tests cannot catch this, since a fake accepts
    any payload.
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        pass
    try:
        return float(raw)
    except (TypeError, ValueError):
        # duration string like "10m"; Ollama validates it
        return raw


# -1 pins the model in memory indefinitely, so only the very first request pays the
# ~2-4 s load cost.
OLLAMA_KEEP_ALIVE = _parse_keep_alive(os.environ.get('OLLAMA_KEEP_ALIVE', '-1'))

# Simplifier models, all loaded at startup so the extension can switch per request
# without a reload delay.
#
# - "online" / "finetuned" / "document": seq2seq (BART) checkpoints, as a local
#   checkpoint directory (e.g. from fine_tune.py or the training notebook) or a Hub repo
#   id. "finetuned" and "document" default to this project's own checkpoints; local
#   paths resolve from backend/, the working directory of `run_dev.sh` and
#   `uvicorn main:app`.
# - "llm_7b" / "llm_3b" / "llm_doc_7b" / "llm_doc_3b": instruction-tuned models served by
#   Ollama and prompted at inference time (see prompting.py); the model id is an Ollama
#   tag. The "_doc_" pair is the same tags at document scope, where granularity selects
#   the prompt.
#
# Set the env var to override a model, or to an empty string to skip it (requests for
# it then get a 400, no silent fallback).
#
# `method` and `granularity` are defined in vocabulary.py; `method` also selects the
# loader in lifespan().
#
# (env var name, default, method, granularity) per key; read in lifespan(), not at
# import, so tests can monkeypatch the env.
MODEL_ENV_CONFIG = {
    'online': (
        'SIMPLIFIER_MODEL_ONLINE',
        'eilamc14/bart-large-text-simplification',
        METHOD_FINE_TUNED_SEQ2SEQ,
        GRANULARITY_SENTENCE_BY_SENTENCE,
    ),
    # On the Hub, not in the repo: the 532 MB safetensors file exceeds GitHub's 100 MB
    # per-file limit (even with free-tier LFS). `from_pretrained` also takes a local
    # path, e.g. research/scratch/simplification_results/best_checkpoint for local runs.
    'finetuned': (
        'SIMPLIFIER_MODEL_FINETUNED',
        'yunvs/bart-base-wikilarge-simplification',
        METHOD_FINE_TUNED_SEQ2SEQ,
        GRANULARITY_SENTENCE_BY_SENTENCE,
    ),
    # Two sizes as separate keys, not one key with a size setting: they produce
    # different output, cache separately and differ ~2x in latency, so the per-key
    # machinery (picker, cache, /health, error states) applies unchanged.
    #
    # Tags checked against `ollama list` 2026-08-14; digests recorded because Ollama
    # tags are mutable. Measured on Apple M3, 11.8 GiB VRAM:
    #
    #   7B  845dbda0ea48   1.85 s/sentence idle,  ~10 s/sentence under load, 17.3 tok/s
    #   3B  357c53fb659c   0.91 s/sentence idle,  ~5 s/sentence under load,  39.3 tok/s
    #
    # 3B is the better interactive choice, 7B the better output. Both stay resident
    # (keep_alive -1) at 4.7 + 1.9 GB, which fits beside the seq2seq checkpoints.
    #
    # If Ollama isn't running or a tag isn't pulled, startup logs it and skips the key;
    # the extension shows that as "never configured".
    'llm_7b': (
        'SIMPLIFIER_MODEL_LLM_7B',
        'qwen2.5:7b-instruct-q4_K_M',
        METHOD_PROMPTED_LLM,
        GRANULARITY_SENTENCE_BY_SENTENCE,
    ),
    'llm_3b': (
        'SIMPLIFIER_MODEL_LLM_3B',
        'qwen2.5:3b-instruct-q4_K_M',
        METHOD_PROMPTED_LLM,
        GRANULARITY_SENTENCE_BY_SENTENCE,
    ),
    # The same two served models at document scope. Prompted, so granularity selects a
    # prompt template: unlike the "document" checkpoint there is no 512-token training
    # ceiling, only a context window (8192, prompting.DOCUMENT_SERVING_DECODING), and no
    # document_text.py lowercasing/pre-tokenization (that matches D-Wikipedia's
    # convention; an LLM wants ordinary prose).
    #
    # Same tags as the sentence keys, so enabling them costs no extra pull or VRAM
    # (Ollama keeps one resident copy per tag). Separate keys for the same reason as
    # the two sizes: each caches, fails and reports health on its own.
    #
    # Overridable independently, but the picker's card names the family from the
    # *sentence* tag, so keep them in step unless the divergence is being measured.
    'llm_doc_7b': (
        'SIMPLIFIER_MODEL_LLM_DOC_7B',
        'qwen2.5:7b-instruct-q4_K_M',
        METHOD_PROMPTED_LLM,
        GRANULARITY_WHOLE_SECTIONS,
    ),
    'llm_doc_3b': (
        'SIMPLIFIER_MODEL_LLM_DOC_3B',
        'qwen2.5:3b-instruct-q4_K_M',
        METHOD_PROMPTED_LLM,
        GRANULARITY_WHOLE_SECTIONS,
    ),
    # Trained on D-Wikipedia whole-article pairs: rewrites the document as a whole
    # (deleting, merging, reordering sentences), so output can't be mapped back onto
    # the input position by position. Lowercased + pre-tokenized in and out; see
    # document_text.py. Hub-hosted like "finetuned".
    'document': (
        'SIMPLIFIER_MODEL_DOCUMENT',
        # Full-scope checkpoint (131,739 documents, 5 epochs), since 2026-08-22. Its
        # 20k/2-epoch predecessor substituted a memorised census year for the input's
        # in 77% of affected documents (this one: 0.2%). Trade-off: heavier deletion
        # (median output/source length ratio 0.70), often dropping a date rather than
        # simplifying its sentence. The predecessor stays on the Hub because
        # research/results/RESULTS.md rows D1/D2/D4/D2' cite it.
        'yunvs/bart-base-dwikipedia-simplification-full',
        METHOD_FINE_TUNED_SEQ2SEQ,
        GRANULARITY_WHOLE_SECTIONS,
    ),
}

# Tokenizer/generation max_length (tokens) per granularity, matching training in
# research/notebooks/train_sentence_and_document_pipeline.ipynb. Chosen from measured
# token-length percentiles, under BART's 1024-token positional limit (thesis §4.5):
# - sentence (WikiLarge): mean ~33, p90 ~52, max ~130; longer sentences are
#   truncated, as in training.
# - document (D-Wikipedia): source mean ~172, p95 ~499, p99 ~742; 512 truncates
#   roughly the longest 5%. No chunking, only truncation, here and in training.
SEQ2SEQ_MAX_LENGTH = {
    GRANULARITY_SENTENCE_BY_SENTENCE: 64,
    GRANULARITY_WHOLE_SECTIONS: 512,
}

_WORD_RE = re.compile(r"[A-Za-z']+")

# clean=False: report boundaries without rewriting the input, so layout is preserved.
_sentence_segmenter = pysbd.Segmenter(language='en', clean=False)


def split_sentences(text: str) -> List[str]:
    """Splits prose into individual sentences for "sentence_by_sentence" models.

    Falls back to the whole text as a single "sentence" if segmentation finds no
    boundaries (e.g. empty input, or a fragment with no terminal punctuation) so no
    content is ever dropped.
    """
    sentences = [s.strip() for s in _sentence_segmenter.segment(text)]
    sentences = [s for s in sentences if s]
    return sentences or [text.strip()]


def split_sentences_with_whitespace(text: str) -> List[Tuple[str, str]]:
    """Split text into sentence-like units while preserving the original
    whitespace separators between them.

    We keep the exact whitespace that followed each chunk in the source text (new
    lines, tabs, blank lines, and single spaces) so a sentence-level simplifier can
    reassemble the output without flattening the layout.
    """
    chunks: List[Tuple[str, str]] = []

    for raw in _sentence_segmenter.segment(text):
        raw = raw or ''
        if not raw.strip():
            continue

        body = raw.rstrip()
        trailing_ws = raw[len(body) :]

        # Split chunks holding several fragments (punctuation + whitespace, or line/tab
        # breaks) so each separator survives: "B. C." and "A\nB\tC" keep theirs, while
        # "This is sentence one." stays whole.
        if any(ch in body for ch in '\n\r\t') or re.search(r'[.!?]\s+\S', body):
            if any(ch in body for ch in '\n\r\t'):
                pieces = re.split(r'(\s+)', body)
            else:
                pieces = re.split(r'(?<=[.!?])(\s+)', body)
            current = ''
            separator = ''
            for piece in pieces:
                if not piece:
                    continue
                if piece.isspace():
                    separator += piece
                    continue
                if current:
                    chunks.append((current, separator))
                current = piece
                separator = ''
            if current:
                chunks.append((current, trailing_ws))
            continue

        chunks.append((body.strip(), trailing_ws))

    if not chunks:
        stripped = text.strip()
        return [(stripped, '')] if stripped else []
    return chunks


def _looks_like_hallucination(original: str, simplified: str) -> bool:
    """Detects the model substituting a short, isolated fragment (e.g. nav/meta
    cruft like "By", "on", "Published") with a fabricated, unrelated phrase
    (observed in practice: several unrelated one-to-three-word inputs all
    coming back as "Other websites") instead of leaving it alone or lightly
    rephrasing it. Heuristic: inputs this short carry too little context for
    the model to work with reliably, so if the output shares none of the
    input's words, treat it as a hallucination rather than show the user a
    non-sequitur.

    This is the seq2seq path's guard. The prompted-LLM path has its own, with a
    different set of failure modes -- see ``prompting.classify_output``. The
    length gate here is what ``_looks_like_corpus_artifact`` below exists to cover:
    the same fabricated phrase also turns up on inputs too long for this rule.
    """
    original_words = {w.lower() for w in _WORD_RE.findall(original)}
    if not original_words or len(original_words) > 3:
        return False
    simplified_words = {w.lower() for w in _WORD_RE.findall(simplified)}
    return original_words.isdisjoint(simplified_words)


# Boilerplate the fine-tuned checkpoints emit as a whole answer instead of a
# simplification. WikiLarge's simple side is Simple English Wikipedia text, where the
# external-links section is headed "Other websites" and see-also "Other pages", so both
# occur in training as short standalone lines. Compared normalized (lowercase, letters
# only), so "Other websites." matches too.
#
# Measured 2026-08-24, sentence checkpoint, 87 heading-like inputs ("See also",
# "Contact us", "Site map", "Bin collection days"): 65 came back as one of these
# (53 "other websites", 12 "other pages"); the document checkpoint 0 of 87.
# _looks_like_hallucination caught 61 of the 65. The 4 that reached the page were all
# "Other pages": the input shared "pages" ("Helpful pages", "Popular pages", "Special
# pages") or exceeded its three-word limit ("You may also like").
#
# Not included: "related pages" (never observed from this checkpoint; entries are added
# on evidence) and "references" (produced twice, but can be a correct simplification of
# a short input like "Bibliography").
_CORPUS_ARTIFACT_OUTPUTS = frozenset({'other websites', 'other pages'})

CORPUS_ARTIFACT = 'corpus_artifact'


def _looks_like_corpus_artifact(original: str, simplified: str) -> bool:
    """Whether the model's entire answer is one of those phrases.

    Separate from ``_looks_like_hallucination`` because it needs no length gate and
    no word-overlap test: this is not "the output might be unrelated to a fragment
    too short to judge", it is a string that is never a simplification of anything.
    Observed on inputs well past the three-word ceiling of the rule above --
    "Municipal recycling policy revisions" came back as "Other websites" -- and
    every such case is a fabrication, so the input is served instead.

    Matched on the *whole* output, not searched for inside it: a sentence that
    happens to mention other websites is a real simplification and is left alone.
    An input that is itself the phrase is left alone as well -- nothing was
    fabricated there, and unchanged output is already reported as unchanged.
    """

    def normalized(text: str) -> str:
        return ' '.join(w.lower() for w in _WORD_RE.findall(text))

    return (
        normalized(simplified) in _CORPUS_ARTIFACT_OUTPUTS
        and normalized(original) not in _CORPUS_ARTIFACT_OUTPUTS
    )


# Simplifier implementations
#
# Both methods implement one interface, so the queue/batching/caching layer is
# method-agnostic. `generate` returns one (text, fallback_reason, model_result) per
# input: `fallback_reason` is None on success, else a short code for why the output was
# rejected and the original served. The codes appear in the API response and feed the
# error categorisation in thesis §6.3.
#
# `model_result` is the model's own output, kept even when a guard rejected it: the
# middle of the input -> model result -> output chain in the extension's History log.


# Device for the seq2seq checkpoints. `auto` picks CUDA if available, else CPU; it does
# not pick MPS on Apple Silicon.
#
#   (2026-08-26, `finetuned` bart-base, four fresh uvicorn processes, device order
#    swapped between rounds, uncached byte-identical inputs, round-to-round spread <3%)
#
#     one request at a time       535 ms/sentence cpu   against    614 ms mps
#     eight in flight              80 ms/sentence cpu   against    136 ms mps
#     whole sections             1652 ms/section  cpu   against   1963 ms mps
#
# Eight in flight is what the extension issues: a batch of eight fills the CPU cores,
# while a model this small never fills the GPU and MPS keeps paying per-call dispatch.
# MPS is also not bit-identical to the CPU numbers in research/results/
# (research/evaluate_sentence.py stays on CPU for that reason), and can deadlock
# against the GIL (see _SEQ2SEQ_EXECUTOR).
#
# `SIMPLIFIER_DEVICE=mps` still selects it explicitly. CUDA is untouched: not measured,
# and the deadlock is MPS-specific.
def _resolve_serving_device() -> str:
    # Empty counts as unset: `SIMPLIFIER_DEVICE="$UNSET_VAR"` or a blank env-file line
    # would otherwise fail every checkpoint load with "Device string must not be empty".
    requested = os.environ.get('SIMPLIFIER_DEVICE', 'auto').strip()
    if requested and requested != 'auto':
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        # no torch (stand-in servers). Logged at debug so a broken torch is traceable.
        logging.debug(
            'torch unavailable for device detection; falling back to cpu', exc_info=True
        )
    return 'cpu'


# `torch.inference_mode`, resolved lazily and cached; a no-op without torch (the test
# stand-ins). No module-level `import torch`: torch/transformers stay out of import
# time (lifespan imports them) so the tests and `--reload` don't pay for them.
_TORCH_INFERENCE_MODE: Any = None


def _inference_mode() -> Any:
    global _TORCH_INFERENCE_MODE
    if _TORCH_INFERENCE_MODE is None:
        try:
            import torch

            _TORCH_INFERENCE_MODE = torch.inference_mode
        except Exception:
            from contextlib import nullcontext

            _TORCH_INFERENCE_MODE = nullcontext
    return _TORCH_INFERENCE_MODE()


# All seq2seq generation runs on this one thread. Not for throughput: it avoids an MPS
# deadlock. torch's MPS backend takes Apple's dispatch queue and the GIL in opposite
# orders on two threads:
#
#   the generating thread   holds the GIL, then calls MPSStream::addCompletedHandler,
#                           which dispatch_sync's onto the MPS serial queue and blocks
#                           waiting to own it
#   the "metal gpu stream"  owns that queue, runs a completion block, and needs the GIL
#                           to incref a tensor's Python object
#
# The process then stays up holding its socket at ~0% CPU and never answers again.
#
# Needs two threads submitting MPS work (one worker per model key plus a thread pool
# made that the normal case). Measured on this machine with the same tokenize ->
# generate -> decode: one checkpoint on one thread ran 746 batches in 10 min clean; two
# checkpoints on two threads wedged before the 25th batch; two checkpoints through one
# thread ran 308 batches with the event loop still responsive.
#
# Work still runs off the event loop, so /health stays answerable. Serialising across
# models costs nothing: a page uses one model at a time and a comparison runs its arms
# in sequence.
_SEQ2SEQ_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='seq2seq')


def _unpadded_lengths(masks: Any, fallback: Any) -> List[int]:
    """How many real tokens each padded row of a batch actually holds.

    The attention mask is 1 per real token and 0 per pad, so a row's sum is its own
    length -- the one number a per-row truncation check can be made against once the
    batch has been padded to its longest member.

    Summed for the whole batch in one op rather than element by element in Python. The
    obvious `sum(int(v) for v in row)` is 280x slower on a CPU batch and ~7x slower again
    on an MPS one, where every `int()` is a separate device sync -- and all of it holds
    the GIL, on the one code path that must not, since it runs in the worker thread that
    the event loop needs the GIL back from to answer /health at all.

    Falls back to the padded row length for a stand-in tokenizer whose mask isn't a
    tensor (or isn't there): an over-report is a warning that names one text too many,
    where an exception here would take down a real generation over a log line.
    """
    try:
        return [int(n) for n in masks.sum(dim=1).tolist()]
    except (AttributeError, TypeError, RuntimeError):
        pass
    try:
        return [int(sum(int(v) for v in row)) for row in masks]
    except (TypeError, ValueError):
        return [len(row) for row in fallback]


class Simplifier(Protocol):
    model_id: str
    # Whether `generate` varies its output with `audience`. False for seq2seq
    # checkpoints, whose behaviour is fixed at fine-tuning time.
    supports_audience: bool
    # Largest group of inputs `generate` should be handed at once.
    batch_size: int
    # One of vocabulary.GRANULARITIES; simplify_text() uses it to decide whether to
    # split the text into sentences or pass it through unsplit.
    granularity: str
    # One of vocabulary.METHODS, fixed per implementation. Published via
    # /health.methods so the extension needn't keep its own mapping.
    method: str

    async def generate(
        self, texts: List[str], audience: Audience
    ) -> List[Tuple[str, Optional[str], str]]: ...

    async def aclose(self) -> None: ...


class Seq2SeqSimplifier:
    """Wraps a Hugging Face seq2seq checkpoint (the "online", "finetuned" and
    "document" keys).

    Batched tokenization and one `generate` call for the whole batch, then a
    per-granularity finish: sentence output is punctuation-tidied and put through the
    short-input hallucination guard, while whole_sections output is de-normalized back
    out of the corpus conventions document_text.py applied on the way in, and rejected
    only if it came back empty.
    """

    supports_audience = False
    method = METHOD_FINE_TUNED_SEQ2SEQ

    def __init__(
        self,
        model_id: str,
        tokenizer: Any,
        model: Any,
        granularity: str = GRANULARITY_SENTENCE_BY_SENTENCE,
        max_length: int = 512,
        device: str = 'cpu',
    ):
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = BATCH_SIZE
        self.granularity = granularity
        self.max_length = max_length
        self.device = device
        # Moved once, not per generation. `hasattr`: the stand-in models
        # (dev_stub_server.py, test fakes) have no device.
        if hasattr(model, 'to'):
            model.to(device)

    async def generate(
        self, texts: List[str], audience: Audience
    ) -> List[Tuple[str, Optional[str], str]]:
        # `audience` is ignored: a fine-tuned checkpoint has no control for it. The
        # request layer normalizes it away for these models so the cache doesn't
        # fragment.
        is_document = self.granularity == GRANULARITY_WHOLE_SECTIONS
        # The document checkpoint's corpus is lowercased and PTB-pre-tokenized; mixed-case
        # prose triggered a factual hallucination in testing. Match the training style
        # going in, undo it coming out (see document_text.py).
        model_inputs = [normalize_for_model(t) for t in texts] if is_document else texts

        # Tokenize/generate/decode are blocking, so they run off the event loop to keep
        # other requests and /health served during a batch (torch releases the GIL for
        # the heavy ops). _SEQ2SEQ_EXECUTOR's single thread because of the MPS deadlock
        # documented there.
        raw_results = await asyncio.get_running_loop().run_in_executor(
            _SEQ2SEQ_EXECUTOR, self._run_model, model_inputs
        )

        # The third element is always the model's own (de-normalized/tidied) answer,
        # never the input, even when a guard rejects it.
        results: List[Tuple[str, Optional[str], str]] = []
        for text, raw in zip(texts, raw_results, strict=True):
            if is_document:
                # the original is the casing reference; the model emits only lowercase
                restored = denormalize_from_model(raw, text)
                # deleting content is legitimate for a document model; emptying the
                # whole document is a failure
                if restored.strip():
                    results.append((restored, None, restored))
                else:
                    results.append((text, 'empty', restored))
                continue
            tidied = tidy_punctuation(raw)
            if _looks_like_hallucination(text, tidied):
                results.append((text, 'hallucination', tidied))
            else:
                results.append((tidied, None, tidied))
        return results

    def _run_model(self, model_inputs: List[str]) -> List[str]:
        """The blocking half of `generate`: tokenize, generate, decode.

        Runs in a worker thread (see the caller), so it must stay free of anything that
        touches the event loop or the shared cache.
        """
        inputs = self.tokenizer(
            model_inputs,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        self._warn_if_truncated(model_inputs, inputs)
        # a stand-in tokenizer returns a plain dict, not a BatchEncoding
        if hasattr(inputs, 'to'):
            inputs = inputs.to(self.device)
        # no autograd graph: saves time and memory with three checkpoints resident
        with _inference_mode():
            outputs = self.model.generate(**inputs, max_length=self.max_length)
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def _warn_if_truncated(self, model_inputs: List[str], encoded: Any) -> None:
        """Logs when input hit `max_length`, since the overflow is silently dropped.

        Truncation isn't recoverable here -- whatever fell off the end is simply
        absent from the output, which for a document model means page content
        disappearing rather than being simplified. Callers are expected to chunk
        below this ceiling (the extension does); this is the safety net that makes
        it visible when something slipped through.

        Measured per row from the attention mask, not the id count: with
        `padding=True` every row has the batch's length, so one over-long input would
        flag every text batched beside it.
        """
        try:
            input_ids = encoded['input_ids']
        except (TypeError, KeyError):  # a stand-in tokenizer in tests
            return
        try:
            masks = encoded['attention_mask']
        except (TypeError, KeyError):  # a tokenizer that returns ids alone
            masks = None
        lengths = (
            _unpadded_lengths(masks, input_ids)
            if masks is not None
            else [len(ids) for ids in input_ids]
        )
        for text, length in zip(model_inputs, lengths, strict=True):
            if length >= self.max_length:
                logging.warning(
                    "input hit the %s-token limit for '%s' (%d chars) -- the overflow was "
                    'dropped, not simplified; chunk the input smaller',
                    self.max_length,
                    self.model_id,
                    len(text),
                )

    async def aclose(self) -> None:
        return None


class PromptedLLMSimplifier:
    """Prompts an open-weight instruction-tuned LLM served by Ollama.

    One HTTP request per sentence: llama.cpp (which Ollama wraps) serves a single
    stream per model instance, so the seq2seq path's trick of handing the model a
    padded batch has no equivalent here -- hence ``batch_size = 1``.

    That also happens to be what keeps mixed-audience batches impossible: a batch of
    one can only ever carry one audience. ``_register`` enforces the general form of
    that invariant.
    """

    supports_audience = True
    method = METHOD_PROMPTED_LLM
    batch_size = 1

    def __init__(
        self,
        model_id: str,
        client: httpx.AsyncClient,
        granularity: str = GRANULARITY_SENTENCE_BY_SENTENCE,
    ):
        self.model_id = model_id
        self.client = client
        # Selects prompt template, stop sequences, token budget and sanitiser rules.
        # Unlike seq2seq (fixed by training), granularity here is a prompt choice, so one
        # served model backs both a sentence and a document condition, with no 512-token
        # training ceiling (see SEQ2SEQ_MAX_LENGTH), only a context window.
        self.granularity = granularity
        self._decoding = serving_decoding(granularity)

    async def _generate_one(
        self, text: str, audience: Audience
    ) -> Tuple[str, Optional[str], str]:
        payload = {
            'model': self.model_id,
            'prompt': build_prompt(text, audience, granularity=self.granularity),
            'stream': False,
            'options': self._decoding,
            'keep_alive': OLLAMA_KEEP_ALIVE,
        }
        try:
            response = await self.client.post('/api/generate', json=payload)
            response.raise_for_status()
            raw = response.json().get('response', '')
        except httpx.TimeoutException:
            logging.warning('Ollama request timed out after %ss', OLLAMA_TIMEOUT)
            return text, 'timeout', ''
        except httpx.HTTPStatusError as exc:
            # Ollama's reason is in the body ({"error": "..."}), which
            # raise_for_status() discards; it is what distinguishes a malformed payload
            # from a missing model (the keep_alive bug above surfaced only as "HTTP 400").
            logging.error(
                'Ollama rejected the request (HTTP %s): %s',
                exc.response.status_code,
                exc.response.text[:500],
            )
            # no generation, so model_result is empty (unlike a rejected generation)
            return text, 'error', ''
        except Exception:
            logging.exception('Ollama request failed')
            return text, 'error', ''
        return sanitize(text, raw, self.granularity)

    async def generate(
        self, texts: List[str], audience: Audience
    ) -> List[Tuple[str, Optional[str], str]]:
        # Sequential: the sidecar serves one stream at a time, so concurrency would only
        # queue inside Ollama and blur failure attribution. With batch_size 1 this runs
        # once; a loop to match the seq2seq interface.
        return [await self._generate_one(text, audience) for text in texts]

    async def aclose(self) -> None:
        await self.client.aclose()


def _make_ollama_client() -> httpx.AsyncClient:
    """Factory for the sidecar HTTP client.

    Kept as a separate function purely so tests can swap in a fake without
    monkeypatching ``httpx.AsyncClient`` globally (which the test client itself
    builds on).
    """
    return httpx.AsyncClient(base_url=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)


async def _probe_ollama(client: httpx.AsyncClient, model_id: str) -> None:
    """Confirm the sidecar is up and the requested tag is actually pulled.

    Raises with an actionable message; the caller turns that into "this model isn't
    available" rather than letting the first user request fail with a 500.
    """
    response = await client.get('/api/tags')
    response.raise_for_status()
    available = [m.get('name', '') for m in response.json().get('models', [])]
    if model_id not in available:
        raise RuntimeError(
            f"Ollama is running but has no model tagged '{model_id}'. "
            f'Pull it with `ollama pull {model_id}`. Currently available: '
            f"{', '.join(available) or 'none'}"
        )


# Device the seq2seq weights were actually moved to, for /health (not re-resolved from
# the env at request time).
serving_device: str = 'cpu'

# Initialized in lifespan().
# model key -> Simplifier; present only if that model loaded, so an unconfigured or
# failed model is absent rather than None.
simplifiers: Dict[str, Simplifier] = {}
# (model_key, audience, cleaned text) -> (simplified, links, fallback_reason,
# model_result). Model and audience are in the key because both change the output.
# `model_result` is cached so a hit reports the same input -> model result -> output
# chain as a fresh generation.
cache: 'OrderedDict[Tuple[str, str, str], Tuple[str, List[Dict[str, str]], Optional[str], str]]' = (OrderedDict())
cache_lock: asyncio.Lock | None = None

# Page ownership of cache entries, in both directions.
#
# The cache key ignores pages: the same sentence on two pages is one generation. But
# the extension's History log keeps only the last 20 pages (background.js's
# MAX_HISTORY_PAGES); entries for pages outside that window are unreachable yet still
# take CACHE_MAX slots, so the log's eviction is mirrored here.
#
# page -> keys (to drop a page's entries) and key -> pages (to check whether another
# page still holds it). An entry goes only when its last holder does.
#
# Requests naming no page (toggle preflight, History comparison runs) are unowned and
# governed only by CACHE_MAX's LRU.
CacheKey = Tuple[str, str, str]
cache_pages: Dict[str, Set[CacheKey]] = {}
cache_key_pages: Dict[CacheKey, Set[str]] = {}
# one queue/worker per loaded model key, so a batch never mixes models.
# queue items are (clean_text, links, audience, Future).
batch_queues: Dict[str, asyncio.Queue] = {}
worker_tasks: Dict[str, asyncio.Task] = {}


def _attribute_cache_entry(cache_key: CacheKey, page: Optional[str]) -> None:
    """Records that ``page`` is holding ``cache_key``.

    Called on fresh generations and on cache hits: a page served entirely from cache
    holds those entries too, so they must survive the generating page aging out.

    Callers must hold ``cache_lock``.
    """
    if not page:
        return
    cache_pages.setdefault(page, set()).add(cache_key)
    cache_key_pages.setdefault(cache_key, set()).add(page)


def _forget_cache_key(cache_key: CacheKey) -> None:
    """Drops one key's page bookkeeping, for a key that has left the cache by some route
    other than page eviction -- LRU overflow, or a wholesale ``/cache/clear``. Without
    this the index would keep naming entries that are no longer there, and a later page
    eviction would report dropping them a second time.

    Callers must hold ``cache_lock``.
    """
    for page in cache_key_pages.pop(cache_key, set()):
        keys = cache_pages.get(page)
        if keys is None:
            continue
        keys.discard(cache_key)
        if not keys:
            del cache_pages[page]


def _drop_cache_pages(pages: Iterable[str]) -> Tuple[int, int]:
    """Forgets the given pages and removes every cache entry no other page still holds.
    Returns ``(entries_dropped, pages_matched)``.

    ``pages_matched`` counts the ids that were actually holding something, which is not
    the same as how many were asked for: a page simplified before this process started,
    or one whose every unit was too short to send, holds nothing. Reporting the two
    separately is what lets a caller tell "nothing to drop" from "dropped nothing".

    Callers must hold ``cache_lock``.
    """
    entries_dropped = 0
    pages_matched = 0
    for page in pages:
        keys = cache_pages.pop(page, None)
        if keys is None:
            continue
        pages_matched += 1
        for key in keys:
            holders = cache_key_pages.get(key)
            if holders is None:
                continue
            holders.discard(page)
            if holders:
                continue  # another page is still reading this generation
            del cache_key_pages[key]
            if cache.pop(key, None) is not None:
                entries_dropped += 1
    return entries_dropped, pages_matched


def _document_max_tokens(simplifier: 'Simplifier') -> Optional[int]:
    """Input-token ceiling for a document request to this model, or None if it isn't a
    document model.

    Two different kinds of limit behind one number:

    - **seq2seq**: the trained `max_length`. Input past it is truncated, i.e. dropped
      rather than simplified, so the caller must chunk below it.
    - **prompted LLM**: the context window, shared between prompt and completion. The
      budget reported is the window minus what the instruction and the reserved output
      need, so a caller chunking to this figure leaves room for the answer. Deliberately
      conservative -- overrunning a context window drops the *start* of the prompt,
      which is the instruction itself, and produces plausible-looking output that
      followed no instruction at all.
    """
    if simplifier.granularity != GRANULARITY_WHOLE_SECTIONS:
        return None
    decoding = getattr(simplifier, '_decoding', None)
    if not decoding:
        return getattr(
            simplifier, 'max_length', SEQ2SEQ_MAX_LENGTH[GRANULARITY_WHOLE_SECTIONS]
        )
    num_ctx = int(decoding.get('num_ctx', 8192))
    num_predict = int(decoding.get('num_predict', 1024))
    # ~200 tokens covers the rendered instruction with headroom for a longer
    # audience clause; the rest of the window is the caller's to fill.
    return max(256, num_ctx - num_predict - 200)


def model_ids() -> Dict[str, str]:
    """Model key -> the actual model id used for it (Hugging Face repo id, local
    checkpoint path, or Ollama tag). Lets callers (e.g. the extension's UI) show the
    real model name instead of just the abstract key."""
    return {key: s.model_id for key, s in simplifiers.items()}


def _register(model_key: str, simplifier: Simplifier) -> None:
    """Make a successfully-initialized simplifier live, with its own queue+worker."""
    if simplifier.supports_audience and simplifier.batch_size != 1:
        # A multi-item batch could mix audiences, but one `generate` call renders one
        # prompt for the whole batch, so audience-sensitive simplifiers take one item
        # at a time.
        raise ValueError(
            f"'{model_key}' supports audiences, so it must use batch_size=1 "
            f'(got {simplifier.batch_size})'
        )
    simplifiers[model_key] = simplifier
    batch_queues[model_key] = asyncio.Queue()
    worker_tasks[model_key] = asyncio.create_task(_batch_worker(model_key))


async def _batch_worker(model_key: str):
    """Background task that reads from ``batch_queues[model_key]`` and performs
    grouped inference on ``simplifiers[model_key]``.  Each item is a
    ``(clean_text, links, audience, Future)`` tuple; when the model finishes we set
    the corresponding future result and update the cache with the simplification,
    link metadata, and any fallback reason.
    """
    queue = batch_queues[model_key]
    simplifier = simplifiers[model_key]
    global cache

    while True:
        try:
            item = await queue.get()
        except asyncio.CancelledError:
            break

        items = [item]
        start = asyncio.get_event_loop().time()
        while len(items) < simplifier.batch_size:
            remaining = BATCH_TIMEOUT - (asyncio.get_event_loop().time() - start)
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(queue.get(), timeout=remaining)
                items.append(nxt)
            except asyncio.TimeoutError:
                break

        texts = [t for t, _, _, _ in items]
        # Safe because audience-sensitive simplifiers are pinned to batch_size=1 (see
        # _register), so any batch larger than one is audience-agnostic.
        audience = items[0][2]

        try:
            results = await simplifier.generate(texts, audience)
        except Exception:
            logging.exception('Batch inference error')
            for _, _, _, f in items:
                if not f.done():
                    f.set_exception(RuntimeError('batch inference failed'))
            continue

        # Final guards, applied once here so both methods are covered and the cache
        # stores what is served. Skipped where the model's own guard already set a
        # reason. `model_result` is never altered: only what is served changes.
        guarded: List[Tuple[str, Optional[str], str]] = []
        for text, (simplified, reason, model_result) in zip(
            texts, results, strict=True
        ):
            if reason:
                guarded.append((simplified, reason, model_result))
                continue
            if _looks_like_corpus_artifact(text, simplified):
                # checked here, not in the seq2seq guard, so it covers both methods
                # and granularities
                guarded.append((text, CORPUS_ARTIFACT, model_result))
                continue
            served, guard_reason = change_guard.apply(text, simplified)
            guarded.append((served, guard_reason, model_result))
        results = guarded

        async with cache_lock:  # type: ignore
            for (text, links, item_audience, fut), (
                simplified,
                reason,
                model_result,
            ) in zip(items, results, strict=True):
                # `.value`, not `str()`: `str(member)` goes through `Enum.__str__`
                # ("Audience.CHILDREN"), which would never match the request path's
                # key and silently disable the cache.
                cache[(model_key, Audience(item_audience).value, text)] = (
                    simplified,
                    links,
                    reason,
                    model_result,
                )
                # LRU eviction; the page index must drop the same key
                if len(cache) > CACHE_MAX:
                    evicted_key, _ = cache.popitem(last=False)
                    _forget_cache_key(evicted_key)
                if not fut.done():
                    fut.set_result((simplified, reason, model_result))


# Renamed model env vars. A stale override would be silently ignored while the default
# loads, so warn once per old name (without keeping it as an alias).
RENAMED_MODEL_ENV_VARS = {'SIMPLIFIER_MODEL_LOCAL': 'SIMPLIFIER_MODEL_FINETUNED'}


def _warn_about_renamed_env_vars() -> None:
    for old_name, new_name in RENAMED_MODEL_ENV_VARS.items():
        if os.environ.get(old_name) is None:
            continue
        logging.warning(
            f'{old_name} is set but no longer read -- it was renamed to {new_name} '
            f"(the model key 'local' is now 'finetuned'). Its value is being ignored."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes each configured model and starts one batch worker per model that
    came up successfully. On shutdown all workers are cancelled and any HTTP clients
    closed."""
    global simplifiers, cache, cache_lock, cache_pages, cache_key_pages, batch_queues, worker_tasks
    global serving_device

    cache = OrderedDict()
    cache_lock = asyncio.Lock()
    cache_pages = {}
    cache_key_pages = {}
    batch_queues = {}
    worker_tasks = {}
    simplifiers = {}

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _warn_about_renamed_env_vars()

    # The device decides serving speed and whether output matches the offline harness
    # (see _resolve_serving_device), so it is logged as a warning: uvicorn's default
    # config hides root-logger info lines.
    serving_device = _resolve_serving_device()
    if serving_device == 'cpu':
        logging.warning(
            "Serving seq2seq checkpoints on 'cpu' (fastest on this path, and identical to "
            'the numbers in research/results/).'
        )
    elif serving_device == 'mps':
        # only reachable explicitly, so say what is given up
        logging.warning(
            "Serving seq2seq checkpoints on 'mps' because SIMPLIFIER_DEVICE asked for it. "
            "Measured slower than 'cpu' on this path (136 against 80 ms/sentence at the "
            'served batch width), not bit-identical to research/results/, and exposed to a '
            "torch MPS/GIL deadlock. Unset SIMPLIFIER_DEVICE for 'cpu'."
        )
    else:
        logging.warning(
            f"Serving seq2seq checkpoints on '{serving_device}'. Not bit-identical to the "
            f'CPU numbers in research/results/; set SIMPLIFIER_DEVICE=cpu to match them '
            f'exactly.'
        )

    for model_key, (env_name, default, method, granularity) in MODEL_ENV_CONFIG.items():
        model_id = os.environ.get(env_name, default)
        if not model_id:
            logging.info(
                f"Skipping '{model_key}' simplifier: no model configured (set {env_name})."
            )
            continue
        try:
            if method == METHOD_FINE_TUNED_SEQ2SEQ:
                # No `revision=`: `model_id` is operator configuration, often a local
                # research/ checkpoint with no commit to pin; a Hub id can carry its
                # own revision.
                tokenizer = AutoTokenizer.from_pretrained(model_id)  # nosec B615
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)  # nosec B615
                model.eval()
                max_length = SEQ2SEQ_MAX_LENGTH[granularity]
                _register(
                    model_key,
                    Seq2SeqSimplifier(
                        model_id,
                        tokenizer,
                        model,
                        granularity=granularity,
                        max_length=max_length,
                        device=serving_device,
                    ),
                )
            elif method == METHOD_PROMPTED_LLM:
                client = _make_ollama_client()
                try:
                    await _probe_ollama(client, model_id)
                except Exception:
                    await client.aclose()
                    raise
                _register(
                    model_key, PromptedLLMSimplifier(model_id, client, granularity)
                )
            else:  # pragma: no cover - guards against a typo in MODEL_ENV_CONFIG
                raise ValueError(
                    f"Unknown simplifier method '{method}' for '{model_key}'"
                )
            logging.info(
                f"Successfully loaded '{model_key}' simplifier model: {model_id}"
            )
        except Exception as exc:
            if method == METHOD_PROMPTED_LLM:
                # usually just Ollama not running (a setup step): one line, no stack
                # trace; the extension shows this state anyway
                logging.warning(
                    f"'{model_key}' simplifier unavailable ({model_id}): {exc}. "
                    f'Start Ollama (`ollama serve`) and pull the model to enable it.'
                )
            else:
                logging.exception(
                    f"Failed to load '{model_key}' simplifier model ({model_id})"
                )

    try:
        yield
    finally:
        for task in worker_tasks.values():
            task.cancel()
        for task in worker_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        for simplifier in simplifiers.values():
            try:
                await simplifier.aclose()
            except Exception:
                logging.exception('Error closing simplifier')


app = FastAPI(lifespan=lifespan)


def extract_and_clean(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Extract links while preserving meaningful whitespace in the payload.

    - If the string contains HTML anchor tags we use BeautifulSoup to pull out
      the hrefs and anchor text, removing the tags from the payload.
    - We intentionally keep newline/tab/paragraph separators intact so sentence-
      granularity simplification can preserve the structure of the original prose.
    """
    links: List[Dict[str, str]] = []
    clean = text
    if '<a' in text:
        soup = BeautifulSoup(text, 'html.parser')
        for a in soup.find_all('a', href=True):
            links.append({'text': a.get_text(), 'url': a['href']})
        clean = soup.get_text()
    return clean.strip(), links


# Allow cross‑origin requests (needed for the extension to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # in production narrow this to your extension/origin
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['*'],
)


class SimplifyRequest(BaseModel):
    text: str = Field(
        ...,
        json_schema_extra={
            'example': (
                'Despite the multiplicity of antecedent conditions and the intervening '
                'variables which collectively obfuscate causal inference, the overarching '
                'regulatory framework mandates uniform compliance across heterogeneous entities.'
            ),
        },
    )
    model: str = Field(
        'online',
        description=(
            "Which loaded simplifier model to use, e.g. 'online', 'finetuned', "
            "'llm_7b'. /health lists the keys actually loaded."
        ),
    )
    audience: Audience = Field(
        DEFAULT_AUDIENCE,
        description=(
            'Who the simplification should target. Only honoured by models whose '
            "method is 'prompted_llm' (/health lists them as `audience_models`); "
            'ignored by the seq2seq checkpoints.'
        ),
    )
    page: Optional[str] = Field(
        None,
        description=(
            'Opaque id for the page load this request belongs to (the extension sends '
            'its per-page session id). It changes nothing about the simplification and '
            'is not part of the cache key -- it only records which page is holding each '
            "cache entry, so `POST /cache/pages/delete` can drop a page's cached work "
            "when that page leaves the extension's History window. Omit it and the "
            'entries this request creates belong to no page and are never dropped that '
            'way.'
        ),
    )


class LinkInfo(BaseModel):
    text: str
    url: str


class SentenceSplit(BaseModel):
    count: int = Field(..., description='How many sentences the input was split into.')
    parts: List[str] = Field(
        ...,
        description=(
            'The per-sentence outputs, in order, whose concatenation (with the '
            'original whitespace between them) is `simplified`.'
        ),
    )


class SimplifyResponse(BaseModel):
    simplified: str = Field(
        ...,
        json_schema_extra={
            'example': (
                'Although many factors make cause and effect unclear, the rules still require '
                'all different organizations to follow the same standards.'
            ),
        },
    )
    links: List[LinkInfo] = Field(
        default_factory=list,
        description='List of extracted links (anchor text + URL) from the original input.',
    )
    cached: bool = Field(
        False,
        description='Whether the result was returned from the cache instead of running the model.',
    )
    audience: str = Field(
        DEFAULT_AUDIENCE.value,
        description=(
            "The audience actually used. For models that don't support audiences this "
            'is always the default, regardless of what was requested.'
        ),
    )
    fallback_reason: Optional[str] = Field(
        None,
        description=(
            "Set when the model's own output was rejected and the original text "
            "returned unchanged: 'hallucination' (seq2seq sentence guard), 'empty' "
            "(either method: the model returned nothing), 'prompt_echo' / 'refusal' / "
            "'degenerate_repetition' / 'no_words' / 'too_long' / 'too_short' / "
            "'unrelated_short_output' / 'timeout' / 'error' (prompted-LLM guard), "
            "'corpus_artifact' (either method: the "
            'whole output was training-corpus boilerplate, e.g. the Simple English '
            "Wikipedia heading 'Other websites'), or 'no_meaningful_change' (either "
            'method: the output differed from the input only by punctuation, or '
            'only by a number it dropped -- see change_guard).'
        ),
    )
    model_result: str = Field(
        '',
        description=(
            'What the model itself produced, kept even when a guard rejected it and '
            '`simplified` therefore holds the original text instead. Equal to '
            '`simplified` whenever `fallback_reason` is null. Empty only where there '
            'was no generation to record at all (a timeout or a transport error). '
            "Together with the request's `text` this makes the chain explicit: input "
            '-> model result -> output, where the output is what the caller should '
            'actually display.'
        ),
    )
    sentence_split: Optional[SentenceSplit] = Field(
        None,
        description=(
            'Present only where the input was actually split: a sentence_by_sentence '
            'model handed text containing more than one sentence. `parts` are the '
            'per-sentence outputs, in order, that were rejoined into `simplified` -- '
            'so a caller can see which sentence a fallback belongs to instead of '
            'having to re-segment the joined string itself. Null for a single '
            'sentence and for every whole_sections request, neither of which '
            'was split.'
        ),
    )


async def _simplify_text_unit(
    model_key: str, audience: Audience, text: str, page: Optional[str] = None
) -> Tuple[str, Optional[str], bool, str]:
    """Runs one unit of text (a whole document, or a single sentence -- the
    caller decides which, based on the model's granularity) through the
    cache-or-enqueue-and-await path. Returns (simplified, fallback_reason,
    was_cached, model_result).

    Links aren't threaded through here: `simplify_text` always returns the
    links it freshly extracted from the *whole* request body, not whatever a
    sub-unit's cache entry happens to hold, so there's nothing meaningful to
    pass through the queue for this -- an empty list is stored instead.

    `page` is bookkeeping only and never reaches the model or the cache key: it
    records which page is holding this unit's entry, so the entry can be dropped
    when that page ages out of the extension's History window.
    """
    cache_key = (model_key, audience.value, text)

    async with cache_lock:  # type: ignore
        if cache_key in cache:
            simplified, _links, reason, model_result = cache[cache_key]
            _attribute_cache_entry(cache_key, page)
            return simplified, reason, True, model_result

    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    try:
        await batch_queues[model_key].put((text, [], audience, future))
    except Exception as exc:
        logging.exception('Failed to enqueue text for batching')
        raise HTTPException(status_code=500, detail='Internal queue error') from exc

    try:
        simplified, reason, model_result = await future
    except Exception as exc:
        # the batch worker's failure surfaces only here; chained so the cause isn't lost
        logging.exception('Text simplification failed')
        raise HTTPException(
            status_code=500, detail='Text simplification failed'
        ) from exc
    # Attributed here rather than carried through the queue, keeping the worker's item
    # shape. The entry may already have been LRU-evicted under load; don't index a
    # missing entry.
    async with cache_lock:  # type: ignore
        if cache_key in cache:
            _attribute_cache_entry(cache_key, page)
    return simplified, reason, False, model_result


@app.post(
    '/simplify',
    response_model=SimplifyResponse,
    summary='Simplify Text (seq2seq or prompted open LLM)',
    description="""
Perform text simplification with one of several independently-configured models, picked
per request via the ``model`` field. ``/health`` lists the keys actually loaded:

- ``online`` / ``finetuned`` / ``document`` -- BART-style seq2seq checkpoints
  (``SIMPLIFIER_MODEL_ONLINE`` defaults to ``eilamc14/bart-large-text-simplification``,
  an off-the-shelf third-party fine-tune; ``SIMPLIFIER_MODEL_FINETUNED`` and
  ``SIMPLIFIER_MODEL_DOCUMENT`` default to the checkpoints fine-tuned for this project,
  on WikiLarge sentence pairs and D-Wikipedia articles respectively).
- ``llm_7b`` / ``llm_3b`` / ``llm_doc_7b`` / ``llm_doc_3b`` -- open-weight
  instruction-tuned models served by a local Ollama sidecar and steered with the BLESS
  Prompt 2 template at sentence or document scope. These also honour the ``audience``
  field.

Each model declares a ``method`` (``"fine_tuned_seq2seq"`` or ``"prompted_llm"``) and a
``granularity`` (``"sentence_by_sentence"`` or ``"whole_sections"``), both reported per
key via ``/health.methods`` and ``/health.granularities`` and both defined in
``backend/vocabulary.py``, which is the only place either is spelled out. For a
``"sentence_by_sentence"`` model, if ``text`` contains more than one sentence it's split
(via ``pysbd``), each sentence is simplified independently, and the results are rejoined
with spaces -- feeding a multi-sentence block through in one piece is what produced
context-free, fractured output in practice. A ``"whole_sections"`` model receives ``text``
unsplit instead, subject to its own length ceiling (``/health.document_max_tokens``).

To improve throughput the backend caches previous results (per model, audience, *and*
sentence/document unit) and groups incoming seq2seq requests into small batches before
forwarding them to the model. Cache hits are returned immediately.

If the model's output fails its quality guard for a given unit, that unit's original
text is returned unchanged in its place and ``fallback_reason`` is set -- for
multi-sentence input, this is the first sentence (in order) that needed a fallback, not
a full per-sentence breakdown. The last of those guards applies to every method: an
output that only moved punctuation around (without splitting a run-on, which does make
it easier to read), or that only lost a number while leaving every word in place, is
not a simplification and is replaced by the original with
``fallback_reason: "no_meaningful_change"``. Word-level edits are never rejected here
-- one dropped word can already be a real simplification. Applying to every method
too: an output that is nothing but training-corpus boilerplate ("Other websites", the
Simple English Wikipedia heading for an external-links section) is a fabrication for
any input, at any length, and is replaced by the original with
``fallback_reason: "corpus_artifact"``.

Whatever the model produced is reported either way as ``model_result``, so a response
describes the whole chain rather than only its ends: **input → model_result → output**,
where ``simplified`` is the output a caller should display. Where a guard fired the two
differ and ``fallback_reason`` says why; where none did they are the same string. For a
sentence-granularity request over more than one sentence, ``sentence_split`` carries the
per-sentence outputs that were rejoined into ``simplified``, so a caller can attribute a
fallback to the sentence it came from without re-segmenting the joined text itself.

Returns a 503 error if no model has finished loading yet, or 400 if the requested
`model` isn't one of the ones currently loaded.
""",
)
async def simplify_text(data: SimplifyRequest):
    raw = data.text

    if not raw:
        raise HTTPException(status_code=400, detail="Missing 'text' field")

    if not simplifiers:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    model_key = data.model
    if model_key not in simplifiers:
        available = ', '.join(sorted(simplifiers)) or 'none'
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_key}' is not available. Loaded models: {available}",
        )

    # Models that ignore the audience get the default, so they share cache entries
    # across requested audiences.
    simplifier = simplifiers[model_key]
    audience = data.audience if simplifier.supports_audience else DEFAULT_AUDIENCE

    clean_text, links = extract_and_clean(raw)

    sentence_split = None
    if simplifier.granularity == GRANULARITY_WHOLE_SECTIONS:
        simplified, reason, was_cached, model_result = await _simplify_text_unit(
            model_key, audience, clean_text, data.page
        )
    else:
        # Each sentence is simplified independently, then the original whitespace is
        # reattached so blank lines and tabs survive.
        sentence_units = split_sentences_with_whitespace(clean_text)
        sentence_results = await asyncio.gather(
            *(
                _simplify_text_unit(model_key, audience, sentence, data.page)
                for sentence, _ in sentence_units
            )
        )
        simplified = ''.join(
            simplified_sentence + trailing_ws
            for (_, trailing_ws), (simplified_sentence, _, _, _) in zip(
                sentence_units, sentence_results, strict=True
            )
        )
        reason = next((r for _, r, _, _ in sentence_results if r is not None), None)
        was_cached = all(c for _, _, c, _ in sentence_results)
        # joined like `simplified`, so the two are directly comparable
        model_result = ''.join(
            unit_model_result + trailing_ws
            for (_, trailing_ws), (_, _, _, unit_model_result) in zip(
                sentence_units, sentence_results, strict=True
            )
        )
        # only when actually split; a single sentence is not a one-part split
        if len(sentence_results) > 1:
            sentence_split = {
                'count': len(sentence_results),
                'parts': [part for part, _, _, _ in sentence_results],
            }

    return {
        'simplified': simplified,
        'links': links,
        'cached': was_cached,
        'audience': audience.value,
        'fallback_reason': reason,
        'model_result': model_result,
        'sentence_split': sentence_split,
    }


@app.get('/health')
async def health():
    """Reports whether the backend is up and which model(s) have finished
    loading, so clients can distinguish "unreachable" from "reachable but
    still loading" without needing to run a real simplification, and can tell
    whether a specific model (e.g. 'finetuned') is actually available to select.

    Also serves the audience list, so the extension builds its dropdown from one
    source of truth (``prompting.Audience``) instead of a hard-coded copy that can
    drift out of sync with the backend."""
    loaded = sorted(simplifiers.keys())
    ids = model_ids()
    return {
        'status': 'ok',
        'model_loaded': bool(simplifiers),
        'models_loaded': loaded,
        'model_names': {k: ids[k] for k in loaded},
        # model keys that actually honour the `audience` request field
        'audience_models': [k for k in loaded if simplifiers[k].supports_audience],
        # model key -> one of vocabulary.METHODS, so the extension doesn't keep its own
        # copy of this mapping (which goes stale when keys are added)
        'methods': {k: simplifiers[k].method for k in loaded},
        # model key -> one of vocabulary.GRANULARITIES. Needed before building any
        # request: the extension collects page text per element vs. per
        # heading-delimited section.
        'granularities': {k: simplifiers[k].granularity for k in loaded},
        # Token ceiling for document requests, so the extension chunks against the real
        # limit. Per model: a seq2seq checkpoint's trained max_length (512, input past
        # it truncated) vs. an LLM's far larger context window shared with its output;
        # a global 512 would chunk the LLM as tightly as the checkpoint.
        'document_max_tokens': {
            k: _document_max_tokens(simplifiers[k]) for k in loaded
        },
        # "cpu"/"mps"/"cuda", seq2seq only (Ollama picks its own). Reported because it
        # changes the generated bytes, so anything citing a served output needs it.
        'seq2seq_device': serving_device,
        'audiences': [
            {'value': a.value, 'label': AUDIENCE_LABELS[a]} for a in Audience
        ],
        'default_audience': DEFAULT_AUDIENCE.value,
    }


@app.post(
    '/cache/clear',
    summary='Clear the simplification cache',
    description="""
Empties the in-memory result cache described under ``/simplify``, so the next request
for text that had been simplified before runs the model again instead of being served
from the cache.

This is what the extension's "Clear cache" button and its toolbar-icon menu entry of the
same name call. It clears the *cache*, not the History log — the log lives in the
extension's own storage, and discarding a cached generation does not unhappen the run
that produced it. The reverse direction does hold: deleting a run from that log drops the
cached simplifications it produced, via ``/cache/pages/delete`` below.

Per-process and in-memory, like the cache itself: nothing is persisted, so restarting
the backend has the same effect. Loaded models, batch queues and health are untouched.
""",
)
async def clear_cache():
    global cache
    if cache_lock is None:
        # before lifespan ran: no lock, nothing cached; same shape, not an error
        return {'cleared': 0}
    async with cache_lock:
        cleared = len(cache)
        cache.clear()
        # keep the page index in sync with the cache
        cache_pages.clear()
        cache_key_pages.clear()
    logging.info('Simplification cache cleared (%d entries)', cleared)
    return {'cleared': cleared}


class CachePagesRequest(BaseModel):
    pages: List[str] = Field(
        default_factory=list,
        description=(
            "The page ids to forget -- the same values sent as `/simplify`'s `page` "
            'field. Ids that hold nothing are ignored rather than being an error.'
        ),
    )


@app.post(
    '/cache/pages/delete',
    summary='Drop the cached simplifications belonging to specific pages',
    description="""
Removes every cache entry held by the given pages and by nothing else, leaving the rest
of the cache alone. The counterpart to ``/simplify``'s ``page`` field, which is what
records the ownership this reads.

An entry shared by several pages survives until its last holder is dropped: two pages
quoting the same sentence share one generation, and taking it away because one of them
was deleted would make the other run the model again. Entries created by requests that
named no page (the extension's toggle preflight, a History comparison run) belong to no
page and are never removed this way.

The extension calls this for three things, all of them the same rule -- a cached
simplification should not outlive the record of the run it came from:

- **automatically**, when a page falls out of the last-20 window its History log keeps,
  so the cache holds the same 20 pages the log does rather than growing to ``CACHE_MAX``
  entries of pages nobody can look up any more;
- when one page's History entry is deleted;
- when the whole History log is deleted, which sends every logged page id.

Per-process and in-memory, like the cache itself: after a backend restart there is
nothing to drop, which is reported honestly as ``{"dropped": 0, "pages": 0}`` rather
than as an error.
""",
)
async def delete_cache_pages(data: CachePagesRequest):
    if cache_lock is None:
        return {'dropped': 0, 'pages': 0}
    async with cache_lock:
        dropped, pages = _drop_cache_pages(data.pages)
    if pages:
        logging.info(
            'Dropped %d cached simplification(s) belonging to %d page(s)',
            dropped,
            pages,
        )
    return {'dropped': dropped, 'pages': pages}


@app.get('/')
async def root():
    return RedirectResponse(url='/docs')
