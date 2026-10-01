"""Runs the real backend with stand-in "models", so a request can be watched all
the way through without torch, checkpoints, or an Ollama sidecar.

Everything except generation is the production path: pysbd sentence splitting,
per-model batching, the shared cache, `<a href>` extraction, the document
checkpoint's normalize/de-normalize round trip, the hallucination/change/corpus
guards, and the whole input -> model_result -> simplified chain in the response.

What makes the splitting and re-joining legible is that the "model" echoes each
unit it was handed with a marker naming the model key and the unit's index in the
batch -- so every fragment of the answer says which unit produced it. This is the
same fake shape backend/tests/ uses (a tokenizer that remembers its last batch, a
model that "generates" indices into it), so it drives code the suite already
covers rather than a second, parallel implementation of it.

Two keys are registered as stubs: `finetuned` (sentence granularity) and
`document` (whole sections). The `llm_*` keys are untouched -- if an Ollama
sidecar happens to be running they load for real, alongside the stubs.

STUB_MODE picks what the stub "generates":

    marker    (default) the unit, plus "[stub ... unit N]"
    guard     the unit with its numbers deleted -- the one edit change_guard.py
              rejects, so the response shows model_result beside the input it
              fell back to, and names the reason
    truncate  the unit less its last sentence -- a document model deleting
              content, which is the case the extension has to divide an answer
              across units for

Usage:
    venv/bin/python dev_stub_server.py                    # port 8000
    PORT=8099 STUB_MODE=guard venv/bin/python dev_stub_server.py
"""

import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# A non-empty ID registers a key (see main.lifespan()); these only appear in /health.
# Set when unset *or empty*, so an env disabling the real checkpoints with
# `SIMPLIFIER_MODEL_DOCUMENT=` still gets stubs.
for _env_name, _stub_id in (
    ('SIMPLIFIER_MODEL_FINETUNED', 'stub-sentence-model'),
    ('SIMPLIFIER_MODEL_DOCUMENT', 'stub-document-model'),
):
    if not os.environ.get(_env_name):
        os.environ[_env_name] = _stub_id
# one stub per granularity is enough
os.environ.setdefault('SIMPLIFIER_MODEL_ONLINE', '')

STUB_MODE = os.environ.get('STUB_MODE', 'marker')

import transformers  # noqa: E402  (patched below, before lifespan imports from it)


class StubTokenizer:
    """Remembers the batch it was called with, so `decode` can map a token id --
    which is just an index into that batch -- back to a deterministic string."""

    def __init__(self, label):
        self.label = label
        self._last = []

    def __call__(self, texts, **kwargs):
        self._last = texts
        return {'texts': texts}

    def decode(self, token_id, skip_special_tokens=True):
        text = self._last[token_id]
        if STUB_MODE == 'guard':
            return re.sub(r'\s*\b\d[\d,.]*\b', '', text)
        if STUB_MODE == 'truncate':
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            return ' '.join(parts[:-1]) if len(parts) > 1 else text
        return f'{text} [{self.label} unit {token_id}]'


class StubModel:
    def eval(self):
        return self

    def generate(self, texts, max_length=512):
        return list(range(len(texts)))


_STUBS = {}


def _stub_for(model_id):
    if model_id not in _STUBS:
        label = 'stub sentence' if 'sentence' in model_id else 'stub document'
        _STUBS[model_id] = (StubTokenizer(label), StubModel())
    return _STUBS[model_id]


class StubAutoTokenizer:
    @staticmethod
    def from_pretrained(model_id):
        return _stub_for(model_id)[0]


class StubAutoModelForSeq2SeqLM:
    @staticmethod
    def from_pretrained(model_id):
        return _stub_for(model_id)[1]


transformers.AutoTokenizer = StubAutoTokenizer
transformers.AutoModelForSeq2SeqLM = StubAutoModelForSeq2SeqLM

import main  # noqa: E402  (must follow the patches above)
import uvicorn  # noqa: E402

if __name__ == '__main__':
    print(
        f'STUB:    generation is faked (STUB_MODE={STUB_MODE}) -- the request path '
        'around it is the real one.',
        file=sys.stderr,
    )
    uvicorn.run(main.app, host='127.0.0.1', port=int(os.environ.get('PORT', '8000')))
