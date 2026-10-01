import sys
from pathlib import Path

import pytest

# make `backend/main.py` importable as `main` regardless of how pytest is
# invoked (bare `pytest`, `python -m pytest`, from a different cwd, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402  (must follow the sys.path insert above)

#: The one model the suite treats as its baseline. Most tests want exactly one seq2seq
#: model loaded and fake it at the `transformers` level, so this stays enabled; every
#: other key is off unless a test asks for it.
BASELINE_MODEL_KEY = 'online'


@pytest.fixture(autouse=True)
def disable_all_models_by_default(monkeypatch):
    """Force every configured model off except the baseline, then let tests opt in.

    Derived from `main.MODEL_ENV_CONFIG` rather than a hardcoded list, so a new model
    key with a truthy default can't silently change what the suite loads. This keeps
    `/health` assertions about loaded models exact, and keeps the Ollama-backed keys
    from probing a real sidecar at startup (which would make the suite depend on
    whether Ollama is running).

    Tests opt back in with their own `monkeypatch.setenv`, which runs after this
    fixture and wins -- including opting the baseline *out*, as the LLM-only tests do.
    """
    for key, (env_name, *_) in main.MODEL_ENV_CONFIG.items():
        if key != BASELINE_MODEL_KEY:
            monkeypatch.setenv(env_name, '')
