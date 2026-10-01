# ba-simplify

Automated simplification of everyday English on web pages, delivered as a Chrome extension over a
local FastAPI service. The simplified text appears in the page itself: only the text changes, and
the layout stays as its author wrote it.

Two `bart-base` checkpoints were fine-tuned for the project, one on WikiLarge at sentence level and
one on D-Wikipedia at document level, and both are compared against an off-the-shelf fine-tune and
against prompted open-weight LLMs. Everything runs on the user's own machine.

## What it does

- Simplifies a complete page in place, preserving markup: links, images, citation markers and
  opted-out subtrees stay where the author put them, and the page can be reverted.
- Serves **three simplification methods** behind one API — an off-the-shelf comparison model, this
  project's fine-tuned checkpoints, and prompted open-weight LLMs via a local Ollama sidecar — at
  **two granularities**, singular sentence and whole document.
- Guards its own output: hallucination, refusal, echo and no-meaningful-change checks, each
  reporting why it fired, with the original served unchanged on failure.
- Logs every run so that input, raw model output and served text stay visible side by side.

## Results

|                               | Sentence level (ASSET) | Document level (D-Wikipedia) |
| ----------------------------- | ---------------------- | ---------------------------- |
| zero-shot `bart-base`         | 21.39 SARI             | 14.34 D-SARI                 |
| **this project's checkpoint** | **37.80 SARI**         | **35.44 D-SARI**             |
| off-the-shelf `bart-large`    | 38.17 SARI             | —                            |
| prompted 7B / 3B              | 45.95 / 47.03 SARI     | 22.37 D-SARI (7B)            |

Fine-tuning beats its own starting checkpoint decisively (+16.42 SARI, 95% CI [15.38, 17.50]) but
only matches a model three times its size, and loses to prompted models on the measures closest to
human judgement. D-SARI and LENS rank the document systems in opposite orders; that disagreement is
unresolved and is the project's most interesting result. Every number, with its provenance, is in
[`research/results/RESULTS.md`](research/results/RESULTS.md).

## Requirements

- Python 3.10+
- Node 18+ (for the extension's jsdom test harness)
- Google Chrome
- Optional: [Ollama](https://ollama.com), for the prompted-LLM models

## Quick start

```sh
# 1. Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run_dev.sh                  # auto-stops after 10 min idle; also serves demo/ and Ollama
```

`run_dev.sh` starts Ollama before uvicorn, since the backend probes it once at startup, and stops
it again on exit. `NO_LLM=1` skips the sidecar. To run the server directly instead:
`uvicorn main:app --reload --port 8000`.

On a machine with no checkpoints, no GPU and no sidecar, `venv/bin/python dev_stub_server.py`
serves the same endpoints with stand-in models — the splitting, batching, caching, link extraction
and all three guards are the real code, only the tensor call is faked.

```sh
# 2. Extension
# chrome://extensions -> Developer mode -> Load unpacked -> select extension/
```

Click the toolbar icon to simplify the current page. [`demo/`](demo/) holds static pages for
checking specific behaviour (DOM targeting, exclusions, dynamic content); see
[`demo/README.md`](demo/README.md).

## Tests

```sh
cd backend && pip install -r tests/requirements-dev.txt && pytest
cd extension/tests && npm install && npm test
```

## Training and evaluation

The research dependencies install on top of the same virtualenv, since
`research/requirements.txt` installs `backend/requirements.txt` first:

```sh
cd research
../backend/venv/bin/pip install -r requirements.txt
../backend/venv/bin/python evaluate_sentence.py --limit 10   # smoke run
```

Run those scripts from `research/` and not from the repository root, since they resolve `scratch/`
and `results/` relative to the working directory. See [`research/README.md`](research/README.md),
and [`research/remote/README.md`](research/remote/README.md) for running the stages on a real GPU.

## Repository layout

```text
backend/                # the service: everything that runs to serve a request
	main.py             # FastAPI app: /simplify, /health, /cache/*, model registry, caching
	vocabulary.py       # the method/granularity values, defined once
	prompting.py        # BLESS Prompt 2 templates for the prompted-LLM models
	change_guard.py     # method-agnostic "is this actually a simplification?" guard
	document_text.py    # D-Wikipedia text conventions, normalize + denormalize
	run_dev.sh          # dev server with auto-idle-stop; also serves demo/ and Ollama
	dev_stub_server.py  # the same app with stand-in models: no torch, no weights, no Ollama
	tests/              # pytest suite, with its own requirements-dev.txt
research/               # the thesis work: nothing here runs in production
	prepare_data.py     # loads/cleans WikiLarge into research/data/
	fine_tune.py        # CLI training (simpler equivalent of the notebook)
	evaluate_sentence.py        # ASSET benchmark, all sentence-level methods
	evaluate_document.py        # D-Wikipedia benchmark, D-SARI/LENS/BERTScore
	evaluate_llm.py             # prompted-LLM generations, local or remote endpoint
	review_document_outputs.py  # reads those generations back: error categories
	profile_deployment.mjs      # extension+backend latency, per request regime
	audit_sites.mjs             # RQ2 site-sample audit: what collection does to real pages
	notebooks/          # tiered training pipeline; Colab document evaluation
	remote/             # running the stages on a real GPU (CL machines, HHU HPC)
	model_cards/        # published alongside the checkpoints on the HF Hub
	results/            # RESULTS.md — every number, with provenance
extension/
	background.js       # service worker: proxies API calls, health checks, menu
	content.js          # DOM targeting, filtering, write-back, on-page notice
	home.html           # homepage + live status dashboard
	history.html        # last-20-pages simplification history
	reports.html        # runs kept out of the log: progress, evaluation, JSON export
	instructions.html   # setup + how it works; also opened on install/update
	error.html          # troubleshooting page for health-check and click failures
	shared/             # every page's .js/.css, plus the toolbar, pickers and run lock
	tests/              # jsdom harness: npm test
paper/                  # the written thesis (LaTeX)
	paper.tex           # main file: preamble + \input of contents/
	paper.pdf           # the final PDF
	contents/           # front matter + one numbered file per chapter
	assets/             # generated figure PDFs
	scripts/            # make_figures.py
demo/                   # static HTML pages for manually verifying extension behaviour
```

## Documentation

| Where                                                        | What                                                                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| [`paper/paper.pdf`](paper/paper.pdf)                         | The thesis: why each part is built this way, the design decisions, the bugs behind them, and the limitations |
| [`research/results/RESULTS.md`](research/results/RESULTS.md) | Every measured number, with its provenance                                                                   |
| [`research/README.md`](research/README.md)                   | The research pipeline, end to end                                                                            |
| [`research/data/README.md`](research/data/README.md)         | Which corpus row count belongs to which run                                                                  |
| [`demo/README.md`](demo/README.md)                           | What each demo page is for                                                                                   |

The research is complete and frozen; the current phase is writing. Detailed notes on the state of
each component, and the roadmap, are kept as working documents outside the repository tree.

## Licence

© 2026 Yunus Oscar Renz. Licensed under [CC BY 4.0](LICENSE): you may share and adapt this work for any purpose,
provided you give appropriate credit, link to the licence, and indicate if changes were made.
Please cite the thesis:

> Renz, Yunus Oscar (2026). *From Benchmark to Browser: A Comparative Evaluation of Automatic Text Simplification
> Approaches for DOM-Preserving Web Deployment*. Bachelor's thesis, Heinrich Heine University Düsseldorf.

The fine-tuned checkpoints on the Hugging Face Hub are a separate release under CC BY-SA 4.0, inherited from their
Wikipedia-derived training data; see [`research/model_cards/`](research/model_cards/).
