# research/

The thesis work: dataset preparation, fine-tuning, and evaluation. **Nothing in here
runs in production** — the service is `../backend/`.

That line is the reason this directory exists. `backend/` was previously holding a
deployed FastAPI service and a research pipeline in the same flat namespace, which made
it unclear which files a request actually touches, and put six evaluation-only
dependencies into the list a container would install. Dataset preparation used to live
in a separate top-level `training/`; it is here now, since it is the same body of work.

|                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fine_tune.py`                                         | CLI training. A simpler equivalent of the notebook, with a single `--prove_loop` flag instead of the notebook's three scope tiers.                                                                                                                                                                                                                                                                                                                  |
| `evaluate_sentence.py`                                 | **ASSET** benchmark (359 sentences, 10 refs). Scores every sentence-level method — off-the-shelf BART, this project's fine-tune, and the prompted Qwen2.5 models — through one shared loader and metric path.                                                                                                                                                                                                                                       |
| `evaluate_llm.py`                                      | Generates prompted-LLM outputs for ASSET, locally via Ollama **or** remotely against any OpenAI-compatible endpoint (Together/Groq/OpenRouter/vLLM/TGI, incl. a free Colab GPU). Computes no metrics — writes generations in `evaluate_sentence.py`'s cache format so one script owns the scoring. Resumable and rate-limit tolerant. Remote is acceptable here and _not_ in the extension because ASSET is public data; see the module docstring.  |
| `evaluate_document.py`                                 | **D-Wikipedia** test split with D-SARI, LENS and BERTScore. Step 4B: the document-appropriate counterpart, since ASSET cannot measure document-level simplification. Checkpoints as it goes — rerunning the same command resumes rather than restarting. `--llm <ollama-tag>` adds a prompted open LLM as a third system at document granularity.                                                                                                   |
| `review_document_outputs.py`                           | Qualitative counterpart to `evaluate_document.py`: reads its cached generations back and reports what the metrics cannot — compression against the reference by document length, and error categories (substituted facts, over-deletion, declining to edit) with worked examples. Costs seconds and loads no model. Re-derives the sample and refuses to run unless it matches the generation run's `sources_sha`.                                  |
| `profile_deployment.mjs`                               | Latency profile of the deployed extension+backend on a real Wikipedia article, per request regime — the sentence path's many small requests versus the document path's few large ones. Loads the extension's real `content.js` under jsdom, so the collection, chunking and 8-slot concurrency queue being measured are the ones that ship — as do the RQ2 audit scripts below. Needs `npm install` and a running backend (**without** `--reload`). |
| `audit_sites.mjs`                                      | RQ2's site-sample audit: what the extension's content selection does to twelve real pages across six categories, at both cuts — how much of a page it touches, how much it leaves alone, and whether the page survives the visit. Drives the real `content.js` under jsdom against a running backend.                                                                                                                                               |
| `summarise_site_audit.mjs`                             | Turns one `audit_sites.mjs` artifact into the tables `results/RESULTS.md` quotes, so no figure in the write-up is hand-copied out of a 24-row JSON file.                                                                                                                                                                                                                                                                                            |
| `sample_for_labelling.mjs`                             | Draws the hand-labelling sample the audit's second half needs (~50 skipped and ~50 unchanged items): the rates say how often the filters act, only a reader can say whether each call was right.                                                                                                                                                                                                                                                    |
| `inspect_emptied_labels.mjs`                           | Turns the audit's `anchor_labels_emptied` count back into markup a person can read — what was sent, what came back, and what the write-back did to the anchor beside it.                                                                                                                                                                                                                                                                            |
| `inspect_revert_diff.mjs`                              | Prints where a reverted page still differs from the page that was served. The audit's text and attribute checks come back clean everywhere; the raw-HTML check does not, and this is what says why.                                                                                                                                                                                                                                                 |
| `prepare_data.py`                                      | Loads and cleans WikiLarge into `data/`. Was `training/prepare_data.py`.                                                                                                                                                                                                                                                                                                                                                                            |
| `select_fewshot.py`                                    | Regenerates the few-shot demonstration pools baked into `../backend/prompting.py`.                                                                                                                                                                                                                                                                                                                                                                  |
| `notebooks/train_sentence_and_document_pipeline.ipynb` | The tiered (`prove_loop`/`reduced`/`full`) training + evaluation pipeline. Both eval scripts were extracted from its Step 4 / 4B cells.                                                                                                                                                                                                                                                                                                             |
| `notebooks/colab_dwikipedia_eval.ipynb`                | Runs `evaluate_document.py` on a Colab GPU (~1.0 s/document measured on a T4, vs ~8.4 s on Apple Silicon). Mounts Drive and writes every artifact there as it is produced, because Colab kills the runtime on disconnect.                                                                                                                                                                                                                           |
| `model_cards/`                                         | The cards published alongside the two checkpoints on the HF Hub. Edit here, then re-upload — the Hub copy is downstream of this one.                                                                                                                                                                                                                                                                                                                |
| `remote/`                                              | The GPU harness: venv bootstrap, per-host profiles, a correctness preflight that gates a long run before it starts, and a resumable stage pipeline. See `remote/README.md`.                                                                                                                                                                                                                                                                         |
| `results/`                                             | `RESULTS.md` (every number in one table) plus provenance-stamped run JSON.                                                                                                                                                                                                                                                                                                                                                                          |
| `data/`                                                | Prepared WikiLarge splits. The `.jsonl` files are gitignored; regenerate with `prepare_data.py`.                                                                                                                                                                                                                                                                                                                                                    |
| `scratch/`                                             | Gitignored. ~11 GB of checkpoints, optimizer state and raw corpora. Regenerable.                                                                                                                                                                                                                                                                                                                                                                    |

## Running things

Run the scripts **from this directory** — both resolve `scratch/` and `results/`
relative to the working directory:

```sh
cd research
../backend/venv/bin/python evaluate_sentence.py --limit 10     # smoke run
../backend/venv/bin/python evaluate_document.py --limit 20     # smoke run
```

The prompted-LLM conditions are slow enough locally (~2 s/sentence idle, ~10 s under
load) that a full ASSET pass runs into hours. `evaluate_llm.py` exists to move that one part
off this machine when you want the numbers sooner — generation only, then score with the
usual harness:

```sh
# local: needs `ollama serve` and the tag pulled
../backend/venv/bin/python evaluate_llm.py --condition llm_7b --model qwen2.5:7b-instruct-q4_K_M

# remote: same prompt, same decoding, ~1-2 orders of magnitude faster
export LLM_API_KEY=...
../backend/venv/bin/python evaluate_llm.py --condition llm_7b_remote --backend openai \
    --base-url https://api.together.xyz/v1 --model Qwen/Qwen2.5-7B-Instruct-Turbo

# scoring is always the same script
../backend/venv/bin/python evaluate_sentence.py --conditions online,local,llm_7b
```

Interrupt either eval script and rerun the same command — both resume from what they
already produced.

One venv covers both trees. `requirements.txt` here starts with
`-r ../backend/requirements.txt` and adds the evaluation and dataset libraries on top, so
the research environment is a superset of the serving one:

```sh
../backend/venv/bin/pip install -r requirements.txt
```

The service's own `requirements.txt` deliberately does _not_ list `datasets`, `easse`,
`bert-score` and friends — it should stay installable as a deployable unit.

Several scripts here import from `../backend` via a `sys.path` insert, deliberately, and
always for the same reason: a number is only comparable to what the service does if it was
produced by the service's own code rather than by a copy of it kept in sync by hand.
`evaluate_sentence.py`, `evaluate_llm.py` and `evaluate_document.py` import `prompting`
(and `evaluate_document.py` also `vocabulary`) so the scored generations use byte-identical
prompts; `remote/denormalize_corpus.py` imports `document_text` so the corpus is
de-normalised exactly as a served answer is; `remote/preflight.py` imports both to check
them before a long run. Every one of those dependencies points this way on purpose — the
service must never import from here.

`lens-metric` is declared but not installed locally — it pins its own
torch/transformers and would conflict with the versions the service needs. LENS is
therefore only computed in the Colab notebook, and both eval scripts record
`"LENS": null` and carry on when it is unavailable.

## The checkpoints

The fine-tuned checkpoints live on the Hugging Face Hub, not in git — a 532 MB
`model.safetensors` is well past GitHub's hard 100 MB per-file limit:

- [`yunvs/bart-base-wikilarge-simplification`](https://huggingface.co/yunvs/bart-base-wikilarge-simplification) — sentence-level
- [`yunvs/bart-base-dwikipedia-simplification-full`](https://huggingface.co/yunvs/bart-base-dwikipedia-simplification-full) — document-level, full scope (131,739 docs / 5 epochs). **The one to use**, and the backend's default.
- [`yunvs/bart-base-dwikipedia-simplification`](https://huggingface.co/yunvs/bart-base-dwikipedia-simplification) — document-level, reduced scope. **Superseded**, kept published because RESULTS.md rows D1/D2/D4/D2′ cite it.

`from_pretrained` takes a Hub id and a local path interchangeably, so point `--model` at
`scratch/simplification_results/best_checkpoint` to score a local training run instead
of the published copy.
