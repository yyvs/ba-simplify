# HHU GPU session results

Produced by `research/remote/pipeline.py`. Every table below names the host, the driver, the commit and the artifact that produced it.

**Read this against `research/results/RESULTS.md` and fold rows in by hand.** Deciding which number supersedes which is editorial — RESULTS.md's own history includes retiring a row whose BLEU pair turned out not to reproduce, and no automatic merge would have caught that.

Stages completed: asset, dsari, eval_doc, eval_doc_full, filters, lens, overlap, review, smoke, smoke_eval, train_doc, train_document_full, train_document_prove_loop

---

## 1. Sentence level — ASSET test (5 conditions)

Benchmark: facebook/asset test · n=359 · 10 references · decoding beam · few-shot 3 · 3 seeds per sampled condition · significance baseline `base`
Host: commit cbd5ef6 · device cuda · 42 min
Artifact: `results/remote/sentence_eval/summary.json`

| Condition | Model                                    | SARI ↑    | BLEU  | FKGL ↓ | BERTScore | unchanged | mean words | SARI sd (seeds) |
| --------- | ---------------------------------------- | --------- | ----- | ------ | --------- | --------- | ---------- | --------------- |
| `base`    | facebook/bart-base                       | **21.39** | 91.64 | 10.02  | 98.58     | 0.939     | 19.7       | —               |
| `online`  | eilamc14/bart-large-text-simplification  | **38.17** | 88.58 | 7.85   | 97.55     | 0.134     | 16.3       | —               |
| `local`   | yunvs/bart-base-wikilarge-simplification | **37.80** | 88.30 | 8.18   | 97.65     | 0.117     | 16.1       | —               |
| `llm_7b`  | qwen2.5:7b-instruct-q4_K_M               | **45.95** | 75.46 | 7.77   | 97.87     | 0.045     | 19.0       | 0.073           |
| `llm_3b`  | qwen2.5:3b-instruct-q4_K_M               | **47.02** | 68.21 | 8.22   | 97.34     | 0.012     | 19.2       | 0.053           |

**Paired bootstrap over sentence indices** (SARI, vs the significance baseline):

- `online`: ΔSARI 16.79 95% CI [15.77, 17.82], p=0.0 over 1000 resamples
- `local`: ΔSARI 16.42 95% CI [15.38, 17.50], p=0.0 over 1000 resamples
- `llm_7b`: ΔSARI 24.63 95% CI [23.64, 25.70], p=0.0 over 1000 resamples
- `llm_3b`: ΔSARI 25.67 95% CI [24.80, 26.51], p=0.0 over 1000 resamples

> Report the CI, not the p-value, as the substantive claim: 1000 resamples cannot resolve below 1/1000, so a p of 0.0 is `p < 0.001`.

---

## 2. Document level — D-Wikipedia test, FULL split (n=8000)

Benchmark: D-Wikipedia test · n=8000 · 1 reference/document · seed full split (unsampled)
Host: commit f49e35c
Artifact: `results/remote/document_eval_full/step4b_dwikipedia_20260821T114841Z_n8000.json`

| System    | D-SARI ↑  | SARI ↑ | BLEU  | FKGL ↓ | BERTScore | LENS  |
| --------- | --------- | ------ | ----- | ------ | --------- | ----- |
| finetuned | **35.44** | 41.95  | 27.21 | 7.86   | 90.61     | 45.61 |
| baseline  | **14.34** | 21.60  | 16.66 | 9.69   | 88.80     | 33.42 |

LENS artifact: `results/remote/document_eval_full/lens_n8000_seed42.json` — descriptive corpus means, not significance-tested. **This is the project's first LENS column.**

- finetuned vs baseline (d_sari, paired bootstrap over documents): p=0.0 over 1000 resamples

Qualitative review: `results/remote/document_eval_full/qualitative_review_full.json` — rerun against these generations, so §6.3's taxonomy describes the checkpoint this table reports.

---

## 2b. Document level — three-way comparison incl. prompted LLM (n=2000)

Benchmark: D-Wikipedia test · n=2000 · 1 reference/document · seed 42
Host: commit cbd5ef6
Artifact: `results/remote/document_eval/step4b_dwikipedia_20260821T224703Z_n2000.json`

| System                                | D-SARI ↑          | SARI ↑        | BLEU          | FKGL ↓       | BERTScore     | LENS              |
| ------------------------------------- | ----------------- | ------------- | ------------- | ------------ | ------------- | ----------------- |
| finetuned                             | **34.93**         | 41.93         | 27.64         | 7.94         | 90.61         | 45.43             |
| baseline                              | **14.42**         | 22.28         | 17.58         | 9.66         | 88.93         | 33.72             |
| llm (seed 1)                          | **22.37**         | 39.60         | 13.65         | 5.89         | 88.64         | 69.27             |
| **llm (mean of 3 seeds, 2026-08-24)** | **22.39** ± 0.039 | 39.67 ± 0.062 | 13.65 ± 0.039 | 5.89 ± 0.008 | 88.66 ± 0.027 | **69.22** ± 0.071 |

LENS artifact: `results/remote/document_eval/lens_n2000_seed42.json` — descriptive corpus means, not significance-tested. **This is the project's first LENS column.**

Seeds 2 and 3 (2026-08-24): `results/remote/document_eval_llmseed{2,3}/`, one directory per seed because the merged generations cache is not keyed by `--llm-seed` (since 2026-08-24 the script enforces that rather than trusting it — RESULTS.md §4, item 15). The seq2seq rows come back bit-identical on every metric including LENS, so only the sampled row moves. **The metric disagreement survives sampling:** fine-tuned − llm on D-SARI is 12.54 ± 0.039, llm − fine-tuned on LENS is 23.80 ± 0.071.

- finetuned vs baseline (d_sari, paired bootstrap over documents): p=0.0 over 1000 resamples
- llm vs baseline (d_sari): p=0.0 · llm vs finetuned: p=1.0 over 1000 resamples

---

## 2c. Normaliser isolation — the previous checkpoint, re-scored under the fixed normaliser (n=500)

Benchmark: D-Wikipedia test · n=500 · 1 reference/document · seed 42
Host: commit cbd5ef6
Artifact: `results/remote/normaliser_isolation/step4b_dwikipedia_20260821T215233Z_n500.json`

| System    | D-SARI ↑  | SARI ↑ | BLEU  | FKGL ↓ | BERTScore | LENS |
| --------- | --------- | ------ | ----- | ------ | --------- | ---- |
| finetuned | **31.62** | 41.13  | 22.29 | 7.58   | 89.94     | —    |
| baseline  | **14.63** | 21.48  | 18.09 | 9.54   | 89.21     | —    |

> LENS absent. It is the only metric here trained on human judgements, and the only reason it has never been reported is a dependency conflict a second venv solves: `bash remote/setup.sh --lens`, then the `lens` stage.

- finetuned vs baseline (d_sari, paired bootstrap over documents): p=0.0 over 1000 resamples

Qualitative review: `results/remote/normaliser_isolation/qualitative_review_old_ckpt.json` — rerun against these generations, so §6.3's taxonomy describes the checkpoint this table reports.

---

## 2d. Document level, natural-case checkpoint (stretch experiment)

_Not produced this session._

---

## Training runs

| Run                   | Corpus rows | Filter | Epochs | Batch | Precision | Best val loss | Wall clock |
| --------------------- | ----------- | ------ | ------ | ----- | --------- | ------------- | ---------- |
| `document_full`       | 131739      | strict | 5      | 32    | bf16      | 0.3340        | 2h44m      |
| `document_prove_loop` | 200         | strict | 1      | 2     | bf16      | 2.8435        | 8s         |

> `Corpus rows` is what the checkpoint actually trained on — cite this, not `data/stats.json`, which was produced by a different script and possibly a different filter.

- `document_full`: beet · NVIDIA RTX A5000 · torch 2.6.0+cu124 · driver 535.309.01 · commit 041f88d · DIRTY TREE · selected `/home/yunusrenz/simple-website/research/scratch/simplification_results_document/checkpoint-16464` at step 20585 (epoch 5.00) · manifest `results/remote/train_document_full.json`
- `document_prove_loop`: beet · NVIDIA RTX A5000 · torch 2.6.0+cu124 · driver 535.309.01 · commit 041f88d · DIRTY TREE · selected `scratch/smoke_document/checkpoint-100` at step 100 (epoch 1.00) · manifest `results/remote/train_document_prove_loop.json`

---

## Audits and threats to validity

### Train/eval overlap

Artifact: `results/remote/overlap_audit.json` · method: exact after NFKC normalisation, lowercasing, and collapsing non-alphanumeric runs to single spaces

| Evaluation set                | Training set             | Overlap   | %      |
| ----------------------------- | ------------------------ | --------- | ------ |
| `asset_test.original`         | `wikilarge_train.source` | 5 / 359   | 1.39%  |
| `asset_validation.original`   | `wikilarge_train.source` | 42 / 2000 | 2.1%   |
| `wikilarge_validation.source` | `wikilarge_train.source` | 71 / 397  | 17.88% |
| `wikilarge_test.source`       | `wikilarge_train.source` | 2 / 121   | 1.65%  |
| `wikilarge_test.source`       | `asset_test.original`    | 121 / 121 | 100.0% |

Few-shot demonstrations found in training data: **none**

> Every figure is a floor: exact matching after normalisation, no fuzzy matching. The validation-split overlap is the one that touches a reported quantity — checkpoint selection runs on validation loss.

### Corpus cleaning: which filter, and what it costs

Artifact: `results/remote/corpus_filter_divergence.json`

| Split      | Raw    | Dropped (strict) | Dropped (targeted) | Only strict | Kept (strict) |
| ---------- | ------ | ---------------- | ------------------ | ----------- | ------------- |
| train      | 123862 | 6206             | 2205               | 4001        | 117656        |
| validation | 417    | 20               | 7                  | 13          | 397           |
| test       | 121    | 0                | 0                  | 0           | 121           |

> The tracked `data/stats.json` and the data statement describe the strict filter; the notebook trained with the targeted one. Quote whichever the reported checkpoint used — every `train_*.json` manifest records it.

### Preflight

Last run: passed

---

## What still is not measured

Kept explicit so the gap does not close by implication:

- **RQ2's site-sample study.** No GPU is needed for it and no GPU run substitutes for it: filter false-positive/false-negative rates over a real page sample remain the single largest evidence gap in the thesis.
- **Human evaluation** — scoped out to Future Work (§7.2), not pending.
- **Cohesion measurement** — scoped out to Future Work (§7.6). If a spare hour appears, the paired document outputs on disk are enough to compute TAACO-style indices with no generation and no GPU.
- ~~**Additional seeds for the document LLM condition.**~~ — **done 2026-08-24.** Seeds 2 and 3 at n=2000 (`results/remote/document_eval_llmseed{2,3}/`, one directory per seed — the merged generations cache is not keyed by `--llm-seed`, so a shared directory would overwrite the previous seed's text; RESULTS.md §4, items 10 and 15). LENS **69.22 ± 0.071**, D-SARI **22.39 ± 0.039**. §2b's `llm` row is now a mean of three, on the same terms as the sentence table's prompted rows.
