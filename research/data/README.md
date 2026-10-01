# Dataset preparation notes — WikiLarge

This file is thesis material:
it documents the dataset source, cleaning decisions, and resulting stats.

## Source

[`eilamc14/wikilarge-clean`](https://huggingface.co/datasets/eilamc14/wikilarge-clean)
on Hugging Face — a pre-cleaned Hugging Face `DatasetDict` built from the
original WikiLarge alignment files (English Wikipedia → Simple English
Wikipedia sentence pairs), licensed CC BY-SA 4.0. Chosen deliberately over
re-deriving WikiLarge from the original raw files: it's already deduplicated
and filtered by token length, compression ratio, and lexical similarity, and
— not incidentally — the same author (`eilamc14`) also published the
pretrained BART model this project's backend already runs
(`eilamc14/bart-large-text-simplification`), so it's very likely the exact
data that model was trained on.

Loaded with:

```python
from datasets import load_dataset
load_dataset("eilamc14/wikilarge-clean")
```

## WikiSmall was deliberately dropped

The roadmap originally called for both WikiLarge and WikiSmall. WikiSmall was
excluded after a manual sanity check (`cestwc/adapted-wikismall` on Hugging
Face, ~89K pairs) turned up pervasive named-entity anonymization: real names,
places, and numbers replaced with placeholder tags baked into the sentences
themselves, e.g.:

```text
"PERSON@1 is a municipality in the district of PERSON@2 in the canton of LOCATION@1..."
"ORGANIZATION@1 , and was released in NUMBER@1 in LOCATION@1 ."
```

This isn't specific to that one mirror — it's inherent to how the original
WikiSmall corpus (Zhu et al., 2010) was constructed, using NER-based sentence
alignment that replaced entities with generic tags. Training on it would
teach the model to expect/produce these placeholder tokens on real input,
which is unusable for simplifying actual user-submitted web page text. Modern
simplification work has largely moved past WikiSmall for this exact reason,
preferring WikiLarge for training and TurkCorpus/ASSET (multi-reference,
no anonymization) for evaluation.

**Decision: WikiLarge only.** At 117,656 clean training pairs after cleaning
(below — 121,657 under the filter the notebook actually trained with; see the filter note),
it's also larger than WikiSmall ever was.

## Cleaning steps applied

1. **Whitespace collapsing** — `re.sub(r"\s+", " ", text).strip()`, identical
   to `extract_and_clean()` in `backend/main.py`, so training-time and
   inference-time preprocessing agree.
2. **Mojibake filtering** — WikiLarge's original alignment pipeline
   occasionally garbles multi-byte UTF-8 sequences (accented characters,
   en-dashes) into byte fragments split by a stray space, e.g.
   `"1809 â '' 11"` instead of `"1809 - 11"`, or `"TÃ xi"` instead of
   `"Táxi"`. This isn't fixable with a generic mojibake repair — `ftfy`
   doesn't catch it, since the byte sequence is no longer contiguous (a
   literal space sits where two bytes of one character used to be). Legitimate
   English/Wikipedia text essentially never contains a bare `â` or `Ã`, so any
   row containing either character was dropped rather than trained on as-is.
   Spot-checked after filtering: genuinely accented text (São Tomé, Príncipe,
   Atlético Madrid, ...) survives untouched — the filter only catches the
   broken cases.

## Resulting stats

| split      | raw rows | dropped (mojibake) | kept    | vocab size | mean src words | mean tgt words |
| ---------- | -------- | ------------------ | ------- | ---------- | -------------- | -------------- |
| train      | 123,862  | 6,206 (5.0%)       | 117,656 | 123,188    | 26.1           | 18.0           |
| validation | 417      | 20 (4.8%)          | 397     | 3,933      | 27.0           | 19.2           |
| test       | 121      | 0 (0%)             | 121     | 1,328      | 21.8           | 16.6           |

Full detail (median/min/max word lengths per split) in `stats.json`.

### ⚠️ These are `prepare_data.py`'s numbers, not the ones the sentence checkpoint trained on

The table above is produced by the **strict** filter in this script: drop any row containing
a bare `â` or `Ã`. The notebook that actually trained
`yunvs/bart-base-wikilarge-simplification`
(`../notebooks/train_sentence_and_document_pipeline.ipynb`) uses a different, **targeted**
filter — roughly twelve regexes for specific broken sequences (`â\s*''`, `Ã\s*¶`, …) — and
keeps more rows:

| filter                                                                              | train kept  | validation kept | test kept |
| ----------------------------------------------------------------------------------- | ----------- | --------------- | --------- |
| **strict** (this script → the table above, `stats.json`, the thesis data statement) | 117,656     | 397             | 121       |
| **targeted** (the notebook → what the published sentence checkpoint trained on)     | **121,657** | **410**         | 121       |

The two are **nested**: every row the targeted filter drops, the strict filter also drops
(`dropped_only_by_targeted` = 0). The strict filter drops 4,001 further train rows, and those
rows are dominated by the `Ã` class the targeted regexes miss — `AmbÃ rieux-en-Dombes`,
`VendÃ e`, `PÃ ter BartÃ k`. So the **targeted filter under-drops**; the strict one is not
over-dropping legitimate text, and the shipped checkpoint trained on the _dirtier_ corpus of
the two.

Practical consequence, and it is a citation rule rather than a code change: **quote the filter
along with the count.** Do not pair 117,656 with a result produced by the notebook. Measured
in `../results/remote/corpus_filter_divergence.json`, discussed in
`../results/RESULTS.md` §2d, regenerate with `python ../remote/dataio.py --report`.

`../remote/` (the GPU pipeline) defaults to **strict** and records the filter it used in each
run manifest, precisely so this cannot recur silently.

### ⚠️ This training split overlaps the validation split it is scored against

17.88% of the WikiLarge validation split (71 of 397 rows) also appears in train — **19.02%
(78 of 410) under the targeted filter the notebook actually applied** — so validation loss is
an optimistically biased number and a training diagnostic only, not a held-out generalisation
estimate. Separately, WikiLarge's 121-row test split is a **100% subset of
ASSET's 359-sentence test split** (both descend from TurkCorpus), so evaluations on the two
are nested rather than independent. Measured in `../results/remote/overlap_audit.json`,
discussed in `../results/RESULTS.md` §2e.

## Output format

`train.jsonl` / `validation.jsonl` / `test.jsonl`, one JSON object per line:

```json
{ "source": "...", "target": "..." }
```

These `.jsonl` files are gitignored (train.jsonl alone is ~29MB) — regenerate
them by running `research/prepare_data.py` rather than expecting them to be
checked into git. `stats.json` and this README are tracked.

## Manual sanity check

10 random train pairs were read by hand after cleaning (printed by
`prepare_data.py` on every run) — no garbled text, no anonymization
artifacts, no encoding issues found in the sample.

## Reproduce

```bash
cd research
../backend/venv/bin/python3 -m pip install -r requirements.txt
../backend/venv/bin/python3 prepare_data.py
```
