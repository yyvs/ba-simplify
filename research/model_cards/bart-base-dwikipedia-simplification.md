---
license: cc-by-sa-4.0
language:
  - en
base_model: facebook/bart-base
# See the sentence model's card for why pipeline_tag is left unset: the Hub rejects
# "text2text-generation" and mislabels the model as decoder-only without it.
tags:
  - text-simplification
  - document-simplification
  - accessibility
  - bart
datasets:
  - d-wikipedia
---

# bart-base-dwikipedia-simplification

`facebook/bart-base` fine-tuned on **D-Wikipedia** (Sun et al., EMNLP 2021) for
**document-level** English text simplification. Unlike a sentence-level simplifier, it
rewrites a whole multi-sentence passage at once and is _expected_ to delete, merge,
split and reorder sentences — so its output has **no position-by-position
correspondence to its input**.

Trained as part of a bachelor thesis on automated simplification of everyday English
web text, and served as the `document` model by the project's backend:
<https://github.com/yyvs/ba-simplify>

> ## ⚠️ Superseded — use [`yunvs/bart-base-dwikipedia-simplification-full`](https://huggingface.co/yunvs/bart-base-dwikipedia-simplification-full)
>
> This is a _reduced-scope_ run (20,000-document train slice, 2 epochs). The full-scope
> checkpoint — the complete 131,739-document corpus at 5 epochs — scores **35.44 D-SARI
> against this model's 31.62** on D-Wikipedia's test split, and it very nearly eliminates
> this model's worst failure: substituting a memorised census year for the one in the
> input, which this checkpoint does in **77%** of affected documents against the
> successor's **0.2%**. See [Evaluation](#evaluation) and [Limitations](#limitations).
>
> This repository stays published because earlier reported results cite it. **For any new
> use, take the successor.**

## ⚠️ Input and output are lowercased and PTB-pre-tokenized

This is the most important usage detail, and getting it wrong degrades output badly.

D-Wikipedia is fully lowercased, PTB-pre-tokenized (`writer , intellectual`,
`women 's`, ` `` the second sex '' `) and one document per line, with **zero
newline characters inside any document body** (verified across 3,000 documents). The
model both **consumes and emits** that convention.

Measured consequences of ignoring it:

- Feeding ordinary mixed-case prose produced a **factual hallucination**
  ("northern Netherlands" → "northern hemisphere") that the lowercased, corpus-style
  input did not.
- Supplying `\n` as a structural separator — the intuitive way to convey document
  structure — produced the **worst output of every variant tested**, losing more
  content and hallucinating more than either alternative, because the model has never
  seen a newline in training.

> ⚠️ **If you used this project's normalizer before 2026-08-21, replace it.** It split
> the period inside single-letter abbreviations, turning `u.s.` into `u . s .` — a token
> D-Wikipedia contains **zero** times against 531 occurrences of `u.s.` in the test split
> alone. It ran on the serving path as well as in scoring. Measured cost when scoring:
> −0.14 D-SARI for this model, but **+1.29 for a copy baseline**, because D-SARI
> penalises sentence-count divergence and the spurious boundaries hit near-verbatim
> output hardest.

Raw output looks like `achtkarspelen is a municipality in friesland .` — you must
de-normalize it before display. The companion project implements the matched
normalize/de-normalize pair in
[`backend/document_text.py`](https://github.com/yyvs/ba-simplify/blob/main/backend/document_text.py):
it re-attaches punctuation, restores quotes, then recovers casing in two passes
(proper nouns and acronyms recovered from the _source_ text, since the model only ever
emits lowercase, followed by sentence-initial capitals).

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("yunvs/bart-base-dwikipedia-simplification")
model = AutoModelForSeq2SeqLM.from_pretrained("yunvs/bart-base-dwikipedia-simplification")

# Normalize to corpus style first: lowercase, PTB-tokenize, no newlines.
doc = "achtkarspelen is a municipality in the northern netherlands . it had a population of 27,944 in 2017 ."

inputs = tok(doc, return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**inputs, max_length=512, num_beams=4,
                     no_repeat_ngram_size=3, repetition_penalty=1.2)
print(tok.decode(out[0], skip_special_tokens=True))
# -> lowercase, pre-tokenized output; de-normalize before displaying
```

**Input past 512 tokens is truncated, not chunked** — the overflow is silently dropped
rather than simplified. Chunk below the limit yourself. A real Wikipedia article
measured 2,459 BART tokens, i.e. ~79% would have been discarded if fed whole.

**Do not pack input to fill 512.** The training median is 120 tokens and 68% of
training documents are ≤200, while only 8% are ≥450. A lone 100-token section is
_more_ representative of the training distribution than a merged 500-token one — 512
is a ceiling for splitting, never a target for packing.

**Do not include headings.** When a heading was included in a section's text, the model
**echoed it back into the body output**. Headings are also far shorter than anything it
saw in training.

## Training

|                        |                                                                   |
| ---------------------- | ----------------------------------------------------------------- |
| Base model             | `facebook/bart-base`                                              |
| Dataset                | D-Wikipedia (Sun et al., EMNLP 2021), 20,000-document train slice |
| Scope                  | **"reduced"** — not a full-corpus run                             |
| Epochs                 | 2                                                                 |
| Batch size             | 2, with `gradient_accumulation_steps=2`                           |
| Gradient checkpointing | Enabled                                                           |
| Precision              | bf16                                                              |
| Device                 | Apple Silicon (MPS)                                               |
| Learning rate          | 3e-5                                                              |
| Weight decay           | 0.01                                                              |
| Max length             | 512 tokens (source p95 ≈ 499)                                     |
| Seed                   | 42                                                                |

The small batch size and gradient checkpointing are MPS accommodations — on Apple
Silicon, GPU and host memory are one unified pool — not modeling choices.

`max_length=512` was measured, not inherited from BART's 1024-token ceiling: source
documents mean ≈172 tokens, p95 ≈499, p99 ≈742; targets mean ≈98. 512 covers ~p95 while
cutting self-attention's O(n²) memory cost to a quarter of the full ceiling.

Observed training loss (step | train | validation): 1250 | 0.7859 | 0.1956 —
2500 | 0.3665 | 0.1907.

## Evaluation

Scored on **D-Wikipedia's own test split** (n = 500, seed 42), the correct benchmark for
a document-trained model. These figures use the **fixed** text normalizer described
above; an earlier version of this card reported no evaluation at all.

|                                    | D-SARI ↑  | SARI ↑ | BLEU  | FKGL ↓   | BERTScore |
| ---------------------------------- | --------- | ------ | ----- | -------- | --------- |
| **This model**                     | **31.62** | 41.13  | 22.29 | **7.58** | 89.94     |
| `facebook/bart-base` zero-shot     | 14.63     | 21.48  | 18.09 | 9.54     | 89.21     |
| **Successor** (full scope, n=8000) | **35.44** | 41.95  | 27.21 | 7.86     | 90.61     |

Paired bootstrap over documents on D-SARI, 1000 resamples: **p < 0.001**.

The successor's row is on the full 8,000-document split rather than this 500-document
sample, so it is not a strictly controlled comparison — but the zero-shot baseline moves
only 0.3 D-SARI between the two samples, which is what makes the gap readable.

### A mismatched evaluation, kept for honesty

This checkpoint was also scored on **ASSET** (359 isolated _sentences_) — the wrong
benchmark for a document-trained model — before the correct one was run: SARI 33.54,
BLEU 50.83, FKGL 7.71 against the zero-shot baseline's 21.34 / 89.89 / 10.02. The
direction suggests some transfer from document training to sentence output. **It is not
a measurement of document-level simplification quality** and must not be quoted as one.

## Limitations

- ⚠️ **Substitutes a memorised date for the one in the input, in 77% of affected
  documents.** Of 35 reviewed documents whose source says "as of the 2010 census", this
  model writes **2000** instead of 2010 in 27 of them and keeps 2010 in only 5; the
  zero-shot baseline writes "2000 census" 4 times, each time copying a source that
  mentions both censuses, and no human reference says 2000. The substitution _shortens_ the
  sentence, so FKGL improves and SARI/BERTScore barely move — **every automatic metric
  here rewards it.** The full-scope successor does this in 0.2% of cases. This alone is
  reason enough to prefer the successor.
- **Superseded.** The full-corpus run completed on 2026-08-21; see the notice at the top.
- **Reduced scope.** 20,000 of 131,739 available training documents, 2 epochs of 5.
- **Output is not aligned to input.** By design it deletes, merges and reorders — you
  cannot map a sentence in the output back to a sentence in the input. Any UI must
  replace a whole block, not individual paragraphs.
- **Requires corpus-style input and output post-processing** (see above). This is not
  cosmetic; skipping it caused a measured factual hallucination.
- **Silent truncation** past 512 tokens.
- **Aggressive deletion.** A review of all 500 outputs found **25.8%** falling below half
  their reference's length, median output/source length ratio 0.64.
- **No human evaluation.** All reported figures are automatic metrics.
- **English only**, encyclopedic register.

## Training data provenance and licence

D-Wikipedia is derived from English Wikipedia and Simple English Wikipedia (**CC BY-SA**).
This model is released under **CC BY-SA 4.0** accordingly.

> Sun, R., Jin, H., & Wan, X. (2021). _Document-Level Text Simplification: Dataset,
> Criteria and Baseline._ EMNLP 2021.

## Citation

Produced for a bachelor thesis (2026). Please cite the project repository:
<https://github.com/yyvs/ba-simplify>
