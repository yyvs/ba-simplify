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

# bart-base-dwikipedia-simplification-full

`facebook/bart-base` fine-tuned on the **full D-Wikipedia** training corpus (Sun et al.,
EMNLP 2021) for **document-level** English text simplification. It rewrites a whole
multi-sentence passage at once and is _expected_ to delete, merge, split and reorder
sentences — so its output has **no position-by-position correspondence to its input**.

Trained as part of a bachelor thesis on automated simplification of everyday English
web text, and served as the `document` model by the project's backend:
<https://github.com/yyvs/ba-simplify>

**This supersedes [`yunvs/bart-base-dwikipedia-simplification`](https://huggingface.co/yunvs/bart-base-dwikipedia-simplification)**,
which is a preliminary 20,000-document / 2-epoch run. That checkpoint is kept published
because earlier reported results cite it; this one is the checkpoint to use. Unlike its
predecessor it has been **evaluated on its actual task** — see [Evaluation](#evaluation).

## ⚠️ Input and output are lowercased and PTB-pre-tokenized

This is the most important usage detail, and getting it wrong degrades output badly.

D-Wikipedia is fully lowercased, PTB-pre-tokenized (`writer , intellectual`,
`women 's`, ` `` the second sex '' `) and one document per line, with **zero
newline characters inside any document body**. The model both **consumes and emits**
that convention.

Measured consequences of ignoring it:

- Feeding ordinary mixed-case prose produced a **factual hallucination**
  ("northern Netherlands" → "northern hemisphere") that the lowercased, corpus-style
  input did not.
- Supplying `\n` as a structural separator — the intuitive way to convey document
  structure — produced the **worst output of every variant tested**, because the model
  has never seen a newline in training.

Raw output looks like `achtkarspelen is a municipality in friesland .` — you must
de-normalize it before display. The companion project implements the matched
normalize/de-normalize pair in
[`backend/document_text.py`](https://github.com/yyvs/ba-simplify/blob/main/backend/document_text.py).

> Note for anyone who used the predecessor's normalizer before 2026-08-21: it split the
> period inside single-letter abbreviations, turning `u.s.` into `u . s .` — a token the
> corpus contains zero times. It was applied on the serving path as well as in scoring.
> Make sure you are on a fixed copy.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("yunvs/bart-base-dwikipedia-simplification-full")
model = AutoModelForSeq2SeqLM.from_pretrained("yunvs/bart-base-dwikipedia-simplification-full")

# Normalize to corpus style first: lowercase, PTB-tokenize, no newlines.
doc = "achtkarspelen is a municipality in the northern netherlands . it had a population of 27,944 in 2017 ."

inputs = tok(doc, return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**inputs, max_length=512, num_beams=4,
                     no_repeat_ngram_size=3, repetition_penalty=1.2)
print(tok.decode(out[0], skip_special_tokens=True))
# -> lowercase, pre-tokenized output; de-normalize before displaying
```

**Input past 512 tokens is truncated, not chunked** — the overflow is silently dropped
rather than simplified. Chunk below the limit yourself. A real Wikipedia article measured
2,459 BART tokens, i.e. ~79% would have been discarded if fed whole.

**Do not pack input to fill 512.** The training median is 120 tokens; a lone 100-token
section is _more_ representative of the training distribution than a merged 500-token
one. 512 is a ceiling for splitting, never a target for packing.

**Do not include headings.** When a heading was included in a section's text, the model
**echoed it back into the body output**.

## Training

|                     |                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------ |
| Base model          | `facebook/bart-base`                                                                 |
| Dataset             | D-Wikipedia (Sun et al., EMNLP 2021), **131,739** training documents after filtering |
| Scope               | **"full"** — the complete train split, strict mojibake filter                        |
| Epochs              | 5 (20,585 steps)                                                                     |
| Batch size          | 32                                                                                   |
| Precision           | bf16                                                                                 |
| Device              | NVIDIA RTX A5000, driver 535.309.01, torch 2.6.0+cu124                               |
| Learning rate       | 3e-5                                                                                 |
| Weight decay        | 0.01                                                                                 |
| Max length          | 512 tokens                                                                           |
| Seed                | 42                                                                                   |
| Wall clock          | 2 h 44 min                                                                           |
| Selected checkpoint | epoch 4 (`checkpoint-16464`), validation loss **0.3352**                             |

Validation loss by epoch: 0.3490 → 0.3408 → 0.3372 → **0.3352** → 0.3341 — monotone, flat
after epoch 3, never turning upward. Note the selected checkpoint is epoch 4's, not the
numerically lowest: evaluation ran every 0.25 epoch while saving ran per epoch, so the
best _measured_ value (0.33403, step 19551) was at an unsaved point. Quote 0.3352.

## Evaluation

Scored on **D-Wikipedia's own test split** — the correct benchmark for a document-trained
model — on the machine that trained it, one code revision throughout.

**Full test split (n = 8,000), against the un-fine-tuned base model:**

|                                | D-SARI ↑  | SARI ↑ | BLEU  | FKGL ↓   | BERTScore | LENS  |
| ------------------------------ | --------- | ------ | ----- | -------- | --------- | ----- |
| **This model**                 | **35.44** | 41.95  | 27.21 | **7.86** | **90.61** | 45.61 |
| `facebook/bart-base` zero-shot | 14.34     | 21.60  | 16.66 | 9.69     | 88.80     | 33.42 |

Paired bootstrap over documents on D-SARI, 1000 resamples: **p < 0.001**.

**Three-way comparison (n = 2,000), adding a prompted open-weight LLM**
(`qwen2.5:7b-instruct-q4_K_M`, zero-shot, sampled, one seed):

|                                | D-SARI ↑  | SARI ↑ | BLEU  | FKGL ↓   | BERTScore | LENS      |
| ------------------------------ | --------- | ------ | ----- | -------- | --------- | --------- |
| **This model**                 | **34.93** | 41.93  | 27.64 | 7.94     | **90.61** | 45.43     |
| Prompted 7B                    | 22.37     | 39.60  | 13.65 | **5.89** | 88.64     | **69.27** |
| `facebook/bart-base` zero-shot | 14.42     | 22.28  | 17.58 | 9.66     | 88.93     | 33.72     |

⚠️ **The metrics disagree about which system is best, and you should know that before
choosing this model over a prompted LLM.** D-SARI ranks this model first by 12.6 points;
**LENS — the only metric here trained on human simplification judgements — ranks the
prompted 7B first by 23.8 points.** Truncation, sample size and text normalization were
each checked and ruled out as explanations. The honest caveat on LENS is that it was
trained on _sentence-level_ human ratings and is being applied to documents, so it is out
of domain; but it cannot be dismissed. This model wins on single-reference n-gram
agreement; the prompted model wins on paraphrase quality and readability (FKGL 5.89 vs
7.94). Pick according to which you need.

LENS is reported descriptively, not significance-tested: each score is a neural forward
pass, so 1000 resamples would cost roughly 3.5 GPU-months.

## Qualitative behaviour, measured

A review of all 8,000 outputs, with the predecessor checkpoint re-probed through the same
script for comparison:

|                                          | predecessor (n=500) | **this model** (n=8000) |
| ---------------------------------------- | ------------------- | ----------------------- |
| median output/source length ratio        | 0.64                | **0.70**                |
| output below half the reference's length | 25.8%               | **18.5%**               |
| output identical to input                | 4.4%                | 7.7%                    |

**On documents whose source says "as of the 2010 census":**

|                             | predecessor | **this model** |
| --------------------------- | ----------- | -------------- |
| states a **wrong** year     | **77.1%**   | **0.2%**       |
| states the **correct** year | 14%         | **43%**        |
| states **no** year          | 9%          | 57%            |

The predecessor substituted a memorised date from the training targets in 77% of affected
documents. That is largely gone — but read the third row: the error was mostly **replaced
by omission**, not corrected. Confident factual corruption nearly vanishes and correct
preservation roughly triples, while a majority of affected documents now drop the date
entirely. For an accessibility tool that is the better failure mode, since a dropped fact
stays recoverable from the original page and a confidently wrong one does not. It is not
the same thing as accuracy.

## Limitations

- **Metric-dependent quality ranking.** See the LENS warning above.
- **Aggressive deletion.** 18.5% of outputs fall below half their reference's length, and
  it frequently omits dates and figures rather than simplifying them.
- **Output is not aligned to input.** By design it deletes, merges and reorders — you
  cannot map an output sentence back to an input sentence. Any UI must replace a whole
  block, not individual paragraphs.
- **Requires corpus-style input and output post-processing.** Not cosmetic; skipping it
  caused a measured factual hallucination.
- **Silent truncation** past 512 tokens.
- **No human evaluation.** The metric disagreement above is exactly the situation human
  judgement would resolve, and it has not been done.
- **Train/eval overlap was audited but only by exact match** after normalization; every
  reported overlap figure is a floor.
- **English only**, encyclopedic register. Trained on Wikipedia prose, not on the
  everyday web text the companion extension targets.

## Training data provenance and licence

D-Wikipedia is derived from English Wikipedia and Simple English Wikipedia (**CC BY-SA**).
This model is released under **CC BY-SA 4.0** accordingly.

> Sun, R., Jin, H., & Wan, X. (2021). _Document-Level Text Simplification: Dataset,
> Criteria and Baseline._ EMNLP 2021.

Metrics: SARI (Xu et al., TACL 2016), D-SARI (Sun et al., EMNLP 2021), LENS (Maddela,
Dou, Heineman & Xu, ACL 2023), BERTScore (Zhang et al., ICLR 2020), FKGL via `easse`.

## Citation

Produced for a bachelor thesis (2026). Please cite the project repository:
<https://github.com/yyvs/ba-simplify>
