---
license: cc-by-sa-4.0
language:
  - en
base_model: facebook/bart-base
# No explicit pipeline_tag: "text2text-generation" is rejected by the Hub's current
# validation list (it renders a warning banner and mislabels the model as decoder-only
# "Text Generation"), and none of the accepted tags fit sentence simplification --
# "summarization" is wrong for a roughly length-preserving rewrite. Left unset so the
# Hub infers it from config.json's BartForConditionalGeneration instead.
tags:
  - text-simplification
  - accessibility
  - bart
datasets:
  - wikilarge
  - facebook/asset
metrics:
  - sari
  - bleu
model-index:
  - name: bart-base-wikilarge-simplification
    results:
      - task:
          type: text2text-generation
          name: Sentence-level text simplification
        dataset:
          name: ASSET (test split)
          type: facebook/asset
        metrics:
          - type: sari
            value: 37.80
          - type: bleu
            value: 88.30
          - type: fkgl
            value: 8.18
            name: FKGL (lower is better)
---

# bart-base-wikilarge-simplification

`facebook/bart-base` fine-tuned on **WikiLarge** for **sentence-level** English text
simplification. It rewrites one sentence at a time into simpler English while
preserving meaning.

Trained as part of a bachelor thesis on automated simplification of everyday English
web text, and served as the `local` model by the project's backend:
<https://github.com/yyvs/ba-simplify>

## Intended use

Simplifying a **single English sentence** of everyday prose. The companion project uses
it behind a browser extension that sentence-splits page text (`pysbd`) and sends each
sentence independently.

**Out of scope:** multi-sentence input, whole documents, languages other than English,
and any setting where a factual error would be harmful. This is a research artifact
from a student project — it hallucinates (see Limitations) and has had no human
evaluation.

For whole-document rewriting, use the companion model
[`yunvs/bart-base-dwikipedia-simplification-full`](https://huggingface.co/yunvs/bart-base-dwikipedia-simplification-full)
instead.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("yunvs/bart-base-wikilarge-simplification")
model = AutoModelForSeq2SeqLM.from_pretrained("yunvs/bart-base-wikilarge-simplification")

text = "Despite the multiplicity of antecedent conditions, the framework mandates uniform compliance."
inputs = tok(text, return_tensors="pt", truncation=True, max_length=64)
out = model.generate(**inputs, max_length=64, num_beams=4,
                     no_repeat_ngram_size=3, repetition_penalty=1.2)
print(tok.decode(out[0], skip_special_tokens=True))
```

**Use `max_length=64`.** That is the length the model was fine-tuned at, and generating
at BART's 512/1024 default degrades output. Input past 64 tokens is truncated, not
chunked — split into sentences first.

## Training

|                       |                                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| Base model            | `facebook/bart-base`                                                                                      |
| Dataset               | WikiLarge, ~117k training pairs after filtering corrupted rows                                            |
| Epochs                | 5 (best checkpoint selected at epoch 4)                                                                   |
| Batch size            | 16                                                                                                        |
| Learning rate         | 3e-5                                                                                                      |
| Weight decay          | 0.01                                                                                                      |
| Max length            | 64 tokens (WikiLarge p90 ≈ 52)                                                                            |
| Seed                  | 42                                                                                                        |
| Selection             | `load_best_model_at_end`, `metric_for_best_model="loss"`, early stopping (patience 4 quarter-epoch evals) |
| Final validation loss | 0.4568                                                                                                    |

Early stopping did not halt the run; epoch 5 (val loss 0.4590) did not improve on
epoch 4, and `load_best_model_at_end` selected epoch 4 correctly.

## Evaluation

Evaluated on the **ASSET** test split (359 sentences, 10 human references each) with
beam search (`num_beams=4`), against the un-fine-tuned `facebook/bart-base` as a
zero-shot baseline.

|                                | SARI ↑    | BLEU  | FKGL ↓   | BERTScore | unchanged |
| ------------------------------ | --------- | ----- | -------- | --------- | --------- |
| **This model**                 | **37.80** | 88.30 | **8.18** | 97.65     | 11.7%     |
| `facebook/bart-base` zero-shot | 21.39     | 91.64 | 10.02    | 98.58     | **93.9%** |

Paired bootstrap on the SARI improvement, 1000 resamples over sentence indices:
**ΔSARI +16.42, 95% CI [+15.38, +17.50]**, p < 0.001. The interval excluding zero by
fifteen points is the substantive claim; the p-value is the weaker statement.

Reproduced three times on different hardware (Apple Silicon MPS, and twice on an RTX
A5000 under a different torch build), agreeing to two decimals.

> ⚠️ **An earlier version of this card reported BLEU 67.26 / 89.89.** Those figures do
> not reproduce and have been retired. On generations giving identical SARI and FKGL,
> BLEU comes out 88.30 / 91.64 under three separate BLEU implementations
> (easse, sacrebleu`[13a]`, sacrebleu`[none]` → 85.31 — none near 67.26). The most
> likely cause is a reference-count mismatch in the original run: 67.26 lands near what
> **k=3** references produce. If you cited 67.26, use 88.30.

### Reading the BLEU number

**BLEU running opposite to SARI is expected, not a contradiction** — but note the gap is
small (3.34 points), not the 22-point chasm this card previously reported. BLEU is biased
toward conservative, low-edit output, and the zero-shot baseline is maximally
conservative: **93.9% of its outputs are byte-identical to the input.** It is being
rewarded for copying. SARI — which rewards appropriate edits against the source — and
FKGL both favour the fine-tuned model decisively (+16.42 SARI, −1.84 grade levels).

### Compared against a _strong_ baseline, not just the un-fine-tuned one

The comparison above is against this model's own starting checkpoint, which is the
weakest available baseline. Measured on the same run, against an off-the-shelf
third-party simplifier and two prompted open-weight LLMs (3-shot, temperature 1.0,
three seeds each):

| Condition      | Model                                      | SARI ↑    | BLEU  | FKGL ↓ | unchanged |
| -------------- | ------------------------------------------ | --------- | ----- | ------ | --------- |
| zero-shot      | `facebook/bart-base`                       | 21.39     | 91.64 | 10.02  | 93.9%     |
| **this model** | `yunvs/bart-base-wikilarge-simplification` | **37.80** | 88.30 | 8.18   | 11.7%     |
| off-the-shelf  | `eilamc14/bart-large-text-simplification`  | **38.17** | 88.58 | 7.85   | 13.4%     |
| prompted       | `qwen2.5:7b-instruct-q4_K_M`               | 45.95     | 75.46 | 7.77   | 4.5%      |
| prompted       | `qwen2.5:3b-instruct-q4_K_M`               | **47.03** | 68.21 | 8.22   | 1.2%      |

⚠️ **Read this honestly: fine-tuning bought no measurable advantage over an
off-the-shelf simplifier** (37.80 vs 38.17, confidence intervals overlapping almost
entirely), and **both prompted LLMs beat it by 8–9 SARI.** The defensible claim for this
model is _efficiency_, not quality: it is a **bart-base** (139M parameters) matching a
**bart-large** roughly three times its size, and it runs locally on CPU without a
prompted-LLM sidecar. If output quality is what you need and you can run a 3B
instruction-tuned model, use that instead.

BLEU runs perfectly inverse to SARI across all five rows (91.64 → 88.58 → 88.30 → 75.46
→ 68.21 against SARI 21.39 → 38.17 → 37.80 → 45.95 → 47.03), which is the cleanest
demonstration of its unsuitability for this task that this project produced.

## Limitations

- **Hallucination.** Short or fragmentary inputs can produce unrelated text; the
  phrase `"Other websites"` (a Simple English Wikipedia section heading present in the
  training data) has been observed as output for several unrelated short inputs. The
  companion backend runs a quality guard that falls back to the original text in this
  case.
- **Conservative rewriting.** Observed output often drops a subordinate clause rather
  than rephrasing dense vocabulary — it does not reliably simplify words like
  "multiplicity" or "heterogeneous entities."
- **Outperformed by prompted LLMs and matched by an off-the-shelf model.** See
  Evaluation. Choose this model for size and local CPU inference, not for peak quality.
- **No human evaluation.** All reported numbers are automatic metrics. No readability,
  meaning-preservation, or accessibility study has been conducted.
- **Single sentences only.** Multi-sentence input is out of distribution and gets
  truncated at 64 tokens.
- **English only**, and trained on Wikipedia prose — encyclopedic register, with
  whatever demographic and topical bias that corpus carries.

## Training data provenance and licence

WikiLarge is derived from English Wikipedia and Simple English Wikipedia, which are
licensed **CC BY-SA**. This model is released under **CC BY-SA 4.0** accordingly.

WikiSmall was evaluated as an alternative and deliberately rejected: it carries
baked-in named-entity anonymization artifacts unsuitable for training.

## Citation

Produced for a bachelor thesis (2026). Please cite the project repository:
<https://github.com/yyvs/ba-simplify>
