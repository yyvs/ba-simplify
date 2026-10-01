# Consolidated evaluation results

Every quantitative number this project has produced, in one table, with the provenance
needed to cite or reproduce it. Assembled 2026-08-15 for thesisDraft.md §6.2, which
previously had these figures scattered across separate run logs.

**Read the Artifact column before quoting anything.** Most historical numbers were
observed live in a notebook session and were never saved: the notebook's cells all
carry `execution_count` values from past runs but **empty stored outputs**
(`outputs: []`), and no results or log file exists under `research/scratch/`. They are
recorded here because they were genuinely measured and reported, not because they can
currently be regenerated. See §4.8 / §6.6.

Rows **S4**, **S5**, **D0** and **D2** are the exception and the template for the rest:
produced by `evaluate_sentence.py` / `evaluate_document.py` on 2026-08-16/17, each backed by
a provenance-stamped JSON carrying model ids, git commit, device, decoding parameters,
reference count, sample size and seed.

**Rows S6, D5, D6 and D2′ (2026-08-21/22) go further: one machine, one code version, one
session** — a reserved RTX A5000 — which retires the cross-hardware comparability caveats
§6.6 carries for everything above them. They are the rows to cite.

**Row D4′ (2026-08-23) is a different kind of row: a re-scoring, not a run.** It recomputes
D4's metrics from D4's own cached generations under the fixed normaliser — no model is loaded
and no text is generated, so it inherits D4's generation provenance exactly and isolates the
scoring change. It supersedes D4, and it reproduces D2′ to every recorded digit (see §3).

> ✅ **Artifact provenance, resolved 2026-08-23.** These rows briefly appeared to have no
> artifacts at all. Two things were true at once: the files had **never been copied off beet**,
> and this file **cited them by the wrong path** — the GPU pipeline writes to
> `results/remote/` (`common.py`'s `GPU_RESULTS`), so every one of these artifacts lives under
> a `remote/` prefix that the citations omitted. Both are fixed: the files were recovered from
> beet on 2026-08-23 and the paths above now carry the prefix.
>
> **Every reported figure was verified against the recovered artifact and matches to the
> decimal** — D5 (D-SARI 35.44 / 14.34), D6 (34.93 / 14.42 / 22.37), D2′ (31.62 / 14.63), and
> both LENS files (45.61 / 33.42 at n=8000; 45.43 / 33.72 / 69.27 at n=2000). S6's five
> conditions are confirmed present with `device: cuda`, 3 seeds, commit `cbd5ef6`, and its
> `unchanged_rate` values (0.939 / 0.134 / 0.117 / 0.045 / 0.012) are the ones §1 quotes.
>
> **Two traps this left behind, both worth keeping in mind when citing.** (1) There are _two_
> `sentence_eval/summary.json` files — the top-level one is **S5**'s two-condition CPU run;
> **S6 is `remote/sentence_eval/summary.json`**. Reading the wrong one is what made S6 look
> missing. (2) The stage logs under `remote/logs/` match `.gitignore`'s `*.log` and are
> therefore **not committed** — they exist locally but will not survive a fresh clone.
>
> The GPU session also wrote its own summary, `remote/RESULTS_GPU.md`, which states the correct
> artifact path under every table. It is the authority on where a 2026-08-21/22 number came
> from; folding its rows into this file is deliberately manual.

**S1, S2 and S3 predate that.** S4 re-measured S3 and reproduced its SARI and FKGL to the
decimal — but not its BLEU, which turned out not to be reproducible under any consistent
setting (see §1). Treat an unbacked number as unverified until re-measured, not as wrong;
but do not assume the whole row survives just because part of it did.

Scales: SARI, D-SARI, BLEU, BERTScore and LENS are 0–100, higher is better. FKGL is a
US grade level, lower is better.

---

## 🔒 RESEARCH FREEZE — declared 2026-08-26

**The research is closed. What is in this file is what the thesis reports.** Every item on
the eight-item completion list in §4 is discharged: the full-scope document model was trained
(D5), evaluated on the complete 8,000-document split, and frozen as a published checkpoint;
the D-SARI port was verified by differential test (§2f); the corpus-cleaning discrepancy
(§2d) and the WikiLarge/ASSET overlap (§2e) are measured and documented; the multi-condition
ASSET comparison ran (S6, plus S7); and the draft was audited against this ledger (§4 item
18). No number in this file is waiting on a run.

**What the freeze forbids, in the terms it was set:** no new research questions, models,
datasets, metrics or features. The project is not getting bigger. Anything that would add a
row to a table in this file is out of scope, and that includes attractive cheap ones —
re-running D1 to rescue its retired BLEU (§4 item 17), a fourth seed, a second decoding
sweep, one more baseline.

**What the freeze does not forbid**, because none of it changes a reported result:

- **Writing.** The thesis-writing phase is what this freeze exists to start.
- **Corrections.** If a figure here turns out to be wrong, it gets fixed and dated. A freeze
  on scope is not a freeze on accuracy.
- **Deployment and robustness engineering** on the backend and the extension. That work has
  its own roadmap item and produces no rows here.
- **The RQ2 site-sample study, which was run on 2026-08-26 — the one dated exception to the
  paragraph above, taken deliberately and with the freeze's own wording behind it.** This
  clause previously read that the study "remains a gap the thesis states rather than a run
  this freeze cancels", i.e. the freeze declined to kill it. Yunus authorised running it the
  same day, and it is now **§2g**: twelve pages, six categories, both cuts, no GPU. Nothing
  else about the freeze moves — no new model, dataset, metric or research question entered
  the project with it, and §1, §2, §2b–§2f are untouched. §2g is the last section this file
  gains.

  Two things the study changed outside itself, both permitted above independently of it:
  it found and fixed two revert defects in the extension (3.21.1, robustness, no row here),
  and it found that `profile_deployment.mjs --conditions document` had thrown since
  2026-08-21 — so §2c's **P3 figures stand as measured on 2026-08-17** while the documented
  command for reproducing them was broken; that is a correction to instructions, not to a
  number (see §5).

**The one open question is stated as a finding, not carried as a gap** (§4 item 16): D-SARI
and LENS rank the two document systems in opposite orders. Every mechanical explanation —
truncation, sample size, the normaliser, sampling, an implementation bug in D-SARI — has been
eliminated, and no automatic metric in this project can adjudicate what remains. Human
evaluation was scoped to Future Work on 2026-08-17 and stays there. §6.5 reports the
document-level ordering as metric-dependent and unresolved.

---

## 1. Sentence-level simplification (WikiLarge-trained)

**Primary result for RQ1.** Fine-tuned model, all rows:
`facebook/bart-base` fine-tuned on WikiLarge.
Baseline, all rows: `facebook/bart-base`, zero-shot, un-fine-tuned.
Decoding, S1–S4 seq2seq: beam search (`num_beams=4`, `no_repeat_ngram_size=3`,
`repetition_penalty=1.2`). S5 is the **checkpoint default**, i.e. what `backend/main.py`
serves — and it is _not_ greedy, though this file called it that until 2026-08-24. The
checkpoint's own `generation_config.json` sets `num_beams=4`, `no_repeat_ngram_size=3`,
`early_stopping=true`, so passing no decoding arguments inherits 4-beam search. The only
thing S1–S4's explicit preset adds over S5's is `repetition_penalty=1.2`. S7 is the first
genuinely greedy row (`num_beams=1`).

**S6 is the row to cite**, and it changes the shape of the claim: it is the first run with a
_strong_ baseline (a published bart-large simplifier) and prompted LLMs beside the
fine-tuned checkpoint, all on one device. It reproduces S4 exactly — see below — so it
supersedes S4 without discarding it.

| #      | Date              | Model evaluated                                                                                                                                                                                                                           | Benchmark      | n       | Refs   | System                                                              | SARI ↑    | BLEU      | FKGL ↓   | p (SARI)       | Artifact                                                                                                                                                       |
| ------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------- | ------ | ------------------------------------------------------------------- | --------- | --------- | -------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1     | before 2026-08-10 | "reduced" scope, ~20k rows / 2 epochs, **unseeded**                                                                                                                                                                                       | WikiLarge test | 50      | 1      | fine-tuned                                                          | 35.85     | 46.56     | 8.64     | < 0.001        | ❌ none                                                                                                                                                        |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | baseline                                                            | 22.26     | 46.16     | 10.10    |                |                                                                                                                                                                |
| S2     | before 2026-08-10 | same run config, **unseeded**, repeat                                                                                                                                                                                                     | WikiLarge test | 50      | 1      | fine-tuned                                                          | 33.85     | 45.53     | 8.64     | —              | ❌ none                                                                                                                                                        |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | baseline                                                            | 22.26     | 46.16     | 10.10    |                |                                                                                                                                                                |
| **S3** | **2026-08-10**    | **"full" scope, ~117k rows / 5 epochs, seeded (42)**<br>`best_checkpoint` = `checkpoint-30416` (epoch 4, val loss 0.4568)<br>now `yunvs/bart-base-wikilarge-simplification`                                                               | **ASSET test** | **359** | **10** | **fine-tuned**                                                      | **37.80** | 67.26 ⚠️  | **8.18** | **< 0.001**    | ❌ none                                                                                                                                                        |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | baseline                                                            | 21.34     | 89.89 ⚠️  | 10.02    |                |                                                                                                                                                                |
| **S4** | **2026-08-17**    | same checkpoint, via the Hub copy<br>**beam** search, matching S3's decoding                                                                                                                                                              | **ASSET test** | **359** | **10** | **fine-tuned**                                                      | **37.80** | **88.30** | **8.18** | **< 0.001**    | ✅ `sentence_eval/step4_asset_20260817T105911Z_beam_n359.json`                                                                                                 |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | baseline                                                            | 21.39     | 91.64     | 10.02    |                |                                                                                                                                                                |
| S5     | 2026-08-17        | same checkpoint<br>**checkpoint default** — 4-beam, matching what `backend/main.py` serves<br>(the artifact's filename and its `seq2seq_decoding.preset` still read `greedy`, the name this preset had when it ran; see the legend above) | ASSET test     | 359     | 10     | fine-tuned                                                          | 37.91     | 87.95     | 8.12     | < 0.001        | ✅ `sentence_eval/step4_asset_20260817T111028Z_greedy_n359.json`                                                                                               |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | baseline                                                            | 21.39     | 91.65     | 10.02    |                |                                                                                                                                                                |
| **S6** | **2026-08-21**    | **five conditions on one device**: the fine-tuned checkpoint, a **strong off-the-shelf baseline**, and **two prompted LLMs** (3-shot BLESS Prompt 2, temperature 1.0, top_p 0.9, **3 seeds** each)                                        | **ASSET test** | **359** | **10** | `base` facebook/bart-base                                           | 21.39     | 91.64     | 10.02    | — (reference)  | ✅ `remote/sentence_eval/summary.json` — **note the `remote/` prefix**: the top-level `sentence_eval/summary.json` is S5's two-condition CPU run, not this one |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | `local` **fine-tuned** (`yunvs/bart-base-wikilarge-simplification`) | **37.80** | 88.30     | 8.18     | < 0.001        |                                                                                                                                                                |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | `online` eilamc14/bart-large-text-simplification                    | **38.17** | 88.58     | 7.85     | < 0.001        |                                                                                                                                                                |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | **`llm_7b` qwen2.5:7b**                                             | **45.95** | 75.46     | 7.77     | < 0.001        |                                                                                                                                                                |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | **`llm_3b` qwen2.5:3b**                                             | **47.03** | 68.21     | 8.22     | < 0.001        |                                                                                                                                                                |
| S7     | 2026-08-24        | same checkpoint<br>**`true_greedy`** — `num_beams=1`, the first actually-greedy row<br>run on CPU to stay comparable with S4/S5                                                                                                           | ASSET test     | 359     | 10     | fine-tuned                                                          | 37.65     | 87.53     | 8.21     | — (not tested) | ✅ `sentence_eval/step4_asset_20260824T155946Z_true_greedy_n359.json`                                                                                          |
|        |                   |                                                                                                                                                                                                                                           |                |         |        | baseline                                                            | 21.25     | 91.67     | 10.03    |                |                                                                                                                                                                |

BERTScore (0–100, not in the table above because S1–S3 never measured it): S4 fine-tuned
97.65 / baseline 98.58; S5 fine-tuned 97.60 / baseline 98.59. Note `evaluate_sentence.py`
stores this on a **0–1** scale while `evaluate_document.py` uses 0–100 — the values here are
rescaled to match this file's stated convention. Worth unifying in the scripts.

Paired bootstrap on the SARI improvement over the zero-shot baseline, 1000 resamples over
sentence indices: **S4 ΔSARI +16.42, 95% CI [+15.38, +17.50]**; **S5 ΔSARI +16.52, CI
[+15.48, +17.55]**. Both p = 0.0000 → report as **p < 0.001** (1000 resamples cannot resolve
finer). The CI excluding zero by more than fifteen points is the substantive claim; the
p-value is the weaker statement.

**S4 supersedes S3 as the citable sentence-level result**, because it is the same
measurement with an artifact behind it. It also reproduces S3 where it matters: fine-tuned
SARI **37.80** and FKGL **8.18** match S3 to the decimal, baseline FKGL matches exactly, and
baseline SARI is 21.39 vs 21.34. That is strong evidence S3's SARI/FKGL were sound.

**⚠️ S3's BLEU pair does not reproduce and should not be quoted.** On generations that give
identical SARI and FKGL, BLEU comes out 88.30 / 91.64, not 67.26 / 89.89. The BLEU
implementation does not explain it (easse 88.30, sacrebleu`[13a]` 88.30,
sacrebleu`[none]` 85.31 — none near 67.26). Sweeping the number of references does, and
badly: the fine-tuned 67.26 lands near **k=3** references (66.38) while the baseline 89.89
lands near **k=9** (90.43).

| k refs     | 1     | 2     | 3         | 5     | 9         | 10    |
| ---------- | ----- | ----- | --------- | ----- | --------- | ----- |
| baseline   | 43.70 | 59.86 | 68.51     | 81.82 | **90.43** | 91.64 |
| fine-tuned | 40.15 | 57.10 | **66.38** | 78.54 | 87.52     | 88.30 |

Two numbers taken from different points on that curve are not a comparison. S3 was run on
2026-08-10, the same day the reference-handling fix described in §4.8 landed, which is the
most likely origin.

**This undercuts the BLEU explanation given below, which was rewritten on 2026-08-23 —
see that paragraph for the replacement.** The
paragraph on S3's BLEU argues that 10 references amplify a conservative baseline's
advantage. The sweep does not support that mechanism: absolute BLEU climbs steeply with k
for _both_ systems, but the gap between them stays ~3–4 points at every k. The consistent
10-reference gap is **3.34** points (91.64 vs 88.30), not 22.63. The direction survives —
the baseline is still slightly ahead on BLEU while losing SARI by 16.4 — but the magnitude
and the stated cause do not. ~~**Open decision:** rewrite that paragraph around the measured
3.34-point gap, and decide whether the "conservative output scores well on BLEU" claim
should instead rest on the `unchanged_rate` evidence.~~ — **done 2026-08-23**: rewritten
below around the 3.34-point gap, with the mechanism resting on the measured
`unchanged_rate` (the zero-shot baseline leaves **93.9%** of sentences untouched and still
scores highest on BLEU, which makes the point far more directly than the reference count
does).

**Decoding barely matters, which retires a worry rather than raising one** — but the
evidence for it was wrong until 2026-08-24, and is now right. The original claim compared
S4 ("beam") against S5 ("greedy") and found 0.11 SARI between them. Both were in fact
4-beam runs: S5 passed no decoding arguments and so inherited the checkpoint's
`num_beams=4`. The pair therefore measured `repetition_penalty=1.2` on against off, which
is why they agreed so closely, and no greedy number existed anywhere in this table.

S7 supplies one. Genuinely greedy (`num_beams=1`) scores **37.65** against S5's 37.91 and
S4's 37.80 — a 0.26 SARI spread across all three, still well inside the ~2.0-point seeding
noise floor S1/S2 establish. So the conclusion survives its correction intact: decoding is
not a threat to any reported sentence-level number, and **S5 remains the row that describes
what the extension actually serves** (the backend passes only `max_length`, so it too gets
4 beams). What changed is that the claim now rests on a greedy-versus-beam comparison
instead of on two beam runs with different penalty settings.

Two caveats on S7: it was run with `--conditions base,local`, so the significance test
against the `online` baseline was skipped (hence "not tested" rather than a p-value), and
its BERTScore is 97.60 fine-tuned / 98.60 baseline on this file's 0–100 convention.

**S6 is the number to report** (S4 for the two-condition comparison alone; they agree to
two decimals). S1/S2 are superseded — smaller single-reference test set, and run under a
harness bug (fixed 2026-08-10) that silently discarded all but the first reference, so
multi-reference scoring was never actually exercised before S3. S3 is superseded in turn:
its SARI and FKGL reproduce, its BLEU pair does not, and it has no artifact behind it.

**S1 vs S2 is the seeding noise floor**, not two competing results: identical
configuration, no fixed seed, 2.0 SARI points apart. The baseline matched to the
decimal across both runs, as expected for a deterministic zero-shot model — which is
what isolated the variance to the training-side RNG. `set_seed(42)` plus an explicit
MPS seed call was added afterwards. Quote this range whenever a single SARI difference
of ~2 points is being treated as meaningful.

**On BLEU moving opposite to SARI (rewritten 2026-08-23; the previous version of this
paragraph argued from S3's retired figures and a mechanism the k-sweep contradicts).**
The reportable BLEU pair is **88.30 fine-tuned vs 91.64 baseline** — a **3.34-point**
gap, measured three times independently (S4, S5, S6). BLEU moving opposite to SARI is
expected and is not a contradictory result, but the reason is _not_ reference
amplification: the k-sweep above shows the gap between the two systems holds at ~3–4
points at every reference count from k=1 to k=10, so ASSET's ten references inflate both
systems' absolute BLEU without widening the gap between them.

The mechanism is edit rate, and S6 measures it directly: **93.9% of zero-shot `base`
outputs are byte-identical to the input**, against 11.7% for the fine-tuned checkpoint.
BLEU rewards conservative, low-edit output, and a system that mostly copies its input
scores well on a metric that counts n-gram overlap with references that themselves share
most of the source's wording. That is the claim to make — it rests on a measured
`unchanged_rate`, not on a reference count — and S6's perfect inversion across all five
conditions (BLEU 91.64 → 88.58 → 88.30 → 75.46 → 68.21 against SARI 21.39 → 38.17 →
37.80 → 45.95 → 47.03) is the evidence for it. SARI (+16.42) and FKGL (−1.84 grade
levels) both favour the fine-tuned model over the zero-shot baseline decisively. Report
BLEU as a documented counterexample of its known limitation (§2.6), and do not read it as
"the baseline is better."

### S6 — the five-condition run, and what it does to RQ1

Run on beet (RTX A5000) in 42 minutes, all five conditions on one device, so none of §6.6's
cross-hardware caveats apply. Bootstrap: 1000 resamples over sentence indices against `base`.

✅ **Benchmark contamination checked against these exact numbers (2026-08-23).** The eight
ASSET test sentences that also appear in WikiLarge train (§2e) were dropped and all five
conditions re-scored on the 351-sentence remainder: every SARI moves by less than 0.12, the
fine-tuned − zero-shot margin by **0.006**, and the un-fine-tuned baseline loses more than the
fine-tuned checkpoint does. S6's figures may be quoted as they stand.

**It reproduces S4 exactly.** Fine-tuned SARI **37.80**, baseline **21.39**, fine-tuned BLEU
**88.30**, baseline BLEU **91.64**, FKGL 8.18 / 10.02 — every figure matches S4 to two
decimals, on different hardware and a different torch build. That is a third independent
measurement agreeing with S4, and it also settles the S3 question above: **S3's BLEU pair
(67.26 / 89.89) is the outlier and should not be quoted; 88.30 / 91.64 is the number.**

**Two new findings, and both cut against the thesis's current framing:**

1. **Fine-tuning bought nothing over an off-the-shelf simplifier.** The project's checkpoint
   scores **37.80** against the published `eilamc14/bart-large-text-simplification` at
   **38.17** — a gap of 0.37, with the two CIs ([15.38, 17.50] and [15.77, 17.82] over the
   same baseline) overlapping almost completely. RQ1 as currently phrased ("does fine-tuning
   help?") is answered decisively _yes_ against a zero-shot baseline (+16.42, CI excluding
   zero by fifteen points) and **not at all** against a strong one. Both comparisons belong
   in §6.2; reporting only the first overstates the contribution.
2. **Both prompted LLMs beat both fine-tuned models, and the smaller one wins.** `llm_3b`
   **47.03** > `llm_7b` **45.95** > `online` 38.17 ≈ `local` 37.80 > `base` 21.39. The
   ordering is not sampling noise: three seeds each, SARI sd **0.053** (3B) and **0.073**
   (7B) — the 1.07-point gap between them is roughly fifteen standard deviations. A 3B model
   at q4_K_M beating a 7B at the same quantisation is worth a sentence of its own; it is
   consistent with the document-level LENS result, where the prompted condition also wins.

**BLEU runs perfectly inverse to SARI across all five rows** (91.64 → 88.58 → 88.30 → 75.46 →
68.21 against SARI 21.39 → 38.17 → 37.80 → 45.95 → 47.03). This is the cleanest available
demonstration of §2.6's argument that reference-overlap metrics punish simplification, and it
is worth citing as such rather than as a caveat.

**The `unchanged` column quantifies the baseline's behaviour for the first time: 93.9% of
`base` outputs are identical to the input.** The claim "the zero-shot baseline mostly copies"
now has a number. The corresponding rates are 13.4% (`online`), 11.7% (`local`), 4.5%
(`llm_7b`) and 1.2% (`llm_3b`).

⚠️ **Few-shot contamination was checked, not assumed.** The prompted conditions are 3-shot,
and the overlap audit confirms **none of the demonstrations appear in the training data**
(`overlap_audit.json`). Worth stating explicitly in §6.6, since a 3-shot prompt whose
examples leaked from training would invalidate the comparison.

---

## 2. Document-level simplification (D-Wikipedia-trained)

**D5 is the document-level result for RQ1**, superseding D4: it is the full-scope checkpoint
scored on the entire test split rather than a 500-document sample. **D6 is the row to cite for
the three-way comparison**, because D5 has no prompted-LLM condition. D4′ is the last reduced-scope row (D4 re-scored under the
fixed normaliser; see below). D1 remains recorded, but only as a methodological cautionary note — it is
not a document-level finding.

**D0–D4 were scored through a defective metric normaliser, and the correction is now measured
rather than extrapolated.** D4′ (2026-08-23) re-scores D4's own cached generations with the fix
and nothing else changed: the baseline moves **+0.15** D-SARI, the fine-tuned system **+0.46**,
the prompted 7B **+0.13**, and D4's fine-tuned − baseline margin becomes **16.99, not 16.68** —
_understated_, not overstated. The earlier "≈1.3 understated / ≈1.4 overstated" reading was
extrapolated from the D2 → D2′ pair, which changed two things at once (**D2 applied no metric
normalisation at all**), and does not transfer to D4. See "The normaliser fix, isolated" below.
Every direction and every significance verdict survives. **D4′ supersedes D4** as the
reduced-scope three-way row; D4 is its pre-fix record.

| #     | Date          | Model evaluated                                                                                                                              | Benchmark                        | n   | System     | SARI  | D-SARI | BLEU      | FKGL  | BERTScore | LENS | Artifact |
| ----- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | --- | ---------- | ----- | ------ | --------- | ----- | --------- | ---- | -------- |
| D1 ⚠️ | 2026-08-11/12 | "reduced" scope, 20k docs / 2 epochs<br>`simplification_results_document/best_checkpoint`<br>now `yunvs/bart-base-dwikipedia-simplification` | **ASSET test — wrong benchmark** | 359 | fine-tuned | 33.54 | —      | ~~50.83~~ | 7.71  | —         | —    | ❌ none  |
|       |               |                                                                                                                                              |                                  |     | baseline   | 21.34 | —      | ~~89.89~~ | 10.02 | —         | —    |          |

**D1's BLEU pair is retired as of 2026-08-26 — do not quote 50.83 or 89.89 anywhere.** Both come
from the 2026-08-10/11 session whose BLEU S4, S5 and S6 all contradict; the baseline half is
known wrong (91.64 re-measured, three times, on two devices) and the fine-tuned half has never
been re-measured and now never will be, since re-running D1 would mean putting a _document_
checkpoint back on the _wrong benchmark_ to rescue one number that no argument depends on.
D1's SARI and FKGL are not retired: 21.34 / 10.02 reproduce as 21.39 / 10.02 under S4–S6, and
33.54 / 7.71 are the only ASSET figures the document checkpoint has. See the BLEU-reversal
note below for what carries that argument instead, and §4 item 17 for the decision.
| D0 | 2026-08-16 | same checkpoint (smoke run) | D-Wikipedia test | 20<br>seed 42 | fine-tuned | 41.84 | 36.05 | 17.63 | 7.20 | 88.85 | ❌ n/a | ✅ `document_eval/smoke/step4b_dwikipedia_20260816T202915Z_n20.json` |
| | | | | | baseline | 17.54 | 13.69 | 15.23 | 9.43 | 88.06 | ❌ n/a | |
| **D2** | **2026-08-16** | **"reduced" scope, 20k docs / 2 epochs**<br>`yunvs/bart-base-dwikipedia-simplification` | **D-Wikipedia test** | **500**<br>seed 42 | **fine-tuned** | **41.14** | **31.76** | 22.25 | **7.58** | 89.97 | ❌ n/a | ✅ `document_eval/step4b_dwikipedia_20260816T203137Z_n500.json` |
| | | | | | baseline | 21.52 | 13.34 | 18.09 | 9.55 | 89.26 | ❌ n/a | |
| D3 | 2026-08-17 | same checkpoint, **plus a prompted open LLM** at document granularity (`qwen2.5:7b-instruct-q4_K_M`, zero-shot, sampled, seed 1) | D-Wikipedia test | 20<br>seed 42 | fine-tuned | 41.84 | 35.77 | 17.51 | 7.20 | 88.79 | ❌ n/a | ✅ `document_eval/step4b_dwikipedia_20260817T114159Z_n20.json` |
| | | | | | baseline | 17.45 | 14.19 | 15.13 | 9.41 | 88.00 | ❌ n/a | |
| | | | | | **prompted 7B** | 38.19 | 22.62 | 10.25 | **6.07** | 87.49 | ❌ n/a | |
| **D4** | **2026-08-17** | same reduced-scope checkpoint, read from the local `scratch/simplification_results_document/best_checkpoint` rather than the Hub copy, **plus the prompted 7B** (`qwen2.5:7b-instruct-q4_K_M`, zero-shot, sampled, seed 1) | **D-Wikipedia test** | **500**<br>seed 42 | **fine-tuned** | **41.13** | **31.16** | 22.15 | **7.58** | 89.88 | ❌ n/a | ✅ `document_eval/step4b_dwikipedia_20260817T125644Z_n500.json` |
| | | | | | baseline | 21.48 | 14.48 | 17.90 | 9.54 | 89.15 | ❌ n/a | |
| | | | | | **prompted 7B** | 39.70 | **21.22** | 11.90 | **6.01** | 88.35 | ❌ n/a | |
| **D5** | **2026-08-21** | **"full" scope, 131,739 docs / 5 epochs**<br>`scratch/simplification_results_document/best_checkpoint` (= `checkpoint-16464`, epoch 4)<br>published 2026-08-22 as **`yunvs/bart-base-dwikipedia-simplification-full`**<br>trained on beet (RTX A5000) | **D-Wikipedia test — FULL split** | **8000**<br>seed 42 | **fine-tuned** | **41.95** | **35.44** | 27.21 | **7.86** | **90.61** | **45.61** | ✅ `remote/document_eval_full/step4b_dwikipedia_20260821T114841Z_n8000.json`<br>LENS: `remote/document_eval_full/lens_n8000_seed42.json` |
| | | | | | baseline | 21.60 | 14.34 | 16.66 | 9.69 | 88.80 | 33.42 | |
| **D6** | **2026-08-22** | same full-scope checkpoint as D5 (`yunvs/bart-base-dwikipedia-simplification-full`), **plus the prompted 7B** (`qwen2.5:7b-instruct-q4_K_M`, zero-shot, sampled, seed 1) | **D-Wikipedia test** | **2000**<br>seed 42 | **fine-tuned** | **41.93** | **34.93** | 27.64 | **7.94** | **90.61** | **45.43** | ✅ `remote/document_eval/step4b_dwikipedia_20260821T224703Z_n2000.json`<br>LENS: `remote/document_eval/lens_n2000_seed42.json` |
| | | | | | baseline | 22.28 | 14.42 | 17.58 | 9.66 | 88.93 | 33.72 | |
| | | | | | **prompted 7B**<br>seed 1 of 3 | 39.60 | **22.37** | 13.65 | **5.89** | 88.64 | **69.27** | |
| **D6′** | **2026-08-24** | **the same run at `--llm-seed 2` and `3`** — seq2seq systems resumed from D6's partials and re-scored bit-identically, so only the sampled condition is new | D-Wikipedia test | 2000<br>seed 42 | **prompted 7B**<br>**mean of 3 seeds** | **39.67**<br>sd 0.062 | **22.39**<br>**sd 0.039** | 13.65<br>sd 0.039 | **5.89**<br>sd 0.008 | 88.66<br>sd 0.027 | **69.22**<br>**sd 0.071** | ✅ `remote/document_eval_llmseed2/step4b_dwikipedia_20260823T214849Z_n2000.json`<br>`remote/document_eval_llmseed3/step4b_dwikipedia_20260823T230121Z_n2000.json`<br>LENS: `lens_n2000_seed42.json` in each |
| D2′ | 2026-08-21 | **D2/D4's** reduced-scope checkpoint (`yunvs/bart-base-dwikipedia-simplification`), re-scored under the **fixed** normaliser — an isolation run, not a new system | D-Wikipedia test | 500<br>seed 42 | fine-tuned | 41.13 | 31.62 | 22.29 | 7.58 | 89.94 | ❌ n/a | ✅ `remote/normaliser_isolation/step4b_dwikipedia_20260821T215233Z_n500.json` |
| | | | | | baseline | 21.48 | **14.63** | 18.09 | 9.54 | 89.21 | ❌ n/a | |
| **D4′** | **2026-08-23** | **D4's own cached generations** — same reduced-scope checkpoint, same 500 documents, same prompted-7B text — re-scored under the **fixed** normaliser. The single-variable isolation of the fix, and **the row to quote for the reduced-scope three-way comparison** | **D-Wikipedia test** | **500**<br>seed 42 | **fine-tuned** | **41.13** | **31.62** | 22.29 | **7.58** | 89.94 | ❌ n/a | ✅ `document_eval/step4b_dwikipedia_20260823T195602Z_n500.json` |
| | | | | | baseline | 21.48 | **14.63** | 18.09 | 9.54 | 89.21 | ❌ n/a | |
| | | | | | **prompted 7B** | 39.70 | **21.36** | 12.02 | **6.01** | 88.42 | ❌ n/a | |

### D5 — full-scope training on the complete test split (n=8000)

**This supersedes D4 as the document-level number for RQ1**, on two counts: it is the
checkpoint trained without Colab's session ceiling (131,739 documents at 5 epochs, against
D4's 20,000 at 2), and it is scored on the entire 8,000-document test split rather than a
500-document sample. Paired bootstrap over documents on the D-SARI improvement: p = 0.0 over
1000 resamples — **report as p < 0.001** (§4.7's resample-resolution caveat applies).

Training: 20,585 steps, batch 32, lr 3e-05, weight decay 0.01, bf16, `max_length` 512,
2h44m on an RTX A5000 (driver 535.309.01). Corpus after the strict mojibake filter: 131,739
of 132,546 raw train rows (807 dropped), 996 of 1,000 validation rows. Manifest:
`remote/train_document_full.json`; loss curve: `remote/training_history_document_full.json`.

**D4 → D5, and why the cross-sample comparison is defensible.** The two runs use different
sample sizes, which normally forbids direct comparison. The zero-shot baseline makes it
readable anyway, because it is the same model on both and moves almost not at all:

**Compare against D4′, not D4**, since D5 was scored under the fixed normaliser and D4 was
not — the like-for-like delta is **+3.82**, not +4.28:

|                                  | D4 (n=500, broken normaliser) | **D4′ (n=500, fixed)** | D5 (n=8000, fixed) | **Δ vs D4′**  |
| -------------------------------- | ----------------------------- | ---------------------- | ------------------ | ------------- |
| fine-tuned D-SARI                | 31.16                         | **31.62**              | **35.44**          | **+3.82**     |
| fine-tuned SARI                  | 41.13                         | 41.13                  | 41.95              | +0.82         |
| fine-tuned BLEU                  | 22.15                         | 22.29                  | 27.21              | +4.92         |
| fine-tuned FKGL                  | 7.58                          | 7.58                   | 7.86               | +0.28 (worse) |
| fine-tuned BERTScore             | 89.88                         | 89.94                  | 90.61              | +0.67         |
| **baseline D-SARI (the anchor)** | 14.48                         | 14.63                  | 14.34              | **−0.29**     |

The anchor moving 0.29 while the fine-tuned system moves 3.82 is the argument that the gain
is the checkpoint and not the sample. It is an argument, not a controlled comparison — but it
is now free of the normaliser confound rather than merely arguing past it, because both rows
either side of the Δ were scored by the same code.

✅ **The normaliser confound is removed rather than argued past** — see "The normaliser fix,
isolated" below. D4′ re-scores D4's generations under D5's normaliser, so the gain can be read
off two rows scored by the same code: **+3.82**, all of it training scope. Against the unfixed
D4 the figure is +4.28; the difference is the fix itself, worth +0.46 to a fine-tuned system.
The cost paid elsewhere is smaller than this file previously claimed and points the other way:
**D4's margin was understated by 0.31**, not overstated by ≈1.4, and the reduced-scope margin
to set beside D5 is D4′'s **16.99**, not D4's 16.68.

**Validation loss is a poor proxy for D-SARI here, which is itself a finding.** The full-scope
run's validation loss improved 2.0% from epoch 2 (0.3408) to epoch 5 (0.3341) — the curve is
essentially flat after epoch 3 and never turns upward, so nothing overfits. That 2% of loss
accompanies a +3.82 D-SARI move against D4′. Loss deltas of this size should not be used to
decide whether a longer document run is worth the compute.

**Checkpoint-selection wrinkle, for anyone quoting the loss.** The manifest reports
`best_metric 0.33403` beside `best_model_checkpoint: checkpoint-16464`, and those are two
different things. 0.33403 was measured at step 19551 (epoch 4.75), an evaluation point where
no checkpoint was saved — evaluation runs every 0.25 epoch while saving runs per epoch — and
that unsaved best then also prevented epoch 5 (0.3341) from being recorded as best. The
weights actually evaluated are epoch 4's, whose own `trainer_state.json` records **0.33517**
(confirmed: `best_checkpoint/model.safetensors` and `checkpoint-16464/model.safetensors` are
byte-identical). Quote **0.3352** as the selected checkpoint's validation loss, or align the
save and evaluation strategies and re-select.

**Environment**, recorded because it differs from every earlier row: Python 3.12.13, torch
2.6.0+cu124 (transformers 5.2.0 refuses `torch.load` below 2.6 — CVE-2025-32434), datasets
5.0.0, numpy 2.5.1, `easse` pinned to `6a4352ec299ed03fda8ee45445ca43d9c7673e89`. The D-SARI
port was differential-tested against the upstream RLSNLP implementation on this machine before
training: **bit-identical across 26 cases**, plus 9 invariants, artifact in the session log.

### The normaliser fix, isolated (D2 vs D2′, and D4 vs D4′)

The one thing that made D5's gain over D4 ambiguous was that `normalize_for_model` was fixed the
same day (commit `979da0b`: it split the period inside single-letter abbreviations, turning
`u.s.` into `u . s .` — a token the test split contains **zero** times against 531
occurrences of `u.s.`).

Two isolation runs exist, and **D4′ is the clean one**: it re-scores D4's own cached
generations, so the _only_ difference between the two rows is the normaliser. D2′ came first
and is the weaker comparison, because D2 applied **no** metric normalisation at all (its
artifact carries none of the normalisation note the later runs do), so D2 → D2′ moves from
_unnormalised_ to _fixed_ rather than from _broken_ to _fixed_.

|                                    | D2 (no normalisation) | D2′ (fixed) | Δ         |
| ---------------------------------- | --------------------- | ----------- | --------- |
| fine-tuned D-SARI                  | 31.76                 | 31.62       | **−0.14** |
| **baseline D-SARI**                | 13.34                 | **14.63**   | **+1.29** |
| fine-tuned SARI                    | 41.14                 | 41.13       | −0.01     |
| baseline SARI                      | 21.52                 | 21.48       | −0.04     |
| fine-tuned BLEU                    | 22.25                 | 22.29       | +0.04     |
| baseline BLEU                      | 18.09                 | 18.09       | 0.00      |
| fine-tuned/baseline FKGL           | 7.58 / 9.55           | 7.58 / 9.54 | ~0        |
| **fine-tuned − baseline (D-SARI)** | **18.42**             | **16.99**   | **−1.43** |

And the clean pair — same cached generations, the normaliser the only difference, so this is
what the _fix itself_ is worth:

|                                       | D4 (broken normaliser)   | D4′ (fixed)           | Δ                     |
| ------------------------------------- | ------------------------ | --------------------- | --------------------- |
| fine-tuned D-SARI                     | 31.16                    | 31.62                 | **+0.46**             |
| baseline D-SARI                       | 14.48                    | 14.63                 | **+0.15**             |
| prompted 7B D-SARI                    | 21.22                    | 21.36                 | **+0.13**             |
| fine-tuned / baseline / 7B SARI       | 41.132 / 21.477 / 39.705 | **bit-identical**     | **0.00**              |
| fine-tuned / baseline / 7B FKGL       | 7.579 / 9.535 / 6.014    | **bit-identical**     | **0.00**              |
| fine-tuned / baseline / 7B BLEU       | 22.15 / 17.90 / 11.90    | 22.29 / 18.09 / 12.02 | +0.14 / +0.19 / +0.12 |
| fine-tuned / baseline / 7B BERTScore  | 89.88 / 89.15 / 88.35    | 89.94 / 89.21 / 88.42 | ≈+0.06 each           |
| **fine-tuned − baseline (D-SARI)**    | **16.68**                | **16.99**             | **+0.31**             |
| **fine-tuned − prompted 7B (D-SARI)** | **9.94**                 | **10.26**             | **+0.32**             |

**Only D-SARI, BLEU and BERTScore move at all: SARI and FKGL come back bit-identical** — `easse` re-tokenises its input, so it never sees the manufactured token, while
D-SARI (this project's own port) and `sacrebleu` with `tokenize="none"` consume the normalised
string as given. That is worth stating in §2.6: the defect was invisible to two of the five
metrics _by construction_, not by luck.

Three conclusions:

1. **D5's gain is training scope, and D4′ lets it be quoted without the confound.** Scored by
   the same code either side, D5 − D4′ = **+3.82** D-SARI. The fix itself is worth +0.46 to a
   fine-tuned system on the clean pair (−0.14 on the conflated one), so it accounts for the
   difference between +4.28 and +3.82 and for nothing else. Quote **+3.82** and quote scope as
   the cause.
2. **The correction to D0–D4 is not the ≈1.3 this file previously carried, and it is not a
   constant that can be propagated.** Measured on D4: baseline **+0.15**, margin **+0.31** in
   the _same_ direction as the finding, i.e. D4 understated its own margin. The ≈1.3/≈1.4
   figures were the D2 → D2′ deltas, which include the effect of applying normalisation at all.
   **Quote D4′, not a corrected D4.** D0 and D3 (n=20 smoke rows, superseded and not cited)
   have not been re-scored; nothing extrapolates a delta onto them.
3. **D2′ and D4′ agree to every recorded digit** — 31.61664632257971 / 14.625060142908666
   D-SARI, and identically for SARI, BLEU, FKGL and BERTScore. Their generations are
   byte-identical (sha256 of the joined predictions matches for both seq2seq systems), which
   makes this the project's strongest cross-device reproduction: an A5000 on cuda (208 s / 484 s)
   and an M3 on mps (1429 s / 2340 s), two copies of the weights — the Hub revision and the
   local checkpoint directory — produce the same 500 documents character for character.

**The mechanism, and the half of it that does not survive the clean pair.** D-SARI penalises
sentence-count divergence, and the broken normaliser inserted a spurious sentence boundary at
every abbreviation. Counted directly on D4's cached generations (both normalisers applied to
the same text): the defect manufactured **154** false boundaries in the baseline's 500
outputs, **84** in the fine-tuned system's and **197** in the prompted 7B's — the baseline does
inherit nearly twice the fine-tuned system's absolute count, exactly as "a near-verbatim
copier inherits its input's abbreviations" predicts. Relative to each system's own sentence
count the inflation is identical (5.2% for both). What does _not_ follow is the score story:
the fine-tuned system recovered **+0.46** D-SARI from the fix and the baseline only **+0.15**,
so the boundary count does not translate proportionally into D-SARI. **The earlier claim that
"the metric's own preprocessing was penalising the copy baseline for copying faithfully" was
read off the conflated D2 pair and the clean pair does not support it** — the defect cost the
fine-tuned system more. What belongs in §2.6 is the surviving, sharper version: a structural
metric penalty makes preprocessing part of the metric, and the same defect moved D-SARI by
three different amounts in three systems while leaving `easse`'s SARI and FKGL untouched.

Note this was never _only_ a scoring bug: the same function normalises input on the serving
path, so the deployed model was being fed malformed abbreviations too.

### LENS — the first human-judgement metric in this project, and it disagrees

LENS (Maddela et al., ACL 2023) is the only metric here trained on human simplification
judgements, and it had never been computed: `lens-metric` conflicts with the service's
torch/transformers pins. A second virtualenv solves it (`setup.sh --lens`). Rescaled to
0–100, corpus mean per system, **descriptive only** — each score is a neural forward pass, so
1000-resample bootstrapping costs orders of magnitude more than SARI's n-gram arithmetic for
the same payoff (§2.6).

| System                  | D-SARI    | LENS      | D-SARI rank | LENS rank |
| ----------------------- | --------- | --------- | ----------- | --------- |
| fine-tuned (full scope) | **34.93** | 45.43     | **1**       | 2         |
| prompted 7B             | 22.37     | **69.27** | 2           | **1**     |
| zero-shot baseline      | 14.42     | 33.72     | 3           | 3         |

n=2000, seed 42. On the full split (n=8000, no LLM condition) the two available systems come
out at fine-tuned **45.61** and baseline **33.42** — within 0.2 and 0.3 of their n=2000
values, so the LENS column is stable across sample size.

⚠️ **LENS reverses the top of the ranking, and this is the most consequential result of the
session.** The two metrics agree that the zero-shot baseline is last. They disagree, totally,
about first place: D-SARI puts the fine-tuned model **+12.54 above** the prompted 7B, and
LENS puts the prompted 7B **+23.80 above** the fine-tuned model (both means of three seeds,
sd 0.039 and 0.071 — see the sampling entry below). Both cannot be reported as "the" answer
to RQ1.

What is _not_ the explanation:

- **Truncation.** LENS's encoder is `roberta-large`, 512 tokens. Measured on this sample:
  5.4% of sources exceed 512 tokens (mean 181.3, median 124), but only 0.0% of fine-tuned,
  0.2% of baseline and 0.6% of LLM _predictions_ do. Truncation touches a twentieth of the
  inputs and essentially none of the outputs; it cannot produce a 23.8-point reversal.
- **Sample size.** The n=8000 column reproduces n=2000 to within 0.3.
- **The normaliser.** Both systems were scored through the same fixed normaliser here.
- ✅ **Sampling — measured 2026-08-24, and this was the open one.** The prompted condition
  samples at temperature 1.0, so LENS 69.27 was a single draw. Two further seeds:
  **69.27 / 69.14 / 69.26, mean 69.22, sd 0.071.** The LENS gap over the fine-tuned model is
  **23.84 / 23.72 / 23.83, mean 23.80, sd 0.071** — the reversal is stable to the second
  decimal, and the seed-to-seed spread is **330× smaller than the gap it would have to
  explain**. D-SARI behaves the same way: 22.37 / 22.37 / 22.44, sd 0.039, against a 12.54
  gap. **Neither ordering is a lucky draw**, so the disagreement is a property of the metrics
  rather than of the sample.

What plausibly _is_ the explanation, and it is a genuine threat to the document claim:

- **LENS was trained on sentence-level human ratings (SimpEval), and this is a document-level
  application** — out of domain by construction, independent of truncation. This is the
  honest caveat and the reason the column is reported descriptively.
- **D-SARI rewards agreement with a single reference; LENS rewards paraphrase quality.** The
  LLM paraphrases freely and diverges from D-Wikipedia's one reference, which D-SARI charges
  it for and LENS does not. Note the same generations give the LLM BLEU 13.65 against the
  fine-tuned model's 27.64 — reference overlap and LENS run in opposite directions across
  every row of the table.
- **The prompted condition is handicapped in a way LENS is insensitive to.** It is fed the
  corpus's lowercased, PTB-pre-tokenised source (§2's notes), which is nothing like natural
  prose; a metric scoring fluency against human judgements may be reading past that surface
  damage where an n-gram metric cannot.

**This corroborates the sentence-level finding rather than contradicting it.** On ASSET the
prompted LLMs beat both fine-tuned models by 8–9 SARI (S6) — measured by the _same_ metric
family that puts the fine-tuned model first on documents. The pattern across both
granularities is that prompted LLMs win on human-facing and paraphrase-sensitive measures
while fine-tuned seq2seq wins on single-reference n-gram agreement. §6.2's RQ1 claim needs
to state which of those it means.

### D6′ — three seeds for the prompted condition, and what they settle (2026-08-24)

**Why this run exists.** Every seq2seq row in this file is deterministic beam search: rerun it
and the same text comes back. The prompted condition samples at temperature 1.0, so D6's LLM
row — including **LENS 69.27**, the most surprising number in the project — was a single draw.
S6 runs three seeds at sentence level and reports the spread; the document row did not. Seeds
2 and 3 were run at n=2000 on beet, identical in every other respect, with the two seq2seq
systems resumed from D6's own partial checkpoints so that only the sampled condition is new.

| Prompted 7B, n=2000 | seed 1 (D6) | seed 2 | seed 3 | mean      | **sd**    |
| ------------------- | ----------- | ------ | ------ | --------- | --------- |
| **D-SARI**          | 22.368      | 22.373 | 22.438 | **22.39** | **0.039** |
| **LENS**            | 69.269      | 69.142 | 69.260 | **69.22** | **0.071** |
| SARI                | 39.601      | 39.724 | 39.682 | 39.67     | 0.062     |
| BLEU                | 13.650      | 13.616 | 13.694 | 13.65     | 0.039     |
| FKGL                | 5.891       | 5.880  | 5.896  | 5.89      | 0.008     |
| BERTScore           | 88.644      | 88.639 | 88.689 | 88.66     | 0.027     |

**What it settles: the metric disagreement is not a sampling artefact.** The two gaps that
define §6.2's problem are stable across seeds — fine-tuned − prompted on D-SARI is
**12.565 / 12.560 / 12.495** (mean 12.54, sd 0.039), and prompted − fine-tuned on LENS is
**23.843 / 23.715 / 23.834** (mean 23.80, sd 0.071). Each spread is two to three orders of
magnitude smaller than the gap it would need to explain. **Both orderings hold in all three
runs**, so §6.2 cannot resolve the disagreement by calling either number a lucky draw, and
the document row may now be quoted as a mean ± sd on the same terms as S6's sentence rows.

**Three checks that the seeds are real seeds.** The generation caches for seeds 2 and 3 hash
differently (`1c059bcf…` vs `eb0b0f3d…`), and of the first 65 documents 64 differ from seed
1's — the failure this run was most at risk of is a cache collision returning seed 1's text
three times (§4, item 15). The **seq2seq rows are bit-identical** across all three runs on
every metric including LENS, which is what resuming from a fingerprinted partial should
produce and confirms nothing but the sampled condition moved. And the LLM's sentence-level
counterpart carries SARI sd 0.053–0.073 (S6); the document condition's 0.062 sits inside that
range, so the sampling behaves the same way at both granularities.

**Cost, for planning.** 70 minutes of GPU per seed, essentially all of it generation
(2.1 s/document); scoring and the 1000-resample bootstrap are minutes. LENS adds ~4 minutes
per seed in its own virtualenv. Resuming the seq2seq systems from D6's partials saved a
further ~57 minutes per seed.

### D4 — the reportable three-way run (n=500)

**This is the row to cite for the document branch.** It is D3's design at D2's sample size:
same 500-document seeded sample (`sample_indices_sha` `b225feab254d4af2`, identical to D2's),
same checkpoint, same decoding, with the prompted 7B added as a third system and every
system's output normalized into D-Wikipedia's convention before scoring. 2h23m on an M3
(mps): 1429 s fine-tuned, 2340 s baseline, 4539 s LLM. Backed by a git commit (`8236c26`),
unlike D2.

**The three-way ranking from D3 survives at real sample size and is decisive.** Fine-tuned
31.16 D-SARI > prompted 7B 21.22 > baseline 14.48. Paired bootstrap over documents, 1000
resamples: the LLM beats the un-fine-tuned baseline (p < 0.001) and loses to the fine-tuned
checkpoint in **all 1000 resamples**. Per document, the LLM beats the fine-tuned model on
**146/500 (29%)** and the baseline on **379/500 (76%)** — so the corpus-level verdict is not
one tail dragging a mean around.

**It also replicates D2 on different hardware**, which is the project's first cross-device
reproduction. D2 ran on a T4 (cuda, batch 16, checkpoint pulled from the Hub) and scored
without normalization; D4 ran on mps (batch 8, local checkpoint directory) and scored with
it. Same documents, same beam parameters:

|            | fine-tuned D2 → D4        | baseline D2 → D4          |
| ---------- | ------------------------- | ------------------------- |
| SARI       | 41.140 → 41.132           | 21.52 → 21.48             |
| FKGL       | 7.580 → 7.579             | 9.545 → 9.535             |
| BLEU       | 22.25 → 22.15             | 18.09 → 17.90             |
| BERTScore  | 89.97 → 89.88             | 89.26 → 89.15             |
| **D-SARI** | **31.76 → 31.16** (−0.60) | **13.34 → 14.48** (+1.14) |

SARI and FKGL reproduce to three decimals across two GPUs, two batch sizes and two copies of
the weights. Only D-SARI moves, in the same direction and by roughly the same magnitude the
D0/D3 pair measured at n=20 (−0.29 fine-tuned, +0.50 baseline), so the shift is the
normalization step, not the device. Normalization changed **127/500** fine-tuned and
**328/500** baseline predictions here, against 499/500 for the LLM — the same ~25% / ~65%
rates seen at n=20, confirming it is not the no-op the script's own note originally claimed.
Under the **fixed** normaliser (**D4′**) those rates fall to **91/500** and **315/500**, the
LLM's to 499/500: the abbreviation defect accounted for 36 of the fine-tuned changes and 13 of
the baseline's.

**The metric disagreement is still systematic, but it has narrowed where it counts.**
Fine-tuned minus prompted 7B, D3 (n=20) → D4 (n=500):

| metric    | n=20   | n=500                         |
| --------- | ------ | ----------------------------- |
| BLEU      | +7.26  | **+10.25**                    |
| D-SARI    | +13.15 | **+9.94**                     |
| SARI      | +3.65  | **+1.43**                     |
| FKGL      | −1.13  | **−1.56** (LLM more readable) |
| BERTScore | +1.30  | +1.53                         |

On plain SARI the two systems are now 1.4 points apart — inside the ~2-point noise floor
S1/S2 establish. The document-aware metric is doing nearly all the separating.

**The argument that made D3 hard to dismiss does not survive.** D3's strongest point was
that the LLM matched the reference's length profile almost exactly while the fine-tuned
model over-compressed. At n=500 it does not:

|                       | mean words      | mean sentences |
| --------------------- | --------------- | -------------- |
| D-Wikipedia reference | 77.2            | 4.3            |
| **prompted 7B**       | **92.0** (+19%) | **6.5** (+52%) |
| fine-tuned            | 59.9 (−22%)     | 3.1 (−28%)     |
| baseline              | 133.6           | 4.9            |
| source                | 132.1           | 4.9            |

The two systems miss the reference's word count by comparable margins in opposite
directions, and the LLM misses its _sentence_ count by more. (Sentence counts use
`review_document_outputs.py`'s splitter; a stricter splitter that requires a capital after
the period gives the LLM 6.37 instead of 6.47, so abbreviations do not explain it.)

**And the compression curve is where the fine-tuned model earns the gap.** Median
output/source word ratio by source length:

| source words | n   | reference | fine-tuned | prompted 7B | baseline |
| ------------ | --- | --------- | ---------- | ----------- | -------- |
| < 60         | 171 | 1.11      | 0.95       | 0.89        | 1.00     |
| 60–120       | 140 | 0.76      | 0.65       | 0.82        | 1.00     |
| 120–250      | 114 | 0.40      | **0.41**   | 0.74        | 1.00     |
| 250+         | 75  | 0.26      | **0.25**   | 0.63        | 1.06     |

The fine-tuned model tracks the human curve across a 4× range (Q1's finding, reconfirmed
with the LLM in frame); the prompted 7B applies roughly uniform mild compression and ends up
2.4× too long on the documents that need shortening most. D-SARI penalises exactly this. So
the n=20 reading — "the system whose shape matches the reference scores 13 points lower,
therefore the metric is measuring in-domain vocabulary" — is not supported at n=500. **The
single-reference in-domain confound is still real and still needs the qualitative pass to
adjudicate, but it is no longer the only available explanation for the D-SARI gap: part of
that gap is a length behaviour the LLM genuinely does not have.**

**The census probe reverses the ranking, and that is the finding to carry into §7.2.** Q1
found the fine-tuned model rewrites "the 2010 census" to 2000 in 27 of the 35 sampled
documents that mention it (a 28th output contains both years). Run over all four systems on the same 35:

| system          | preserves 2010               | says "2000 census" |
| --------------- | ---------------------------- | ------------------ |
| **prompted 7B** | **35/35**                    | 0                  |
| baseline        | 34/35                        | 4                  |
| reference       | 14/35 (rest drop the clause) | 0                  |
| fine-tuned      | 5/35                         | **28**             |

The system that loses by 9.9 D-SARI points is the only one that never corrupts the year.
No metric in this file registers that, which is the argument for human evaluation stated as
sharply as the data allows.

**Also measured:** the LLM's output guards rejected 1/500 generations (`too_short`); the
seq2seq systems were never over-deleted enough to trip a guard. Over-deletion relative to
the reference: fine-tuned 125/500 (25%) fall below half the reference's length, LLM 60/500
(12%), baseline 44/500 (9%).

**Caveats carried over from D3, unchanged:** one seed for a sampling system against
deterministic beam search (BLESS aggregates three); the LLM is prompted with the corpus's
lowercased pre-tokenized text, which is nothing like the prose it sees in the extension, and
this likely understates it; LENS still not computed.

### D3 — first three-way document run (exploratory, n=20)

**Superseded by D4**, which ran the same design at n=500. Kept because the comparison
between the two is informative: the ranking survived, the length argument below did not.

**Not a reportable result**: n=20, one seed, and the LLM condition samples while both
seq2seq systems use deterministic beam search. It is here because the _pattern_ is
informative and survives into the n=500 run or doesn't.

Significance (paired bootstrap over documents, 1000 resamples, D-SARI): the LLM beats the
un-fine-tuned baseline (p < 0.001) and loses to the fine-tuned checkpoint in **all 1000
resamples** (p = 1.0). The D-SARI gap is not marginal.

**But the metrics disagree, and the disagreement is systematic.** Ordered by how much each
metric depends on exact n-gram overlap:

| metric    | depends on exact n-grams | fine-tuned − prompted 7B      |
| --------- | ------------------------ | ----------------------------- |
| BLEU      | entirely                 | +7.26                         |
| D-SARI    | heavily                  | +13.15                        |
| SARI      | heavily                  | +3.65                         |
| FKGL      | not at all               | **−1.13** (LLM more readable) |
| BERTScore | semantic                 | +1.30                         |

The more a metric rewards matching the reference's _wording_, the worse the LLM looks. On
the semantic metric they are 1.3 points apart; on FKGL the LLM wins.

**The length profile is what makes this hard to dismiss.** D-SARI exists precisely to
penalise length and sentence-count deviation from the reference (§2.6), and the LLM matches
the reference better than the fine-tuned model does:

|                       | mean words | mean sentences |
| --------------------- | ---------- | -------------- |
| D-Wikipedia reference | 78.1       | 5.0            |
| **prompted 7B**       | **74.5**   | **5.4**        |
| fine-tuned            | 50.5       | 2.9            |
| baseline              | 121.0      | 5.3            |

⚠️ **This table does not replicate at n=500.** The LLM's apparent match to the reference is
a 20-document coincidence: over 500 documents it runs 19% long in words and 52% long in
sentences. See D4.

The fine-tuned model over-compresses badly — 50 words against a 78-word reference, 2.9
sentences against 5.0 — yet scores 13 D-SARI points higher than the system whose shape
almost exactly matches. The most economical explanation is the one that should be stated
plainly in §6.2: **D-Wikipedia has one reference per document, and the fine-tuned model was
trained on this corpus.** A single-reference n-gram metric rewards reproducing that
reference's vocabulary, which the in-domain model can do and an out-of-domain paraphrase
cannot, independent of quality. This is a confound, not a finding about simplification
ability, and the qualitative pass (§2b) is the only thing that can adjudicate it.

**What is _not_ an explanation**: the metric-normalization step introduced in the same run
(below). It moves D-SARI by less than 0.5 points; the gap under discussion is 13.

### D0 vs D3 — the cost of metric normalization, measured

D3 normalizes every system's output into D-Wikipedia's lowercased, PTB-pre-tokenized
convention before scoring, because the corpus's sources _and_ references use it and an LLM
emits ordinary prose. D0 is the same n, seed and checkpoints **without** that step, so the
pair isolates its effect exactly:

| system     | SARI  | D-SARI    | BLEU  | FKGL  | BERTScore |
| ---------- | ----- | --------- | ----- | ----- | --------- |
| fine-tuned | ±0.00 | **−0.29** | −0.12 | ±0.00 | −0.06     |
| baseline   | −0.09 | **+0.50** | −0.09 | −0.01 | −0.05     |

Small, and mixed in direction. Two consequences:

1. **D2 and D3 are not scored identically**, so they must not share a table row-for-row
   without this note. The effect is well inside the ~2-point noise floor established by
   S1/S2, so it does not threaten D2's conclusion.
2. **It was claimed to be a no-op for the seq2seq systems and it is not.** 4/20 fine-tuned
   and 12/20 baseline predictions changed (against 20/20 for the LLM). Two causes pulling
   opposite ways: `"actor.he was born"` → `"actor . he was born"` repairs a real seq2seq
   artifact, but `"u.s."` → `"u . s ."` **manufactures a token the corpus never uses** —
   D-Wikipedia's test split has `u.s.` 531 times and `u . s .` zero times.

✅ **That second cause was a bug in the shared serving path, not just in scoring** — fixed
2026-08-21 in `document_text.py` (commit `979da0b`), which fixed both at once. Until then
`normalize_for_model` split any period not adjacent to a digit, so the document checkpoint was
fed `u . s .` in production too, in a form its training data never contained. Every row up to
and including D4 was scored before the fix; **D4′** re-scores D4's generations after it.

**D4′ is the number to report for the document-level reduced-scope branch** (D4 re-scored
under the fixed normaliser, 2026-08-23), and D2 is now its cross-device corroboration rather
than a primary row. D-SARI **+16.99** (31.62 vs 14.63; D4's pre-fix +16.68, D2's
un-normalized +18.42 on the same generations), p < 0.001 on a paired bootstrap over
documents (1000 resamples — quote as p < 0.001, not p = 0.00000, per the resample-resolution
caveat in §4.7). FKGL −1.96 grade levels in both. D2 ran on a T4 in 1824 s, D4 on an M3 in
8565 s including the LLM condition; both use the same seeded 500-document sample of the
cleaned 8,000-document test split.

**D0 is superseded by D2 and D4** and should not be quoted. It was the 20-document smoke run that
validated the harness; its D-SARI (36.05) sits 4.3 points above D2's on the same checkpoint
and benchmark, which is a useful illustration of small-n instability and nothing more.

**BLEU rewards the copier, and which system that is depends on the benchmark. Rewritten
2026-08-26, off retired numbers and onto measured ones.** This paragraph used to make the
point with D1 against D2 — same two models, opposite BLEU verdict. The claim survives; the
D1 half of its evidence does not, so the argument now rests on the two rows that are backed
by artifacts and reproduce across devices.

- **The ASSET side is S6, not D1.** BLEU there runs **perfectly inverse to SARI across all
  five conditions** (91.64 → 88.58 → 88.30 → 75.46 → 68.21 against SARI 21.39 → 38.17 →
  37.80 → 45.95 → 47.03): the more a system simplifies, the worse its BLEU, monotonically,
  with no exception in the ordering. The mechanism is measured in the same run — the
  zero-shot baseline that tops BLEU leaves **93.9%** of sentences untouched.
- **The D-Wikipedia side is D2, and it is where the verdict flips.** The fine-tuned model
  wins BLEU **22.25 vs 18.09**. The systems did not change their behaviour between the two
  benchmarks; the benchmark changed what copying is worth. ASSET's references stay close to
  their sources, so a near-copying system scores well on overlap; D-Wikipedia's references
  are whole-document rewrites, so copying the source earns almost nothing and a model
  willing to delete and merge is paid for it.

⚠️ **The mechanism is edit rate against benchmark difficulty, not reference count** —
corrected 2026-08-23 and repeated here because this paragraph asserted the reference-count
reading for a week. The k-sweep in §1 rules it out: varying k from 1 to 10 lifts both
systems' absolute BLEU together while the gap between them holds at ~3–4 points. Reference
count inflates scores; it does not reverse rankings.

Absolute BLEU is also far lower on D-Wikipedia (22 vs 88) — expected with one reference and
document-length outputs, and not comparable across the two benchmarks.

**Do not over-read BERTScore.** 89.97 vs 89.26 is a 0.7-point gap on a metric that
truncates inputs past ~510 tokens and is compressed near the top of its range for any pair
of fluent English texts. It is reported because §4.7 committed to it; it separates these
systems far less than D-SARI does.

~~**Still outstanding for the document branch:** LENS was never computed (the Colab runtime
did not have `lens-metric` importable), so the one metric trained on human simplification
judgments is missing from every row. And D2 is the **reduced-scope** checkpoint — the
"full"-scope document run still has not finished.~~ — **both closed 2026-08-21/22.** The
full-scope run finished (**D5**, 131,739 documents, scored on the complete 8,000-document
split) and **LENS was computed** for D5 and D6. Neither closure made the document branch
tidier: LENS reverses D-SARI's ranking (§"LENS — the first human-judgement metric"), so the
missing metric turned into a contested one. What is still outstanding here is the _editorial_
question of which metric §6.2 privileges, plus three seeds for the prompted condition.

**D1 must not be presented as the document-level result.** ASSET is a sentence-level
benchmark with sentence-level references; this measures how a document-trained model
handles isolated single sentences, which is not the task it was fine-tuned for. It was
run only because Step 4B did not exist yet. The direction is mildly encouraging (same
SARI-up / FKGL-down / BLEU-down pattern as S3, suggesting some transfer from
document-level training), but it is not an apples-to-apples measurement.

**On D-SARI's implementation.** It is a port of the original paper's reference
implementation, verified bit-identical against that implementation's own test cases before
any of these numbers were produced — worth stating when D-SARI carries the headline claim
for the whole document branch.

**Also outstanding:** the "full"-scope document run itself never finished (Colab
session-length cap). D0/D1/D2 all concern the reduced-scope checkpoint.

### Training-loss-only observations (not evaluation results)

| Run                | Date              | Config                  | Observation                                                                                                                                                                                  |
| ------------------ | ----------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sentence "full"    | 2026-08-10        | 5 epochs, seeded        | Val loss 0.4568 @ epoch 4 (selected); 0.4590 @ epoch 5. `load_best_model_at_end` correctly kept epoch 4; `best_checkpoint/model.safetensors` confirmed byte-identical to `checkpoint-30416`. |
| Sentence "reduced" | before 2026-08-10 | 2 epochs                | Best-checkpoint val loss ≈ 0.538                                                                                                                                                             |
| Document "reduced" | 2026-08-11/12     | 20k docs, 2 epochs, MPS | Step 1250: train 0.7859 / val 0.1956. Step 2500: train 0.3665 / val 0.1907.                                                                                                                  |

Train and validation loss are **not directly comparable at a given step** — Trainer logs
training loss as a mean over the preceding logging interval, validation loss as a single
point measurement. The gap is a reporting-interval artifact, not evidence of fitting
behaviour.

---

## 2b. Qualitative review of the document-level outputs (Q1)

| #       | Date           | Reviewed                                                                                                          | Source                                                                                     | Artifact                                                          |
| ------- | -------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| **Q1**  | **2026-08-17** | all **500** outputs from D2, both systems                                                                         | `document_eval/generations_n500_seed42.json`                                               | ✅ `document_eval/qualitative_review_n500_seed42.json`            |
| **Q2**  | **2026-08-22** | all **8000** outputs from **D5**, both systems                                                                    | `remote/document_eval_full/generations_n8000_seed42.json` (`sources_sha=c30622d902a3288f`) | ✅ `remote/document_eval_full/qualitative_review_full.json`       |
| **Q1′** | **2026-08-22** | all **500** outputs from the **old** checkpoint, re-probed with the same script for a clean comparison against Q2 | `remote/normaliser_isolation/generations_n500_seed42.json`                                 | ✅ `remote/normaliser_isolation/qualitative_review_old_ckpt.json` |

Not a metric row — it carries no score and nothing here is significance-tested. It is
included in this table because it is evidence about D2 that the D2 row cannot express, and
because it is reproducible on the same terms as everything else: `review_document_outputs.py`
re-derives the sample and **aborts unless it matches the `sources_sha` the generation run
recorded** (`778390834dafdf20`), so the predictions are provably paired with the right
documents.

Three findings, in the order they matter:

- **The model learned compression, not a fixed length.** Median output/source length ratio
  by source length: 0.94 (<60 words), 0.62 (60–120), 0.39 (120–250), 0.23 (250+), against
  reference ratios of 1.09 / 0.72 / 0.38 / 0.24. It tracks the human curve across a 4×
  range. The zero-shot baseline sits at 1.00 in every bucket.
- **A systematic factual-error class that every metric in this file rewards.** Of the 35
  sampled documents whose source says "as of the 2010 census", the fine-tuned model
  rewrites the year to **2000 in 27 (77%)**; the baseline's 4 outputs containing "2000 census"
  all copy a source that mentions both censuses, and no reference says 2000. The substitution shortens the sentence (FKGL improves) and preserves nearly
  all reference n-grams (SARI/BERTScore barely move). This is the strongest argument in the
  project for the human evaluation now in §7.2.
- **Over-deletion is the dominant quality failure.** 129 of 500 outputs (26%) fall below
  half their reference's length; the worst cases are long articles reduced to a simplified
  lead paragraph, compounding `max_length=512` input truncation with learned deletion.
  D-SARI's length and sentence-count penalties already price this in, so D2's +18.4 is a
  net figure.

### Q2 — the same review against the full-scope checkpoint, and the error class is gone

Q1's findings above describe the **reduced-scope** checkpoint. Re-running the identical
script against D5's generations (Q2), with Q1′ re-probing the old checkpoint through the same
code so the two are strictly comparable:

|                                                | Q1′ old checkpoint (n=500)    | Q2 full-scope (n=8000)  |
| ---------------------------------------------- | ----------------------------- | ----------------------- |
| sources saying "as of the 2010 census"         | 35                            | 576                     |
| **fine-tuned rewrites it to 2000 (the error)** | **27** (**77.1%** of sources) | **1** (**0.2%**)        |
| fine-tuned keeps 2010 (correct)                | 5 (14%)                       | 248 (**43%**)           |
| **fine-tuned states no year at all (omits)**   | 3 (**9%**)                    | 327 (**57%**)           |
| → accuracy among outputs that state a year     | 16%                           | **>99%**                |
| baseline contains "2000 census" (the control)  | 4 / 35 (11.4%)                | 52 / 576 (9.0%)         |
| median output/source length ratio              | 0.64                          | 0.70                    |
| below half the reference's length              | 129 / 500 (**25.8%**)         | 1482 / 8000 (**18.5%**) |
| unchanged (verbatim copy)                      | 22 / 500 (4.4%)               | 613 / 8000 (7.7%)       |
| output introduces a year not in the source     | 32 / 500 (6.4%)               | 433 / 8000 (5.4%)       |

The three census rows exclude each other. The probe JSON counts "2000 census" and "2010 census"
separately (28 and 5 on Q1′, 5 and 248 on Q2), and some outputs contain both because they copy
a source that mentions both censuses: 1 on Q1′, 4 on Q2. Those count as keeping 2010.

**The substitution error is essentially eliminated — but read the omission row before quoting
that.** The memorised-census-year substitution, 77% of affected documents on the old
checkpoint and the single strongest argument in this file for human evaluation, occurs in
**0.2%** on the full-scope checkpoint. What replaced it is mostly _silence_, not correctness:
the model states no census year at all in **57%** of affected documents, up from 9%. So the
three honest figures are:

|                             | old       | full-scope |
| --------------------------- | --------- | ---------- |
| states a **wrong** year     | **77.1%** | **0.2%**   |
| states the **correct** year | 14%       | **43%**    |
| states **no** year          | 9%        | 57%        |

Correct preservation roughly **triples** (14% → 43%) and confident corruption nearly vanishes,
but a majority of affected documents now simply lose the date. For an accessibility tool that
is the better failure — a dropped fact is recoverable from the original, a confidently wrong
one is not — and it is consistent with the over-deletion profile the length rows show. It is
**not** "the model now gets the date right".

Two internal checks that this is real and not a sampling artefact:

1. **The baseline is the control and it barely moves**: 11.4% → 9.0% across the two samples.
   It is the same zero-shot model in both, so a probe behaving consistently on it is evidence
   the probe itself is stable across n=500 and n=8000.
2. **Over-deletion improved in the same direction** (25.8% → 18.5% below half the reference,
   length ratio 0.64 → 0.70), so this is not a trade where one failure mode was swapped for
   another.

⚠️ **Scope and checkpoint are confounded, as everywhere else in this comparison.** Q1′ is the
old checkpoint at n=500 and Q2 the new one at n=8000, so "full-scope training fixed it" is
the plausible reading rather than the demonstrated one. What _is_ demonstrated is that the
checkpoint the thesis now reports does not exhibit the error the thesis currently describes
as its most serious qualitative failure — which means §6.3 must be rewritten rather than
merely updated.

**This is also the clearest evidence in the project that the automatic metrics understate
what changed.** D-SARI moved +3.82 from D4′ to D5. Over the same change, the rate of
confidently stating a wrong date fell by a factor of over 400 (77.1% → 0.2%) and correct
preservation tripled — and a new failure mode, outright omission, appeared in 57% of affected
documents. A metric suite that prices all of that at +3.82 points is not measuring the
thing the error analysis cares about — which is the argument §2.6 and §7.2 both make, now
with numbers rather than assertion.

### A corpus-quality finding the review turned up, now counted (2026-08-24)

Q1 found one document whose "simplification" was not one: a disambiguation stub
(`jan kobuszewski is the name of :`) paired with a full biography. On the complete split it is
a class, not an instance. `review_document_outputs.py` gained a `corpus_pairing_probe`, and
Q1/Q2 were regenerated with it — every pre-existing figure in both artifacts is unchanged:

|                                                                        | Q1 (n=500)   | **Q2 (n=8000)**  |
| ---------------------------------------------------------------------- | ------------ | ---------------- |
| sources that are disambiguation stubs (`… may refer to :`)             | 15 — 3.0%    | **317 — 3.96%**  |
| of those, reference is an **expansion** (≥20 words and ≥3× the source) | 9 — 1.8%     | **169 — 2.11%**  |
| median source → reference length for those                             | 7 → 77 words | **5 → 61 words** |

**Why this is an evaluation finding and not only a corpus complaint.** For those documents no
correct output exists: the reference adds content the model was never shown, so D-SARI charges
it for a deletion it did not make _and_ for an addition it could not have invented. Roughly
2% of the document-level test split is scored against an impossible target — small enough not
to overturn any row here, large enough to state, and it belongs beside §4.1's WikiSmall
anonymisation finding as a limitation of automatically aligned corpora rather than of this
system.

Regenerate with `python review_document_outputs.py` from `research/` (seconds, no GPU, no
model load — it only reads cached generations).

## 2c. Deployment latency by request regime (P1–P4)

Not model quality — this is what the _deployed system_ costs on a real page, and it is the
first latency measurement the project has. §6.4 could previously only state that the two
granularities send request counts an order of magnitude apart, with no figure attached to
either. Produced by `profile_deployment.mjs`, which drives the extension's real
`content.js` (collection, chunking, and its 8-slot concurrency queue) under jsdom against
a live backend, so the request stream being timed is the one that ships.

Page for every row: the English Wikipedia article _International Game Developers
Association_ (REST HTML, revision 1356072684) — the same article §6.4's Findings 1 and 7
were measured on. Host: Apple M3, 8 cores, 16 GB. **The backend does no device placement,
so all seq2seq inference here is on CPU** — not comparable to the T4 figures above.
Backend run without `--reload`, restarted between cold runs so the cache starts empty.

| #      | Model                                | Granularity | Requests | Median chars/req | Median latency | p95     | Wall clock | Preflight | chars/s |
| ------ | ------------------------------------ | ----------- | -------- | ---------------- | -------------- | ------- | ---------- | --------- | ------- |
| **P1** | `finetuned` (bart-base, ours)        | sentence    | **79**   | 124              | **1.26 s**     | 3.28 s  | **15.5 s** | 1.36 s    | 869     |
| **P2** | `online` (bart-large, off-the-shelf) | sentence    | 79       | 124              | 3.07 s         | 11.07 s | 43.6 s     | 3.36 s    | 308     |
| **P3** | `document` (bart-base, ours)         | document    | **11**   | **1,262**        | **19.13 s**    | 19.14 s | **22.5 s** | 1.73 s    | 596     |
| **P4** | `llm_3b` (Qwen2.5-3B, prompted)      | sentence    | 79       | 124              | 13.08 s        | 20.45 s | 132.7 s    | 4.06 s    | 101     |

One cold pass each, all four within one run at 1-minute load 3.7–5.0.
Artifact: `deployment_profile/profile_igda_20260817T120443Z.json`.

**The regimes send the same prose cut two ways, which is what makes P1 and P3 comparable
at all**: 13,448 characters over 79 requests versus 13,403 over 11. Same page, same base
checkpoint, same client — a 7.2× difference in request count and a 10.2× difference in
median payload size.

**Fewer, larger requests is not faster here.** P3 sends a seventh as many requests and
still takes ~45% longer end to end, because 11 requests fill the 8-slot client queue
barely twice while 79 fill it ten times — the document path cannot hide its per-request
latency behind concurrency the way the sentence path can. Both saturate the queue (peak
in-flight 8 in every row).

**But it is far steadier, and that is the more useful finding.** P3's p95 equals its
median (19.14 vs 19.13 s): all 11 requests land in the same batches and finish together.
P1's p95 is 2.6× its median. Under load the gap becomes the whole story — see below.

**Repeatability (the two core rows, three independent cold runs each):**

|                        | Wall clock   | Median latency |
| ---------------------- | ------------ | -------------- |
| sentence (`finetuned`) | 15.8 s ± 0.5 | 1.38 s         |
| document (`document`)  | 20.8 s ± 1.1 | 17.36 s        |

Artifacts `profile_igda_20260817T120044Z.json`, `…120139Z.json`, `…120234Z.json`.

**Contention sensitivity — the finding worth quoting.** An earlier run of the same four
conditions landed while this machine was running an ASSET evaluation (1-minute load ~6.4
vs ~4). The two regimes did not degrade alike:

|                     | idle-ish (load ~4) | contended (load ~6.4) | factor   |
| ------------------- | ------------------ | --------------------- | -------- |
| sentence wall clock | 15.5 s             | 121.5 s               | **7.8×** |
| sentence p95        | 3.28 s             | 62.7 s                | **19×**  |
| document wall clock | 22.5 s             | 21.8 s                | **1.0×** |
| `online` wall clock | 43.6 s             | 350.5 s               | 8.0×     |
| `online` p95        | 11.1 s             | 240.3 s               | 22×      |
| `online` preflight  | 3.36 s             | 62.7 s                | 19×      |

The many-small-requests profile pays the contention penalty once per batch round and
there are ten of them; the few-large-requests profile pays it twice. §6.2 already records
a ~4× machine-state effect for the prompted path — this shows it is not specific to that
path, and that it interacts with the request regime rather than applying uniformly.
Artifact: `profile_igda_20260817T113240Z.json` (its rows predate per-row load capture;
the load figure is from the shell, not the file).

**Warm cache, whole page:** 41 ms (document) to 183 ms (sentence), median request 7–11 ms.
Re-toggling a page the backend has already seen is effectively free, which is worth
stating because it is the case a demo most easily shows by accident.

**`online`'s preflight independently reproduces §3.4's figure.** 3.36 s here against the
3.65–3.79 s recorded there by `curl`, measured a month apart through a different harness.

**Guard base rate, incidentally measured** (§6.4's content-selection subsection has no
other number): a guard replaced **at least one sentence** of the answer on **13–14 of 79**
sentence-path requests (16–18%) on this article, across all three sentence models, and on
**0 of 11** document-path requests. That is a base rate, not a false-positive rate — no
ground truth says whether those replacements were correct.

> ⚠️ **Wording corrected 2026-08-26.** This paragraph previously read that the guard
> "rejected the model's output and fell back to the original" on those 13–14 requests. The
> counts are right and unchanged; the description was not. `fallback_reason` is per
> _request_ but, as `backend/main.py` states, for multi-sentence input it names "the first
> sentence (in order) that needed a fallback, not a full per-sentence breakdown" — so a
> flagged request can still return a changed answer, and these 13–14 are requests
> containing a guarded sentence, not answers discarded whole. §2g hit the same trap from the
> other side: 435 of SEP's 480 requests carry a reason and 405 of them changed anyway. The
> per-sentence rate is not derivable from this response shape.

Regenerate (backend must be running **without** `--reload`; `cd research && npm install` once):

```sh
node profile_deployment.mjs --page igda --conditions finetuned,online,document,llm_3b
```

> ✅ **The document condition of this command was broken from 2026-08-21 to 2026-08-26 and
> is fixed.** Commit `963a29b` moved the granularity vocabulary into
> `extension/shared/model-labels.js`, and this script loaded `extension/content.js` alone —
> so `--conditions document` threw `GRANULARITY_WHOLE_SECTIONS is not defined` from that day
> on, silently, because nothing re-ran it. **P3 was measured on 2026-08-17, before the move,
> and stands exactly as reported**; what was broken was the documented way to reproduce it.
> Both this script and `audit_sites.mjs` now read the injected list from `manifest.json`, so
> the next moved constant cannot do this again. Found while running §2g.

---

## 2d. Corpus cleaning: two mojibake filters, and which one actually trained (2026-08-19)

Artifact: `results/remote/corpus_filter_divergence.json`. Regenerate with
`python remote/dataio.py --report`.

**The discrepancy, stated plainly: the row count this project publishes as its data
statement is not the row count the shipped sentence checkpoint was trained on.** Two
different mojibake filters exist in the repository, and each produced one of those numbers.

| filter       | where it lives                                         | rule                                                                            | train kept           | validation kept | test kept |
| ------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------- | -------------------- | --------------- | --------- |
| **strict**   | `prepare_data.py`, `remote/dataio.py` (default)        | drops any row containing a bare `â` or `Ã`                                      | **117,656** (−6,206) | **397** (−20)   | 121 (−0)  |
| **targeted** | `notebooks/train_sentence_and_document_pipeline.ipynb` | drops rows matching ~12 specific broken-sequence regexes (`â\s*''`, `Ã\s*¶`, …) | **121,657** (−2,205) | **410** (−7)    | 121 (−0)  |

Raw rows: 123,862 / 417 / 121.

**The two filters are nested, not merely different.** Every row the targeted filter drops is
also dropped by the strict one (`dropped_by_both` = 2,205 = `dropped_targeted`;
`dropped_only_by_targeted` = **0**). The strict filter drops **4,001** further train rows.

**Those 4,001 rows are genuine mojibake, so the targeted filter under-drops — the strict
filter does not over-drop.** They are dominated by the `Ã` class, which the targeted
regexes do not match: `AmbÃ rieux-en-Dombes`, `VendÃ e`, `PÃ ter BartÃ k`,
`JosÃ Miguel GonzÃ lez Rey`, `Ã glisottes-et-Chalaures`. Corrupted accented characters, not
legitimate text. This settles the direction of the discrepancy, which matters more than its
size: the published figure is the _cleaner_ corpus, and the model trained on the dirtier one.

**What this means for each reported number:**

- `research/data/stats.json`, `research/data/README.md` and thesis §2.3/§4.1 all quote the
  **strict** figures (117,656 / 397 / 121), because `prepare_data.py` produced them.
- The published sentence checkpoint `yunvs/bart-base-wikilarge-simplification` (rows
  **S3–S6**) was trained by the **notebook**, i.e. under the **targeted** filter, on
  **121,657** rows — ~4,000 more than the data statement claims (3.4% of the split), of which
  4,001 contain mojibake.
- The document runs go the other way: `remote/` defaults to `strict`, so D5's corpus of
  **131,739** D-Wikipedia documents is a strict-filtered count, and its run manifest records
  the filter it used rather than inheriting another script's number.

**The correction is editorial, not experimental.** Nothing needs re-training: 4,001 mojibake
rows in 121,657 is a data-quality footnote, not a confound that could move SARI by 16 points.
What must not survive into the thesis is the _pairing_ — quoting 117,656 as the training-set
size of a checkpoint trained on 121,657. Either quote the targeted count for S3–S6 and name
the filter, or re-run training under the strict filter so the published and trained numbers
are the same one. The first is the cheap fix and the one taken.

---

## 2e. Benchmark overlap: WikiLarge train, WikiLarge test and ASSET (2026-08-19)

Artifact: `results/remote/overlap_audit.json`, from `remote/checks/check_overlap.py`.
Matching is exact after NFKC normalisation, lowercasing and collapsing non-alphanumeric runs
to single spaces — so it sees through the tokenisation difference between the corpora, but
does no fuzzy matching. **Every figure below is a floor.**

| left                           | right                      | n       | overlapping | % of left   |
| ------------------------------ | -------------------------- | ------- | ----------- | ----------- |
| ASSET test `.original`         | WikiLarge train `.source`  | 359     | 5           | **1.39%**   |
| ASSET validation `.original`   | WikiLarge train `.source`  | 2,000   | 42          | **2.10%**   |
| WikiLarge validation `.source` | WikiLarge train `.source`  | 397     | 71          | **17.88%**  |
| WikiLarge test `.source`       | WikiLarge train `.source`  | 121     | 2           | 1.65%       |
| **WikiLarge test `.source`**   | **ASSET test `.original`** | **121** | **121**     | **100.00%** |

Three findings, in descending order of how much they affect the thesis.

**1. WikiLarge's test split is a strict subset of ASSET's test split (121/121, 100%).** This
is by construction, not a leak: ASSET (Alva-Manchego et al., 2020) re-annotated TurkCorpus's
359 test sentences with 10 references each, and WikiLarge (Zhang & Lapata, 2017) took its
held-out splits from that same TurkCorpus. But it means **S1/S2 (WikiLarge test) and S3–S6
(ASSET test) are nested evaluations, not independent ones.** That is the explanation for the
fine-tuning improvement replicating across them, and it has to be stated wherever that
replication is offered as evidence — the second measurement re-uses every sentence of the
first, so it corroborates the _scoring_, not the _generalisation_. Belongs in §6.6 as a
comparability caveat, not in §6.2 as a contamination disclosure: nothing here leaked from
training.

**2. 17.88% of the WikiLarge validation split also appears in the training split, and this
one does affect a reported quantity.** Training selects `best_checkpoint` with
`metric_for_best_model="loss"` against that split, so **the absolute validation losses this
project reports (0.4568 at the selected epoch-4 checkpoint, ~0.538 for the reduced run) are
optimistically biased** — roughly one sentence in six was seen during training. What is _not_
affected is checkpoint selection: every checkpoint faces the same contaminated split, so the
relative ordering that picked epoch 4 stands. Report those losses as training diagnostics,
never as held-out generalisation estimates. No reported SARI/BLEU/FKGL figure depends on this
split.

**3. ASSET test's overlap with training data is 1.39% — five sentences — under the strict
filter, and 2.23% (eight sentences) under the filter that actually trained** (see the re-audit
below; cite the latter). ✅ **The delta is now measured, not asserted** — every system's SARI
moves by less than 0.12 when those eight sentences are dropped, and the RQ1 margin by 0.006.
See "The held-out-remainder delta" below.

### The held-out-remainder delta, measured (2026-08-23)

The question this answers: **the eight contaminated sentences are argued to be immaterial —
does the argument survive removing them?** `evaluate_sentence.py --exclude-indices` drops them
by position (`comparisons[0].overlapping_indices` in `overlap_audit_targeted.json`:
60, 135, 137, 141, 144, 211, 228, 357) and re-scores the 351-sentence remainder from S6's own
cached generations. No model is loaded and nothing is regenerated, so the two runs differ in
exactly one thing: which sentences are scored.

| Condition                               | SARI n=359 | SARI n=351 | Δ          |
| --------------------------------------- | ---------- | ---------- | ---------- |
| zero-shot `facebook/bart-base` (`base`) | 21.389     | 21.296     | **−0.093** |
| off-the-shelf bart-large (`online`)     | 38.168     | 38.055     | **−0.113** |
| **fine-tuned (`local`)**                | **37.797** | **37.710** | **−0.087** |
| prompted 7B (3 seeds)                   | 45.953     | 45.918     | **−0.035** |
| prompted 3B (3 seeds)                   | 47.025     | 47.022     | **−0.003** |

**The contamination is immaterial, and the control is what makes that a measurement rather
than a restatement.** If those eight sentences were flattering the fine-tuned checkpoint
because it trained on them, dropping them would cost _it_ more than the others. It costs it
**less** (−0.087) than the un-fine-tuned `base` (−0.093), which has never seen WikiLarge at
all, and less than the two prompted LLMs' direction-of-travel would suggest for a
training-data effect — they were never fine-tuned on WikiLarge either and also fall. Every
system moving the same way by a similar tiny amount is what "these eight sentences happen to
be slightly easier for everyone" looks like; a contamination effect looks different.

**Every margin that carries a claim survives to the second decimal:**

| SARI margin                           | n=359      | n=351      | Δ          |
| ------------------------------------- | ---------- | ---------- | ---------- |
| **fine-tuned − zero-shot (RQ1)**      | **16.408** | **16.414** | **+0.006** |
| fine-tuned − off-the-shelf bart-large | −0.371     | −0.345     | +0.026     |
| prompted 7B − fine-tuned              | 8.156      | 8.208      | +0.052     |
| prompted 3B − fine-tuned              | 9.227      | 9.311      | +0.084     |

Paired bootstrap, fine-tuned vs zero-shot: **+16.418 [15.376, 17.499] → +16.403
[15.399, 17.443]**, p < 0.001 in both. **Put the deltas beside the noise this table already
carries:** the prompted rows' seed-to-seed sd is 0.053 (3B) and 0.073 (7B), so the RQ1 margin's
0.006 shift is an order of magnitude below the variance S6 already reports, and the largest
per-system delta (0.113) is about 1.5 seed-sds. BLEU moves −0.10 to −0.31, FKGL by at most
0.02, BERTScore in the fifth decimal.

**The n=359 control also reproduces S6 on different hardware.** It re-scores the same caches on
this laptop's CPU, and SARI, BLEU, FKGL, `unchanged_rate`, `mean_words`, the seq2seq
BERTScores and all four bootstrap CIs come back **bit-identical** to the beet/cuda run; the
only differences anywhere are the two multi-seed LLM BERTScore means, at 4×10⁻⁸ and 8×10⁻⁸.
Artifacts: `asset_heldout/step4_asset_20260823T202852Z_beam_n359.json` (control) and
`asset_heldout/step4_asset_20260823T203409Z_beam_n351.json` (remainder, which records the
excluded indices and where they came from).

**Two properties of the flag worth knowing before reusing it.** It **refuses to generate**:
cache filenames are keyed by condition, seed and decoding and say nothing about which
sentences they cover, so a subset run that generated would write a short file under the
canonical name and every later full-split rescore would read it back as complete. And a subset
run **does not overwrite `summary.json`**, because that path is what other notes mean by "the
latest run" and a 351-sentence result is not it.

**Few-shot contamination was checked and is clean.** All three BLESS Prompt 2 demonstrations
(ASSET validation indices 285, 1516, 1116) are **absent** from WikiLarge train. A 3-shot
prompt whose examples had leaked from the comparison system's training data would invalidate
S6 outright, so this is a load-bearing negative result rather than a formality.

### Audited again against the corpus that actually trained (2026-08-23)

The table above searched the **strict**-filtered 117,656-row train set, while the checkpoint
it audits trained under the **targeted** filter on 121,657 rows (§2d). Re-run with
`check_overlap.py --filter targeted`; artifact `results/remote/overlap_audit_targeted.json`,
alongside the strict run's `overlap_audit.json`.

| left ∩ WikiLarge train   | strict (117,656) | **targeted (121,657)** |
| ------------------------ | ---------------- | ---------------------- |
| ASSET test (359)         | 5 — 1.39%        | **8 — 2.23%**          |
| ASSET validation (2,000) | 42 — 2.10%       | **48 — 2.40%**         |
| WikiLarge validation     | 71/397 — 17.88%  | **78/410 — 19.02%**    |
| WikiLarge test (121)     | 2 — 1.65%        | **3 — 2.48%**          |

**Every figure went up, which is the expected direction and confirms the strict numbers were
floors** — the targeted filter keeps 4,001 more train rows, so there is more corpus for a test
sentence to collide with. Two things did not move: **WikiLarge test is still a 100% subset of
ASSET test** (121/121 — a property of how the corpora were built, not of cleaning), and the
**three few-shot demonstrations are still absent from train**, so S6's prompted conditions
remain uncontaminated under either filter.

Two defects in `check_overlap.py` were fixed to produce this table, both of which affect
anyone reproducing it. (1) **Both filter modes wrote the same filename.** Running the two
commands §5 lists, in the order it lists them, replaced the strict artifact with the targeted
one — the reason the targeted run's artifact had to be renamed by hand the first time. Strict
keeps `overlap_audit.json`; every other mode now writes `overlap_audit_<mode>.json`. (2) **The
artifact recorded five example sentences but not which positions overlapped**, so the
held-out-remainder rescore below could not be driven from it without re-implementing the
normalisation. Each comparison now carries `overlapping_indices` in full. Both artifacts were
regenerated on 2026-08-23 and **reproduce every previously published count exactly** (strict
5/42/71/2/121, targeted 8/48/78/3/121).

**Which column to cite: the targeted one**, for the same reason §2d gives — it describes the
corpus the reported checkpoint actually saw. The revisions are small but they do change two
sentences of the write-up: ASSET test contamination is **8 sentences (2.23%)**, not five, so
the held-out remainder is **351** rather than 354; and the validation-in-train share is
**19.02%**, so the optimistic bias on the reported validation losses is nearer one sentence in
five than one in six. Neither changes any conclusion above.

---

## 2f. Metric-implementation verification: D-SARI ported, then differential-tested (2026-08-23)

D-SARI carries the entire document-level claim, and no maintained Python package implements
it, so `evaluate_document.py` ports it from the paper's own reference implementation
(`D_SARI.py` in `RLSNLP/Document-level-text-simplification`).

**The verification claim used to be unreproducible.** Thesis §2.6 stated the port was
"verified numerically identical (bit-for-bit, to full floating-point precision) against that
reference implementation's own bundled test cases" — but upstream ships no test cases, there
was no test for this in the repository, and there were no tests for `research/` at all. The
claim was plausible in substance and impossible to check, which for the metric the whole
document-level result rests on is the weakest link in the chain.

**It is now executed rather than asserted:** `research/remote/checks/test_d_sari.py` downloads
upstream `D_SARI.py` at run time, calls both implementations on the same inputs, and asserts
agreement to full float precision. Last run 2026-08-23:

```text
10 checks, 0 failed
PASS  bit-identical to upstream across 66 cases
REFERENCE VERIFICATION: upstream comparison PASSED
```

The 66 cases are drawn from real D-Wikipedia documents (train/validation/test), not
hand-written fixtures. Nine invariants hold with or without network access (`--offline`):
range and finiteness on identity / copy / over-deletion / sentence-split / single-token /
multi-reference inputs, determinism across repeated calls, scoring the reference above a copy
of the source, and the sentence-count penalty favouring the reference's segmentation. The
script's final line always states whether the upstream comparison actually ran, because
"invariants pass" and "verified against upstream" are different claims.

**Two upstream quirks are deliberately preserved**, so scores stay comparable to D-SARI as the
literature reports it rather than to a version this project judged more principled: the
_delete_ n-gram score returns **precision only**, not an F1 like keep and add (which follows
SARI itself — Xu et al., 2016 use deletion precision by design), and one intermediate
delete-recall term is computed and never used.

**The port's only textual divergence from upstream is two `max(…, 1)` divisor floors, and they
are unreachable no-ops.** Upstream divides `LP_1` by `output_length` and `SLP` by
`max(reference_sentence_number, output_sentence_number)` with no floor; the port floors both.
Neither can be hit, because `"".split(" ")` returns `[""]` — length 1, never 0 — so
`output_length` is never zero. Checked directly: on an empty prediction the port and upstream
both return **0.26672979797979796**, identical to the last digit. Upstream's `LP_2` already
carries the same `max(input_length - reference_length, 1)` floor the port has.

**What this does and does not establish.** It establishes that this project's D-SARI _is_ the
published D-SARI, so D5's 35.44 and D6's 34.93 are comparable to D-Wikipedia numbers in the
literature. It does not validate D-SARI as a metric — §2's LENS disagreement, which ranks the
prompted model 23.8 points the other way, is the standing argument that it may be measuring
the wrong thing.

---

## 2g. RQ2 site-sample audit: twelve real pages, both cuts (2026-08-26)

**The study §6.4 recorded three times as not done.** Not model quality and not latency —
what the deployed selection, replacement and revert machinery does to pages nobody wrote
for the purpose. Produced by `audit_sites.mjs`, which runs the extension's own content
scripts (every file `manifest.json` injects, in manifest order) under jsdom against a live
backend, so the selection being audited is the one that ships: `findContentScope()`,
`collectSections()`, `computeLeafCandidates()`, `collectChunkNodes()`,
`shouldSimplifyElement()` and `isSimplifiableChunk()` all decide for themselves, and the
skip reasons are reported by content.js's own `skipBreakdown()` rather than recomputed.

Twelve pages, two per category, chosen before any was run. Both models on CPU, one host,
one commit (`5989ab4`): `finetuned` = `yunvs/bart-base-wikilarge-simplification` (sentence),
`document` = `yunvs/bart-base-dwikipedia-simplification-full` (whole sections).
Artifacts: `site_audit/site_audit_20260826T044008Z.json` plus `_items.jsonl` (11,867
registered leaves), `_units.jsonl` (2,571 requests) and `_labelling.jsonl` (130 hand-labelled
items). Pages are cached under `research/scratch/sites/` with fetch time and a sha256 prefix.

### Coverage

| Cut      | Pages | Found | Skipped         | Sent        | Changed                 | Skip-share range |
| -------- | ----- | ----- | --------------- | ----------- | ----------------------- | ---------------- |
| sentence | 12    | 8,777 | 6,553 (**75%**) | 2,224 (25%) | 1,622 (**73%** of sent) | 22% – 95%        |
| document | 12    | 3,412 | 3,101 (**91%**) | 311 (9%)    | 258 (**83%** of sent)   | 54% – 99%        |

**Category predicts coverage better than anything else does, and the spread is the result.**
Sentence-cut skip share runs from **22%** (Stanford Encyclopedia) to **95%** (MDN); the
document cut from **54%** to **99%** (Allbirds, which sent 3 of 330). A single-page test
could have concluded almost anything, which is the methodological point §6.4 already makes
about hand-written fixtures, now measured across a sample instead of argued.

**The two cuts' skip rates are not comparable and must not be quoted side by side without
this sentence.** Document mode registers every leaf and sends heading-delimited sections, so
its skips are dominated by two reasons that are decisions rather than rejections:
`section-heading` (the delimiter it cuts on) and `not-document-body-tag`. On Wikipedia those
two are 226 of 949 skips; on Python docs 552 of 725.

### Content selection, judged rather than counted

149 (sentence) and 156 (document) leaves of ≥12 words sat inside the page's own content
scope, outside `nav/footer/header/aside/form`, and were not sent — the candidate false
negatives. A stratified, content-hash-deterministic sample of **69 skipped** and **61
unchanged** items was read and labelled (`_labelling.jsonl`):

| Skipped (n=69)            |        | Unchanged (n=61)                |       |
| ------------------------- | ------ | ------------------------------- | ----- |
| correct                   | 48     | already plain                   | 49    |
| **should have been sent** | **15** | **should have been simplified** | **9** |
| correct but costly        | 6      | language unsupported            | 3     |

**The code filter has a measured false-positive rate, and one character causes it.**
`shouldSimplifyElement()` rejects text matching
`/\b(function|var|let|const|=>)\b|[{}<>;=]/`. Of **487** `code-like-text` skips, **102** were
triggered by a semicolon alone; 31 of those are prose-length items inside the content area,
and reading them splits 17 running prose against 14 bibliography entries where the semicolon
separates author names. The false positives are core content: the _Photosynthesis_ article's
`Carbon dioxide is converted into sugars in a process called carbon fixation; photosynthesis
captures energy from sunlight…` (97 words), GOV.UK guidance, four SEP paragraphs, five Python
doc paragraphs. **`{` and `}` cost nothing** — every brace-only skip in the sample was real
code, including a Wikipedia `<span>` carrying a stylesheet. Angle brackets cost 7, of which 5
are MDN prose quoting CSS value syntax (`flex: <flex-grow> 1 0%`), where withholding is
defensible. This is the first false-positive rate any filter in this project has.

**The opt-out convention costs real prose on one page, correctly.** Six sampled IKEA
paragraphs of genuine product guidance (`Keep in mind that you need enough space between the
top of the furniture and the ceiling…`) are inside `translate="no"`/`aria-hidden` subtrees.
The extension honours the page's own opt-out, which is right; the reader still loses the
text. Labelled `correct-but-costly` rather than folded into either column.

> **Why two different totals for "sent" appear below (2,224 and 2,222).** The coverage table
> takes the extension's own counters, incremented as each item resolves; the per-item dump is
> a post-run DOM walk over `CANDIDATE_SELECTOR`, and `<input>` is not in that selector — so
> the two `<input>` placeholders that were sent (`Search news, topics and more` on the BBC
> page, `What are you looking for?` on IKEA's, both five words) are counted by the tally and
> invisible to the walk. One item per page, both accounted for; every per-item figure below
> is over 2,222 and every counter figure over 2,224.

**The send threshold is the largest source of wasted requests.** `MIN_WORDS_TO_SIMPLIFY` is 4. Of the 2,222 sentence-cut items the per-item dump carries, **647 (29%)** are under 6 words — dates, nav labels,
`Edit this page on GitHub` — and 19 of the 61 sampled unchanged items are of that kind:
correctly unchanged, and never worth a request. Raising the threshold to 6 words would drop
those 647 while touching **no** item of ≥12 words; its real cost is 349 short-text rewrites
(22% of all changes) whose value is unlabelled.

### DOM and link preservation

Across all 24 runs: **no tag was ever lost** (`tags_lost` empty on every row), every `<img>`
survived with its `src`, every `<a>` kept its `href`, and the anchor count was identical
before and after on every page. The failure mode is narrower and is about link _labels_.

| Cut      | Anchors | Labels emptied | …misplaced by the write-back | …deleted by the model |
| -------- | ------- | -------------- | ---------------------------- | --------------------- |
| sentence | 6,320   | 331 (5%)       | **94**                       | 237                   |
| document | 6,320   | 422 (7%)       | **34**                       | 388                   |

**Splitting that count is the whole finding, because the two halves have different owners.**
Where the model deleted the phrase a link sat on, nothing anchors and the link is left
standing but empty — the documented, tested behaviour of `writeChunkText` (see
`inline-markup.test.js`'s `aDeletedLabelLeavesTheLinkInPlace`: "The link stays in the page —
it is not the extension's to remove"). What the audit adds is how often that stated limit is
reached. The **misplaced** half is the §5.4 fragmentation class recurring: words the model
kept, written into the wrong node of the same element. It is **concentrated almost entirely
on one page** — 93 of the sentence cut's 94 are the _Photosynthesis_ article, whose paragraphs
carry dense inline linking; every other page in the sample scores 0 or 1. Link density, not
page size, is the driver: SEP sends more prose than Wikipedia (480 requests against 278) and
misplaces nothing.

### Revert

| Check                                                      | Result      |
| ---------------------------------------------------------- | ----------- |
| `document.body.textContent` identical after `revertPage()` | **24 / 24** |
| per-element attribute _set_ identical                      | **24 / 24** |
| raw HTML string identical                                  | **18 / 24** |

The six remaining differences are cosmetic and fully accounted for — five of them are a
net one to four bytes (tagesschau, Wikipedia in both cuts, IKEA in both cuts) and the sixth
is the Allbirds page at +913. Touching
`el.style.backgroundColor` makes CSSOM re-serialise the page's _own_ inline styles in
canonical form (`margin: 0 0 1.15rem` → `margin: 0px 0px 1.15rem`, `color: #1a1a1a` →
`color: rgb(26, 26, 26)`; 133 such lines on the Allbirds page, which is the whole of its
+913-byte delta), and on one element a trailing space inside a class list is lost to
`classList.add`/`remove`. Computed styles are unchanged. Restoring the `class` and `style`
strings verbatim would fix the bytes and introduce a worse bug — it would discard whatever
the page's own JavaScript changed while the page was simplified — so it is deliberately not
done; the clean fix is to stop writing an inline style at all.

**Two revert defects were found here and fixed (extension 3.21.1).** The page's own `title`
attribute was deleted by the revert, because the simplified element's tooltip is written over
it and `revertPage()` called `removeAttribute("title")` unconditionally; and `class=""` /
`style=""` were left on elements that had shipped neither. Both violated a contract
`inline-markup.test.js` already claimed to test — it compares the simplified element's
`innerHTML`, which cannot see the element's own attributes. `revert-attributes.test.js` covers
both, and fails against the old code. The first fix was itself wrong in a way this sample
caught: removing an empty attribute at revert time deletes a `class=""` that nextjs.org
genuinely ships (4 `<li>`s), so the shipped version records what it added instead of
inferring it.

### The finding that matters most: unguarded meaning inversion

**Five sentences came back verbatim except that a negation had been deleted, and no guard
fired on any of them.** Detected conservatively — the output sentence must equal the input
sentence with exactly one negation token removed — so this is a lower bound, blind to any
inversion where the sentence was also reworded.

| Page       | Input                                                                                                                | Output                                  |
| ---------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| allbirds   | `Final Sale items, or returns that do not comply with our terms and conditions **cannot** be returned or exchanged.` | `…can be returned or exchanged`         |
| allbirds   | `If you decide **not** to buy return coverage…`                                                                      | `If you decide to buy return coverage…` |
| pydocs     | `The aws iterable must **not** be empty.`                                                                            | `must be empty`                         |
| tagesschau | `Ein koordinierter Wiederaufbau ist **nicht** in Sicht.`                                                             | `ist in sicht`                          |
| tagesschau | `Der Weg hin zu den eigenen vier Wänden war **nicht** einfach.`                                                      | `war einfach`                           |

Three are English, so this is not a consequence of the German page. **The cause is
documented policy, not an oversight**: `backend/main.py` states that "Word-level edits are
never rejected here -- one dropped word can already be a real simplification", and a dropped
negation is exactly that edit class. `_looks_like_hallucination`, `no_meaningful_change` and
`corpus_artifact` all pass these outputs. For a system whose purpose is to make public-
information text easier to read, an inverted returns policy and an inverted API contract are
the most serious results in this file. A polarity check in `change_guard.py` is the obvious
mitigation and is **not** implemented: it changes the system under measurement.

### The German page, as a language-coverage result

Both checkpoints are English-only and nothing checks the language of what is sent. On the
tagesschau article the sentence cut sent 65 units and changed 42. Besides the two inversions
above, the document cut applied D-Wikipedia's lowercasing convention (§2's Finding 2) to
German and produced non-words — `die meisten häusers zer stört` for `die meisten Häuser
zerstört`, `grauen betontrümmenn` for `Betontrümmern`, `seines zersteten Hauses` for
`zerstörten` — and truncated the section at the 512-token ceiling. No guard fired. The
extension has no language gate; the honest statement is that the system is English-only in
capability and silent about it in behaviour.

### The content-scope question, answered as a counterfactual

Sentence mode walks the whole `<body>`; document mode scopes to
`findContentScope()` (`article` → `main` → `#content` → `#mw-content-text` → `body`). Whether
the sentence cut should do the same was tested without changing the extension:

| Variant                                | Items it would stop sending (of 2,222) | Prose (≥12 words) lost |
| -------------------------------------- | -------------------------------------- | ---------------------- |
| scope to `findContentScope()`          | 592 (27%)                              | 151                    |
| exclude `nav/footer/header/aside/form` | 578 (26%)                              | 143                    |
| raise the send threshold to 6 words    | 647 (29%)                              | **0**                  |

The two structural variants are within 5% of each other on both axes — the intuition that
excluding furniture is much safer than scoping is not supported. Both concentrate their cost
on one page: 133 of scoping's 151 lost prose items are Allbirds' returns policy, privacy
rights and terms of service, at 160–214 words each. For a plain-language tool that is close
to the highest-value text on the page. `findContentScope()` also falls through to `body` on
3 of 12 pages here, so it is not reliable enough to trust silently, which is the §5.6
leaf-`<div>` failure mode. The threshold is the better lever and the only one with no
prose cost.

### Limits of this study, stated

- **jsdom is not a browser.** These are the DOMs these sites _ship_; no page JavaScript runs,
  so post-hydration DOMs are out of scope. The two SPA entries are server-rendered apps
  chosen for that reason; a client-only rendered app is not covered by any claim here.
- **No layout is computed**, so visual regressions — overflow, clipping, reflow — are invisible.
- **The e-commerce pair is a convenience sample.** Amazon, eBay, Waterstones, Decathlon and
  Patagonia all answered a scripted fetch with a bot challenge; the two shops here are the
  ones that served their product page.
- **n=12 pages, one page per (site, category).** Every rate here is a base rate over this
  sample, not an estimate with an interval.
- **`wall_ms` in the artifact is not a latency figure** — the backend caches on
  `(model, text)`; §2c owns latency.
- **Sent-side selection counts are null for the document cut by construction**: it resolves a
  section against its first _unit object_, not an element, so no element-keyed "sent" status
  exists for it. The skipped side is element-keyed and measured in both cuts.
- **The per-sentence guard rate is not derivable** from the response shape:
  `fallback_reason` names the first guarded sentence in a request, not a breakdown, so
  `requests_with_a_guarded_sentence` (1,007 of 2,224 sent, 45%) counts requests, not
  sentences. Reading it as a rejection rate is wrong — 435 of SEP's 480 requests carry a
  reason and 405 of them still changed.

## 3. Consistency check

The ASSET zero-shot `facebook/bart-base` baseline is identical across S3 and D1
(SARI 21.34 / BLEU 89.89 / FKGL 10.02), as it must be — same model, same benchmark, no
sampling. This is the one cross-run internal consistency check the current numbers
support, and it passes.

⚠️ **Qualified 2026-08-23: it passes as an internal check and fails as an external one.** S3
and D1 agree with each other because they come from the same 2026-08-10/11 session — and that
session's BLEU is the figure S4, S5 and S6 all contradict. Re-measured, this baseline's ASSET
BLEU is **91.64**, not 89.89. So the check confirms the two runs were consistent _with each
other_, which is all a shared-session check can confirm; it does not validate the value. The
SARI and FKGL halves do reproduce (21.39 / 10.02), and those are the parts to lean on. The
practical consequence was for **D1**, whose fine-tuned BLEU of **50.83 had never been
re-measured** and inherited the same doubt. **Resolved 2026-08-26 by retirement rather than by
re-measurement** (§4 item 17): D1's BLEU pair is withdrawn, the BLEU-reversal argument it used
to carry now runs on S6 and D2, and nothing in §6.2 cites either figure. What this consistency
check still confirms is the SARI/FKGL half, which does reproduce.

✅ **A second check, and a much stronger one, exists as of 2026-08-23: D2′ ≡ D4′.** Two runs
on different machines — an RTX A5000 on cuda (208 s / 484 s of generation) and an M3 on mps
(1429 s / 2340 s) — with two copies of the weights, the Hub revision and the local checkpoint
directory, produced **byte-identical generations** for both seq2seq systems (sha256 of the
joined predictions matches) and therefore identical D-SARI, SARI, BLEU, FKGL and BERTScore to
every digit recorded (31.61664632257971 / 14.625060142908666 D-SARI). This is the project's
first end-to-end cross-device reproduction of a document row, and unlike the S3/D1 check it
compares two _sessions_ rather than two rows of one.

D2's baseline is the same _model_ but a different _benchmark_, so its numbers are not
expected to match D1's and cannot be used as a second consistency check. That the two
disagree is itself the clearest available demonstration that benchmark choice, not model
quality, drives absolute metric values here — SARI 21.52 vs 21.34 is coincidental proximity,
and the BLEU gap is an order of magnitude wide in either version of the ASSET figure (18.09
against D1's retired 89.89, or against S6's re-measured **91.64** for the same model on the
same benchmark, which is the one to quote).

---

## 4. What has to happen before §6.2 can be finalised

0. ~~**Copy the 2026-08-21/22 artifacts off beet.**~~ — **done 2026-08-23.** They had never
   been copied back _and_ were cited here by the wrong path (the pipeline writes under
   `results/remote/`). Recovered, paths corrected, and every figure re-verified against its
   artifact — all match to the decimal. Two residual notes: **S6 is
   `remote/sentence_eval/summary.json`**, not the top-level `sentence_eval/summary.json` (that
   one is S5); and `remote/logs/*.log` is gitignored, so the stage logs exist locally but not
   in a fresh clone. `remote/RESULTS_GPU.md` came back with them and is the authority on which
   artifact produced which 2026-08-21/22 row.
1. ~~**Run D2**~~ — done 2026-08-16, artifact committed under
   `research/results/document_eval/`, alongside the cached generations and the partial
   checkpoints the run resumed from. One provenance gap to note when citing it: the JSON's
   `git_commit` is `null`, because the script ran in Colab from an uploaded copy rather
   than a git checkout. The model id, weights revision, decoding parameters, n and seed are
   all recorded, so the run is reproducible; the _code_ version is pinned only by
   `script_version: "1.1"`. Pushing branch `2.0` and cloning in Colab instead of uploading
   would close that gap for the full-scope run.
2. ~~**Re-run S3 and save the output.**~~ — done 2026-08-17 as **S4** (beam, matching S3's
   decoding) and **S5** (the checkpoint default, matching what the extension serves; recorded
   as "greedy" at the time, corrected 2026-08-24 — see the decoding legend in §1), both with artifacts
   under `research/results/sentence_eval/`. Doing it required three fixes to
   `evaluate_sentence.py`, each of which had been silently producing a different
   measurement than the one this table described: it had **no zero-shot `facebook/bart-base`
   condition** at all (its `base` condition is new; the pre-existing `online` condition is
   the _other_ off-the-shelf baseline), its seq2seq decoding **passed no arguments while this
   table said beam search** (recorded then as "greedy"; it was in fact the checkpoint's own
   4-beam default, which is a second, subtler version of the same mislabelling and was not
   caught until 2026-08-24), and its significance test was **hardcoded against `online`** rather
   than the baseline S3 used. **Two decisions remain open**, both flagged inline in §1:
   whether to formally retire S3's BLEU pair, and how to rewrite the BLEU explanation now
   that the reference-amplification mechanism it asserts is contradicted by the k-sweep.
3. ~~**Decide how to present S1/S2.**~~ — **decided 2026-08-24: keep them, never quote them
   as a result.** They will not be re-run, because a rerun would not recover _these_ numbers:
   the harness they used had the single-reference-truncation bug fixed on 2026-08-10, and the
   50-sentence single-reference slice they scored is superseded by S4–S6 on ASSET's 359
   sentences with ten references. What they remain the only evidence of is the **unseeded
   run-to-run noise floor (~2 SARI points)**, which §1 uses to argue that smaller differences
   are not findings. So they are cited as a **variance observation**, described as coming from
   a session that predates the provenance harness and is not reproducible; every performance
   claim cites S4, S5 or S6. Thesis §6.2 states this in one sentence.
4. ~~**Cohesion metrics (§2.5/4.7)** were promised in Ch. 2 and never implemented. Cut
   them or move them to Future Work.~~ — **decided 2026-08-17: moved to Future Work**
   (thesis §7.6), together with formal human evaluation (§7.2). Neither is a pending
   measurement any more, so no row will ever be added here for them, and Ch. 6 carries no
   results subsection for either. What stands in for human evaluation is structured
   manual inspection by the author, of which the reportable instance is the qualitative
   review below.
5. ~~**S3's BLEU pair.**~~ — **settled 2026-08-21/22 by S6**, which reproduces S4's
   88.30 / 91.64 on different hardware. S3's 67.26 / 89.89 is the outlier; retire it.
6. ~~**Full-scope document training and the complete test split.**~~ — done 2026-08-21 as
   **D5** (131,739 docs / 5 epochs, n=8000).
7. ~~**Isolate the metric-normaliser confound.**~~ — done 2026-08-21 as **D2′**, and
   ~~**re-score D4 rather than caveat it**~~ — **done 2026-08-23 as D4′** (CPU rescore of
   D4's cached generations, 14 min, no GPU, no regeneration: `--rescore` reused all three
   conditions from `document_eval/generations_n500_seed42.json`). It changed the conclusion
   this item had recorded. The ≈1.3/≈1.4 correction was **wrong as a propagated constant** —
   it was the D2 → D2′ delta, and D2 applied _no_ metric normalisation at all, so that pair
   conflates adding normalisation with fixing it. Measured on the clean pair: baseline
   **+0.15**, fine-tuned **+0.46**, prompted 7B **+0.13**, and D4's margin **16.68 → 16.99**,
   i.e. D4 _understated_ its own margin. Quote **D4′**; do not correct D4 by a constant.
   Two by-products worth keeping: SARI and FKGL are unchanged to 15 significant figures
   (`easse` re-tokenises, so the defect was invisible to them by construction), and D4′
   reproduces **D2′ to every recorded digit** off byte-identical generations produced on
   different hardware.
8. ~~**Compute LENS.**~~ — done 2026-08-22, and it is the item that opened new work rather
   than closing it: **LENS ranks the prompted LLM 23.8 points above the fine-tuned model
   that D-SARI ranks first.** §6.2 cannot report a single ordering for RQ1 without saying
   which metric it privileges and why.
9. ~~**RQ1's answer depends on the baseline and the metric.**~~ — **framed 2026-08-24; the
   measurements were never the missing part.** All four comparisons stand as measured: +16.42
   SARI against the zero-shot starting point, +0.37 against a strong off-the-shelf baseline
   with overlapping CIs, −8 to −9 against prompted LLMs at sentence level, and +12.54 D-SARI
   against the prompted 7B at document level while losing it by 23.80 LENS. §6.5 now leads
   with **one sentence** — fine-tuning reliably beats its own starting point, roughly matches
   a model three times its size, and loses to prompted models on the measures closest to human
   judgement — and names the two contributions that follow from it: **efficiency** (a 139M
   checkpoint matching a 406M one, which is what makes the browser deployment possible) and
   **the metric disagreement itself**. What stays open is only the tie-break, which is item 16.
10. ~~**Three seeds for the prompted document condition.**~~ — **done 2026-08-24 as D6′.**
    D6's LLM row was one draw from a sampling system, which mattered because **LENS 69.27** is
    the project's most surprising number. Seeds 2 and 3 at n=2000 give **LENS 69.22 ± 0.071**
    and **D-SARI 22.39 ± 0.039**, and the two gaps that define §6.2's problem are stable to the
    second decimal (D-SARI 12.54 ± 0.039, LENS 23.80 ± 0.071). **The metric disagreement is not
    a sampling artefact.** Artifacts under `remote/document_eval_llmseed{2,3}/`; the merged
    generations caches stayed on beet, since their seq2seq halves duplicate D6's — what is
    committed is each run's scores, provenance and the LLM text that produced them
    (`partial_llm_n2000_seed42.json`). One directory per seed, for the reason item 15 gives.
11. ~~**Verify the D-SARI port.**~~ — done 2026-08-23 as **§2f**. The thesis's
    "verified bit-identical against the reference implementation's own bundled test cases"
    was unreproducible (upstream ships no test cases; `research/` had no tests at all). It is
    now a differential test against upstream that runs on demand:
    `remote/checks/test_d_sari.py`, **bit-identical across 66 real-document cases**, 10 checks,
    0 failed. §2.6's wording has to change from "bundled test cases" to what actually happened.
12. ~~**Resolve the corpus-cleaning discrepancy.**~~ — measured 2026-08-19, documented
    2026-08-23 as **§2d**. Two mojibake filters; the data statement quotes **strict**
    (117,656 train rows) and the shipped sentence checkpoint trained under **targeted**
    (**121,657**). The filters are nested and the targeted one _under_-drops, so the model
    trained on the dirtier corpus. **Editorial fix, no re-training** — but §2.3/§4.1 must
    stop pairing 117,656 with S3–S6.
13. ~~**Document the WikiLarge/ASSET overlap.**~~ — measured 2026-08-19, documented
    2026-08-23 as **§2e**. Three findings: **WikiLarge test ⊂ ASSET test (121/121, 100%)**, so
    S1/S2 and S3–S6 are nested rather than independent evaluations; **17.88% of WikiLarge
    validation is in train**, so the reported validation losses (0.4568 / ~0.538) are
    optimistically biased and are training diagnostics only; **ASSET test ↔ train is 1.39%**
    (five sentences under the strict filter; **2.23%, eight sentences, under the filter that
    actually trained** — cite that one), stated rather than waved away. Few-shot demonstrations
    are clean. The held-out-remainder delta that this item deferred is now measured — see 14.
14. ~~**The held-out-remainder delta for ASSET.**~~ — **done 2026-08-23**, in two parts.
    ~~Re-run `check_overlap.py --filter targeted`~~: artifact `remote/overlap_audit_targeted.json`;
    every overlap figure rose, no conclusion moved, few-shot demonstrations clean under both
    filters. ~~Measure the delta~~: `evaluate_sentence.py` gained `--exclude-indices`
    (and `--cache-dir`), and the 351-sentence remainder was scored from S6's cached generations
    on CPU. **Every system's SARI moves by less than 0.12 and the RQ1 margin by 0.006** — with
    the un-fine-tuned baseline, which never saw WikiLarge, losing _more_ than the fine-tuned
    checkpoint, which is the control that turns "immaterial" into a measurement. §2e carries the
    table; §6.6 may now state the number instead of the assurance.
15. ~~**Fix the `--llm-seed` cache collision.**~~ — **done 2026-08-23**, and it was blocking
    item 10. `evaluate_document.py` keyed the prompted condition's generation cache on the
    _sampling_ seed, so a second `--llm-seed` resumed from the first one's finished
    generations and logged "generation already complete" — three seeds would have produced
    three identical rows with no warning. The cache filename now carries `llm_seed`.
    **Consequence for the recovered artifacts, corrected 2026-08-23 after actually doing it:**
    the existing `partial_llm_n{500,2000}_seed42.json` caches were written under the old name
    and a `--llm-seed 1` _generation_ run will no longer find them (rename them to
    `…_seed42_llmseed1.json` if that matters). It does not affect `--rescore`, which reads the
    **merged** `generations_n{n}_seed{seed}.json` — that file already contains
    `llm_predictions`, which is why D4′ re-scored all three conditions on CPU without touching
    a partial or a model.

✅ **The same defect existed one level up, and is fixed as of 2026-08-24 — by a guard
rather than by a rename.** The merged cache's filename carries the _sampling_ seed but not
the LLM seed, while its contents depend on both, so a non-`--rescore` run with a second
`--llm-seed` in a directory holding seed 1's cache would **overwrite seed 1's generations
under a name recording no difference** — the same class of bug, with data loss instead of a
silent duplicate.

Renaming the cache was the obvious fix and is the wrong one: it would orphan the four
caches this file cites by path, and it would stop a _no-LLM_ rescore from reusing the
seq2seq half of an LLM-bearing cache, which is work worth reusing. Instead the cache now
**records what produced it** (`llm_model`, `llm_seed`) and `check_llm_cache_identity()`
refuses any run whose LLM identity differs from what is recorded — before the corpus is
loaded, before the Ollama preflight, and before any model is loaded. The error names the
per-seed `--outdir` to use, so the convention item 10 follows is now enforced instead of
remembered. A rescore that passes no `--llm` at all is still allowed to score the cached
LLM row, and now attributes it from the recorded identity rather than reporting a score
with no model behind it.

**The four existing caches were backfilled**, which is what actually closes the hole for
D3, D4/D4′, D6 and the smoke run — an unmarked cache is allowed through with a warning, so
without the backfill the trap could still have sprung in D6's own directory. The values
were not typed in: each was taken from the `step4b_*.json` artifacts sitting beside the
cache, accepting only same-`n` siblings that agree unanimously (all four: the 7B tag,
seed 1). Re-verify with:

```sh
python3 -c "import json,glob;from pathlib import Path;[print(c, json.load(open(c)).get('llm_model'), json.load(open(c)).get('llm_seed'), sorted({(json.load(open(s))['models']['llm']['model_id'], json.load(open(s))['models']['llm']['seed']) for s in glob.glob(str(Path(c).parent/'step4b_*.json')) if json.load(open(s)).get('models',{}).get('llm')})) for c in sorted(glob.glob('results/**/generations_n*_seed*.json', recursive=True)) if json.load(open(c)).get('llm_predictions')]"
```

Verified end to end on beet at n=20 with the GPU and Ollama running: seed 1 generates and
records its identity; seed 2 into the same directory is **refused before generating**;
seed 2 into its own directory generates normally, and its text differs from seed 1's while
both seq2seq conditions stay byte-identical. The two seed directories from item 10 hold
caches written before the guard existed and are unmarked; they are one run per directory,
so nothing can collide there.

16. **Open, and it is now the only open measurement: the D-SARI/LENS tie-break needs people.**
    Everything mechanical has been eliminated — truncation, sample size, the normaliser,
    sampling (D6′) and an implementation bug in D-SARI (§2f). The disagreement is real, and no
    automatic metric in this project can adjudicate it. Formal human evaluation was scoped out
    to Future Work on 2026-08-17 and stays there (thesis §7.2); what changed is that the study
    now has a specific and much cheaper question — ~30–50 documents, two systems, three raters,
    meaning preservation / simplicity / fluency, inter-annotator agreement — rather than an
    open-ended accessibility evaluation. Until it is run, **§6.5 states RQ1's document-level
    ordering as metric-dependent and unresolved**, which is a finding rather than a gap.

17. ~~**D1's ASSET BLEU pair.**~~ — **retired 2026-08-26, not re-measured.** Item 2 left this
    dangling: the 2026-08-10/11 session's BLEU is contradicted by S4, S5 and S6, the baseline
    half is re-measured at **91.64** against D1's 89.89, and the fine-tuned half (**50.83**) had
    no artifact and no second measurement. Re-running D1 was the other option and is the wrong
    one — it would put a _document_ checkpoint back on the _wrong benchmark_ purely to rescue a
    number, which is scope expansion in the shape of housekeeping. Both figures are struck
    through in the D1 row and quoted nowhere. **What the retirement costs is nothing**, because
    the argument they carried is better served by rows that have artifacts: the ASSET half is
    **S6**, where BLEU runs perfectly inverse to SARI across all five conditions with the
    93.9%-unchanged baseline on top, and the D-Wikipedia half is **D2**, where the fine-tuned
    model wins BLEU 22.25 vs 18.09. Same claim — BLEU pays for copying, and the benchmark
    decides who is copying — on two reproducible rows instead of one unbacked pair. D1's SARI
    and FKGL stand.

18. **Item 8 of the research-completion list: the thesis draft (2026-08-26).** The draft was
    audited against this ledger and the claims the evidence had overtaken were rewritten in
    place, dated, rather than deleted: §1.4's gap list (which still named the full-scope
    document result, LENS and the multi-condition ASSET comparison as gaps — all three have
    landed), §1.4's second copy of the "bundled test cases" claim §2f retired, the four TODOs
    conditioned on runs that have since completed, §4.8's "neither file is committed to git"
    and the notebook's superseded filename, §4.9's abandoned four-condition run (superseded by
    S6, not resumable), §6.6's five-gaps bullet (four closed), and Ch. 8's conclusion numbers,
    which still quoted retired rows S3 (+16.5 SARI) and D2 (+18.4 D-SARI) instead of S6 and
    **D5**. Item 17 was applied in the same pass. No measurement was run and no scope added.

## 5. Regenerating these numbers

```sh
cd research

# S4/S5 -- RQ1's "does fine-tuning help?" comparison against the un-fine-tuned
# checkpoint. --conditions and --significance-baseline both matter: the default
# baseline is `online` (the shipped bart-large), which is a DIFFERENT claim.
python evaluate_sentence.py --conditions base,local --decoding beam \
    --significance-baseline base --outdir results/sentence_eval     # -> S4
python evaluate_sentence.py --conditions base,local --decoding checkpoint_default \
    --significance-baseline base --outdir results/sentence_eval     # -> S5
                                        # (this preset was called `greedy` when S5 ran)

python evaluate_sentence.py --conditions base,local --decoding true_greedy \
    --device cpu --seeds 1 --outdir results/sentence_eval           # -> S7
# CAUTION: every run writes `<outdir>/summary.json`, so re-running any of the above into
# results/sentence_eval/ silently overwrites the summary of whichever run wrote it last
# (S7's run did exactly this to S5's, restored from git 2026-08-24). The timestamped
# step4_*.json artifacts are safe -- they are the citable ones -- and so are the per-run
# generation caches, which are keyed by condition, seed and decoding. Only summary.json
# collides. Pass a fresh --outdir for a new run, or expect to restore it.

python evaluate_document.py --outdir results/document_eval          # -> D2

# Q1 -- qualitative review of D2's cached generations. Reads only; no model, no GPU.
python review_document_outputs.py                                   # -> Q1

# S6 / D5 / D6 / D2' -- the one-machine session. Needs a reserved GPU and the
# toolkit's two virtualenvs (remote/setup.sh, then --lens for the second).
source remote/profiles/beet.env
python remote/pipeline.py --only sentence                           # -> S6
python remote/pipeline.py --only eval_doc_full                      # -> D5
python remote/pipeline.py --only eval_doc                           # -> D6
python remote/pipeline.py --only lens                               # -> the LENS column

# D2' / D4' -- the normaliser isolation. D2' regenerated on beet; D4' is a pure CPU rescore
# of D4's cached generations (all three conditions), which is the single-variable pair.
# --llm is still required for D4' because the LLM condition is scored: the tag is preflighted
# even under --rescore, so `ollama serve` must be up, but nothing is generated.
python evaluate_document.py --rescore --limit 500 --seed 42 \
    --model scratch/simplification_results_document/best_checkpoint \
    --baseline facebook/bart-base --llm qwen2.5:7b-instruct-q4_K_M --skip-lens \
    --outdir results/document_eval                                  # -> D4'

# Item 10's seeds for the prompted document condition. One outdir per seed: the merged
# generations cache is not keyed by --llm-seed, and since 2026-08-24 the script refuses to
# generate over a cache belonging to a different LLM run rather than silently replacing it
# (item 15). Copy the two seq2seq partials in first and they resume instead of regenerating
# (~70 min per seed instead of ~127).
for S in 2 3; do
  mkdir -p results/remote/document_eval_llmseed$S
  cp results/remote/document_eval/partial_{finetuned,baseline}_n2000_seed42.json \
     results/remote/document_eval_llmseed$S/
  python evaluate_document.py --model scratch/simplification_results_document/best_checkpoint \
      --baseline facebook/bart-base --limit 2000 --seed 42 --device cuda --batch-size 16 \
      --llm qwen2.5:7b-instruct-q4_K_M --llm-seed $S --resamples 1000 --skip-lens \
      --outdir results/remote/document_eval_llmseed$S
  ../venv-lens/bin/python remote/lens_only.py \
      --generations results/remote/document_eval_llmseed$S \
      --outdir results/remote/document_eval_llmseed$S
done

# 2g -- the RQ2 site-sample audit. No GPU. Needs the backend up (without --reload) with
# both seq2seq checkpoints loaded, and network access on the first run to fetch the twelve
# pages into scratch/sites/ (they are cached with fetch time and sha256 afterwards, so a
# later run audits the same bytes -- re-fetch deliberately with --refetch yes).
#
# ~40 min on CPU for all 24 rows. The backend's cache is CACHE_MAX=1024 against ~2,500
# requests here, so a second pass is mostly cold again, not a cache replay.
node audit_sites.mjs                                    # -> 2g's three artifacts
node summarise_site_audit.mjs                           # -> every table in 2g, from the artifact
node sample_for_labelling.mjs results/site_audit/site_audit_<stamp>   # -> the labelling sheet
# The two follow-ups 2g's findings were read from, both taking --site/--cut:
node inspect_emptied_labels.mjs --site wikipedia --cut sentence   # emptied labels, as markup
node inspect_revert_diff.mjs --site allbirds --cut sentence       # what a revert leaves behind

# Provenance audits (§2d/§2e/§2f). None needs a GPU; the D-SARI test needs network access
# for the upstream download, or --offline to run invariants only.
python remote/dataio.py --report                        # 2d: corpus-filter divergence
python remote/checks/check_overlap.py                   # 2e: benchmark overlap -> overlap_audit.json
python remote/checks/check_overlap.py --filter targeted  # 2e: the corpus that trained
                                                        #     -> overlap_audit_targeted.json

# 2e: the held-out-remainder delta. Both read S6's cached generations and generate nothing;
# the control exists so the delta is measured within one environment rather than against a
# figure produced on other hardware. ~47 min each on CPU, almost all of it the bootstrap.
python evaluate_sentence.py --conditions base,online,local,llm_7b,llm_3b --decoding beam \
    --seeds 3 --significance-baseline base \
    --cache-dir results/remote/sentence_eval --outdir results/asset_heldout   # n=359 control
python evaluate_sentence.py --conditions base,online,local,llm_7b,llm_3b --decoding beam \
    --seeds 3 --significance-baseline base \
    --cache-dir results/remote/sentence_eval --outdir results/asset_heldout \
    --exclude-indices 60,135,137,141,144,211,228,357 \
    --exclusion-note "..."                                                    # n=351 remainder
python remote/checks/test_d_sari.py                     # 2f: D-SARI vs upstream
python evaluate_document.py --model yunvs/bart-base-dwikipedia-simplification \
    --limit 500 --seed 42 --device cuda --skip-lens \
    --outdir results/remote/normaliser_isolation                    # -> D2'
```

LENS needs the second virtualenv and a hard `transformers` pin — `lens-metric` asks only for
`transformers>=4.8`, resolves whatever is current, and its RoBERTa encoder then unpacks a
return signature that no longer exists. `remote/requirements-lens.txt` pins 4.30.2.

Run them from `research/`, not from `backend/` — both resolve `scratch/` and
`results/` relative to the working directory. `evaluate_document.py` is the slow one;
`notebooks/colab_dwikipedia_eval.ipynb` runs it on a GPU instead.

**Do not run `evaluate_sentence.py` bare and read the result as S3.** Two defaults will
quietly give you a different measurement: `--outdir` writes into gitignored
`scratch/eval_methods`, and `--significance-baseline` defaults to `online`
(`eilamc14/bart-large-text-simplification`), not the zero-shot `facebook/bart-base` that
S3 was measured against. Until 2026-08-17 the script had no `base` condition at all, and
its seq2seq decoding passed no arguments while this table described every row as beam
search — which is why S3 could not be reproduced from it and why S4/S5 exist as separate
rows. That un-argumented path was itself recorded as "greedy" until 2026-08-24, when it
turned out to inherit the checkpoint's 4 beams; S7 is the genuinely greedy row.

Both scripts write timestamped JSON carrying the model ids, resolved checkpoint paths,
weights mtime/size, git commit, device, decoding parameters, sample size and seed — so a
row in this table can cite a file instead of a memory. That provenance block is the whole
reason these exist as scripts rather than notebook cells.
