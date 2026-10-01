# Running this thesis's experiments on the HHU GPUs

Everything in this directory exists to answer one question well: **given a real GPU and no
session limit, what should actually be run, in what order, and what does each run buy the
thesis?**

Read the two sections below before touching anything. The plan matters more than the scripts.

---

## 1. What a GPU actually unlocks — and what it does not

The honest framing first, because it changes how you should spend the reservation.

**Most of what is missing from this thesis is not compute-bound.** The largest evidence gap is
RQ2's site-sample study — filter false-positive/false-negative rates over real web pages — and
it needs a browser, ten pages and an afternoon. No GPU hour helps with it. The same is true of
the writing, which is the binding constraint on submission.

So the GPU is not the thing that finishes the thesis. It is the thing that closes a specific,
well-defined set of gaps that have been blocked on compute for weeks. There are five, and they
are worth stating precisely:

| Gap                                | Why it has been stuck                                                                                                                                                               | What the GPU changes                                                                                                                                               |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **The full-scope document run**    | Colab's hard 5h20m session cap against an ~9h ETA. §4.5 records the 2-epoch cap as "an experimental-design constraint imposed by the compute environment, not a modeling decision." | No session cap. 5 epochs, early stopping decides when to stop, and the cap stops being a caveat.                                                                   |
| **The four-condition ASSET table** | Abandoned after one condition: ~3 h projected at 3.4 tok/s under machine load.                                                                                                      | Seq2seq conditions are seconds; the prompted ones are minutes. RQ1's "compared with an off-the-shelf baseline" finally gets a _strong_ baseline in the same table. |
| **LENS on any row**                | `lens-metric` pins torch/transformers versions that break the backend if installed alongside it. §6.6: "not a metric this project can report routinely."                            | A second virtualenv. That is the entire fix, and it has been a one-line problem for weeks.                                                                         |
| **Sampled document evaluation**    | 500 documents on an M3 took 2h23m for three systems. The full 8,000-document split was never affordable.                                                                            | The two-system RQ1 comparison runs on the complete split, retiring §6.6's "seeded sample, not the full split" caveat outright.                                     |
| **Natural-case document training** | A whole second training run, never affordable. §7.1 calls it "the highest-value follow-up identified this session."                                                                 | Affordable. It removes `document_text.py`'s normalize/de-normalize layer from the deployed path — a genuinely new contribution, not a finished measurement.        |

And one more that is not about any single number:

> **Every result in this thesis currently comes from a different machine.** An Apple Silicon
> Mac on MPS, a Colab T4, CPU inference on an M3, at different batch sizes and precisions,
> through two different corpus filters. A large part of §6.6 exists only to caveat that mixing.
> One session on one host with one commit retires most of those caveats by regeneration rather
> than by argument.

That is the real prize, and it is why every script here stamps its output with the host, the
driver, the commit and the seed.

### What this does _not_ fix

Stated plainly so it does not close by implication:

- **RQ2's site-sample study.** Still the largest gap. Needs no GPU.
- **Human evaluation** — scoped out to Future Work (§7.2). Not pending.
- **Cohesion measurement** — scoped out to Future Work (§7.6). If you want it anyway, the
  paired 500-document outputs already on disk are enough to compute TAACO-style indices with
  no generation and no GPU at all.
- **The writing.** Twelve chapters of prose do not come out of a card.

---

## 2. Two machines, and they are not interchangeable

|               | **cl-hhu boxes** (`beet`, `aker`, …)         | **HHU HPC cluster**                                                    |
| ------------- | -------------------------------------------- | ---------------------------------------------------------------------- |
| Access        | Plain SSH; you have `beet` now               | PBSPro batch scheduler; needs an account and a project                 |
| Scheduling    | A shared **calendar**, no queue system       | `qsub`, walltime, real queues                                          |
| Filesystem    | Local disk                                   | `/gpfs/project` (backed up) + `/gpfs/scratch` (**60-day auto-delete**) |
| Internet      | Yes                                          | **Compute nodes: no.** Only `storage.hpc…` can reach the Hub           |
| Python        | System python + your venv                    | `module load Python/3.10.12`, pip via a local mirror                   |
| Session limit | None — you hold the box                      | Whatever walltime you requested, then killed                           |
| Best for      | Interactive work, Ollama, the whole pipeline | Long unattended training, bigger cards (A100)                          |

**Use `beet` for the whole session.** RTX A5000, 24 GB, Ampere. It fits everything here with
room, it runs Ollama without ceremony, and no walltime kills a run mid-epoch. The cluster is
worth it only if the A5000 is booked solid or you want an A100 for the training stage — job
scripts for that case are in `hpc/`.

> **Reserve before you run.** <https://calendar.online/5afa284259e0c92ff6a2>
> There is no queue system on these boxes. Nothing stops two people training at once except
> one of them looking first. `gpucheck.py` prints who else is currently on the card.

### 2.1 Getting in: `uzziel` is the door, `beet` is the room

`uzziel.phil.hhu.de` is the faculty **SSH gateway**, not a compute machine — do not train on
it. `isi-8.phil.hhu.de` has no route from outside the university network, so every connection
to `beet` is two hops: your laptop → `uzziel` → `beet`. You have separate credentials for
each; the gateway account comes from IKM, the `beet` account from cl-hhu.

Let `ssh` do the second hop itself rather than opening a shell on the gateway and typing `ssh
beet` inside it. With `ProxyJump`, `scp`, `rsync`, `git` and VSCode all reach `beet` directly,
and the gateway stays what it is — a turnstile.

Copy `ssh/config.example` into `~/.ssh/config`, fill in the two usernames:

```sshconfig
Host uzziel
    HostName uzziel.phil.hhu.de
    User <your-IKM-username>

Host beet
    HostName isi-8.phil.hhu.de
    User <your-cl-hhu-username>
    ProxyJump uzziel
    ServerAliveInterval 60          # a NAT timeout must not drop a 6-hour run
    ControlMaster auto              # one authentication, many sessions
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Then `ssh beet` is the whole command.

**Set up keys before tomorrow, in this order** — the gateway first, because the second
`ssh-copy-id` travels through it:

```bash
ssh-keygen -t ed25519 -C "hhu"      # if you do not already have a key
ssh-copy-id uzziel                  # gateway password, once
ssh-copy-id beet                    # beet password, once, via the gateway
ssh beet 'hostname; nvidia-smi -L'  # should now print isi-8 and the A5000, no prompts
```

`ControlMaster` matters more than it looks. VSCode Remote SSH opens several connections to a
host, and each one is a fresh password prompt on _both_ hops without connection sharing —
roughly six prompts to open a folder. With it, you authenticate once and every later
connection rides the same socket. Two hops also make VSCode's first connect slow while it
installs its server on `beet`; that is normal and happens once.

> **Credentials belong nowhere in this repository.** Not in a profile, not in a comment, not
> in a commit you later amend — the history keeps it. That applies to both accounts you now
> hold. Keys and `~/.ssh/config` live in your home directory, outside the repo, which is
> exactly why the setup above is worth ten minutes tonight.

Two consequences worth knowing before you rely on them:

- **The backend and Ollama are not reachable from your laptop**, even once you are on `beet`.
  Nothing here needs them to be — the pipeline drives both in-process. If you do want to point
  the extension at a GPU-served backend for a demo, forward the port over the same jump:
  `ssh -L 8000:localhost:8000 beet`, then the extension's `http://localhost:8000` works
  unchanged. Do not bind the backend to `0.0.0.0` on a shared faculty machine.
- **A dropped connection still kills a foreground run.** `ProxyJump` adds a second link that
  can fail, and `ServerAliveInterval` reduces that risk without removing it. `tmux` on `beet`
  is what actually makes the session survivable — §4 starts with it for this reason.

### 2.2 Getting the code onto `beet`, and the results back off

The repository is private, and `ForwardAgent no` means your laptop's key does not reach GitHub
from `beet`. Do not paste a personal access token onto a shared faculty machine, and do not
forward your agent to one — anyone with root there can use a forwarded key for as long as the
socket is open. Instead give `beet` its own read-only key, generated on `beet`, which never
leaves it:

```bash
ssh beet
ssh-keygen -t ed25519 -f ~/.ssh/gh_deploy -N ""
cat ~/.ssh/gh_deploy.pub
# GitHub → the repo → Settings → Deploy keys → Add, read-only, paste
printf 'Host github.com\n  IdentityFile ~/.ssh/gh_deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config
git clone git@github.com:yyvs/ba-simplify.git
```

One repository, read-only, revocable from a web page, and nothing of yours is exposed if the
machine is compromised. `rsync -av --exclude scratch/ ./ beet:ba-simplify/` also works and
skips GitHub entirely, but it does not keep `git_commit()` honest across later edits — every
artifact this session writes stamps the commit it was produced at, and that is worth keeping
real.

**The thesis draft will not be there, and should not be.** It is not in the repository, so
no clone brings it, and nothing in the pipeline reads it. Leave it that way rather than
copying it onto a shared machine: do the thesis edits at home, against the artifacts you
bring back.

Bring the results back the same way you got in:

```bash
# from your laptop, after the session
rsync -av beet:ba-simplify/research/results/remote/ research/results/remote/
rsync -av beet:ba-simplify/research/scratch/simplification_results_document_full/best_checkpoint/ \
          ./best_checkpoint_document_full/
```

The `remote/` artifacts are a few MB of JSON and the thing the thesis actually cites. The
checkpoint is ~535 MB and belongs on the Hub, not in git. Run `cleanup.py` on `beet` before you
log off — it deletes resume state, and refuses to touch weights nothing else preserves.

**Why this stays one repository.** It is tempting to split the training code out, since `beet`
plays no part in the shipped extension. Don't: `research/` imports `backend/` at six sites
(`evaluate_sentence.py:53`, `evaluate_document.py:73`, `evaluate_llm.py:74`,
`remote/denormalize_corpus.py:46`, `remote/preflight.py:65,99`), and the comment at
`evaluate_sentence.py:52` states the invariant those imports exist to hold — the evaluation
sends **byte-identical** prompts and normalisation to what the service actually serves, and
`backend/` never imports back. Splitting forces you to duplicate `prompting.py` and
`document_text.py` across two histories, where they drift silently, and the first thing that
drifts is a prompt constant. That produces the specific failure a thesis cannot recover from:
numbers describing a system nobody ran. The deploy key in §2.2 costs five minutes and removes
the only real reason to split.

---

## 3. Setup

On `beet`, once:

```bash
git clone <your-repo> && cd ba-simplify/research
bash remote/setup.sh --lens          # main venv + the isolated LENS venv
source remote/profiles/beet.env      # HF_HOME, thread caps, venv on PATH
python remote/gpucheck.py            # is the card free, and which card is it
```

`setup.sh` installs torch **first**, from the wheel index matching the host driver's CUDA
version, and only then the pinned requirements. Backwards and pip resolves a torch that does
not match the driver — which you discover when `torch.cuda.is_available()` is `False` after
everything else is already installed.

Ollama, for the prompted-LLM conditions, needs no root:

```bash
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz
mkdir -p ~/.local && tar -xzf /tmp/ollama.tgz -C ~/.local
export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_MODELS="$PWD/scratch/ollama"    # 6+ GB of weights, keep them off $HOME
ollama serve &
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_K_M
```

**Keep Ollama; do not switch to vLLM.** vLLM would be faster, but the thesis describes an
Ollama sidecar as the served stack and pins the tags by digest. On a GPU, speed has stopped
being the constraint — so switching would buy nothing and would make the reported numbers
describe a system the extension does not run.

On the HPC cluster instead, the order is different because compute nodes are offline:

```bash
export HHU_KENNUNG=<your-username>
ssh $HHU_KENNUNG@storage.hpc.rz.uni-duesseldorf.de     # the ONLY node with internet
cd /gpfs/project/$HHU_KENNUNG/ba-simplify/research
source remote/profiles/hpc.env
bash remote/setup.sh --hpc --lens
bash remote/hpc/prefetch.sh                            # every model and corpus, cached
# then, on the login node:
module load hpc-tools && gpus_available
qsub remote/hpc/train_document_full.pbs
```

---

## 4. Run it

```bash
tmux new -s gpu                       # a dropped SSH must not kill a 6-hour run
source remote/profiles/beet.env
python remote/preflight.py           # the correctness gate — read its output
python remote/pipeline.py --list     # the plan, with estimates and current status
python remote/pipeline.py            # run it; resumable, idempotent, logged per stage
```

If a reservation ends mid-run:

```bash
python remote/pipeline.py --from train_doc     # continues; train.py resumes from its checkpoint
```

### The stages, and what each one buys

| Stage                                    | Est.   | What it produces                        | Why it is at this position                                                                                                                                                                                                        |
| ---------------------------------------- | ------ | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `filters`                                | 2 min  | Filter-divergence artifact              | **Gates training.** The tracked data statement describes one mojibake filter; the notebook trained with another; they disagree on ~4,000 rows. Training again without settling this reproduces the ambiguity in a new checkpoint. |
| `overlap`                                | 5 min  | Contamination artifact                  | Gives §6.6 its threats-to-validity numbers. Cheap, and an examiner who knows ASSET's lineage from WikiLarge will ask.                                                                                                             |
| `dsari`                                  | 5 min  | Differential test vs upstream           | **Gates every document number.** D-SARI carries the whole document claim; §2.6 says the port was verified bit-identical, and nothing in the repo could show it. Now it can.                                                       |
| `smoke` + `smoke_eval`                   | 20 min | A prove_loop checkpoint, a 20-doc score | Catches a wrong torch wheel, a missing tokenizer, a full disk or a dead Ollama tag in 20 minutes rather than at hour six. This project has already shipped 16 passing mocked tests over a completely non-functional path (§4.9).  |
| `asset`                                  | 1–2 h  | Four-condition ASSET table, 3 seeds     | Short, so a lost reservation still leaves one finished result. Completes RQ1's baseline comparison.                                                                                                                               |
| `train_doc`                              | 5–8 h  | Full-scope document checkpoint          | The single biggest unfinished research item in the project.                                                                                                                                                                       |
| `eval_doc`                               | 2–4 h  | Three-way document scores, n=2000       | Four times D4's sample, all systems on identical documents. Tests whether D4's ranking and its length-profile reversal survive.                                                                                                   |
| `eval_doc_full`                          | 1–2 h  | Two-system scores, **full 8,000 split** | The primary document RQ1 claim, unsampled. Retires a stated §6.6 limitation outright.                                                                                                                                             |
| `lens`                                   | 1–2 h  | The project's first LENS column         | Runs in `venv-lens`. Reads cached generations; no model, no regeneration.                                                                                                                                                         |
| `review`                                 | 10 min | Error taxonomy on the new outputs       | So §6.3's census-year finding describes the checkpoint the thesis reports. CPU only.                                                                                                                                              |
| `collect`                                | 1 min  | `RESULTS_GPU.md`                        | Every table with its host, driver, commit and artifact.                                                                                                                                                                           |
| `train_doc_natural` + `eval_doc_natural` | 6–10 h | Natural-case checkpoint and its scores  | `--stretch`. A new contribution, not a finished measurement. Only after everything above lands.                                                                                                                                   |

**Roughly 12–18 GPU hours for the essential set, 20–28 with the stretch experiment.** Plan two
reservations rather than one heroic block; every stage is resumable.

> Estimates are extrapolations, not measurements. Basis: the project's own Colab T4 run
> measured 22.08 examples/sec for document mode at batch 16 / 512 tokens, and an A5000 is
> roughly 2× a T4 for bf16 training. Check the first epoch's reported ETA against the table
> and adjust — `gpucheck.py --benchmark` will also tell you if the card is being shared.

### Two evaluations, not one, and why

The prompted LLM generates a ~130-token completion per document at 2–4 s each even with GPU
offload. Over the full 8,000-document split that is 5–9 hours — more than the training it
would be evaluating. The seq2seq systems are ~20× faster per document, so the full split is
cheap for them.

Hence the split: the three-way comparison runs at n=2000, and the two-system RQ1 comparison
runs on the complete split. Both are honest; a single table mixing sample sizes across systems
would not be a comparison at all.

---

## 5. Hardware decisions, and why they differ from the Colab runs

`train.py` is a faithful port of the notebook's `run_fine_tuning()` — same optimiser, same
fixed-length padding, same chunked tokenisation, same step-based cadence, same early stopping.
Four things deliberately differ, all because the target is a 24 GB Ampere card with no session
limit rather than a 16 GB Turing T4 inside a 5h20m window:

- **bf16, not fp16.** The notebook sets `fp16=torch.cuda.is_available()` and notes that T4s
  "predate hardware bf16 support entirely". The A5000 (sm_86) does not — it has real bf16
  throughput, and bf16's wider dynamic range removes loss scaling as a failure mode. Gated on
  `torch.cuda.is_bf16_supported()`, so a Pascal box (`turnip`, `shai`) still gets fp16.
- **TF32 matmuls on.** Free on Ampere, off by default in recent torch.
- **No generation during evaluation.** The notebook runs `predict_with_generate=True`, calling
  `generate()` over the whole validation split at every eval step — while
  `metric_for_best_model="loss"`, so not one of those generations affects checkpoint
  selection. This is the largest single speedup available and it changes nothing about which
  checkpoint wins. `--generate-during-eval` restores the old behaviour.
- **5 epochs for document/full, not 2.** The 2-epoch cap is explicitly a Colab artefact.
  Removing it is most of the point of this session; early stopping now decides.

Batch sizes are chosen from the card, not from a constant — `default_batch_size()` in
`train.py`, and `gpucheck.py` prints its suggestion for whatever card you are on.

### Why the sentence checkpoint is _not_ retrained

It would cost ~1.5 GPU hours and it is the wrong trade. That checkpoint already has
artifact-backed results (RESULTS.md rows S4/S5, ΔSARI CI [+15.38, +17.50]) and is published on
the Hub. Retraining changes the weights the thesis cites and invalidates a finished result. The
sentence branch needs re-_evaluation_ on one device — which the `asset` stage does — not a new
checkpoint.

---

## 6. Shared-machine discipline

- **Reserve the slot.** <https://calendar.online/5afa284259e0c92ff6a2>
- **Look before you start.** `gpucheck.py` lists other users' processes on the card.
- **Cap your threads.** `profiles/beet.env` sets `OMP_NUM_THREADS` to half the cores. Left
  unbounded, torch and OpenMP each spawn one thread per core and a dataloader multiplies that.
  This project has measured what CPU contention does to its own numbers — the sentence request
  path degraded **7.8×** between load ~4 and ~6.4 on one host (§6.4). Being the cause of that
  for someone else is avoidable.
- **Clean up.** The motd asks directly. `python remote/cleanup.py --dry-run`, read it, then
  `--apply`. It never removes `best_checkpoint/`, generations caches, `*.partial.json` or
  training histories — and it refuses to remove any checkpoint whose weights nothing else
  preserves, because §6.4 cites a document checkpoint by step number and deleting it would
  settle that attribution question by destroying the evidence.
- **Caches off `$HOME`.** Two BART checkpoints, `roberta-large` for BERTScore, the LENS
  checkpoint and 6 GB of Ollama weights will fill a quota'd home directory. The profiles point
  `HF_HOME` and `OLLAMA_MODELS` at `scratch/`.
- **On the HPC cluster: `/gpfs/scratch` deletes files older than 60 days.** Never leave a
  checkpoint there as its only copy.

---

## 7. What this changes in the thesis

The point of the session, mapped to sections. Work through this after `collect_results.py`.

| Section    | Current state                                                                                              | After                                                                                                                                                        |
| ---------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| §4.3, §6.1 | Corpus statistics come from `prepare_data.py`; the model trained through a different filter                | State which filter trained the reported checkpoint, and cite `train_*.json`'s own `kept_rows`. `corpus_filter_divergence.json` is the evidence for the note. |
| §4.5       | Document/full capped at 2 epochs by Colab's session limit                                                  | The cap is gone. Report the real epoch count and what early stopping did.                                                                                    |
| §4.7, §2.6 | D-SARI "verified bit-identical" — unverifiable from the repo                                               | Cite `checks/test_d_sari.py`: a differential test against the paper's own implementation, runnable.                                                          |
| §6.2       | Four-condition ASSET table unfinished; document rows at n=500, reduced scope                               | Complete table with a strong baseline and three seeds; document rows at full scope, one on the complete test split.                                          |
| §6.2       | **LENS missing from every row**                                                                            | A LENS column. Descriptive means, not significance-tested, exactly as §2.6 says.                                                                             |
| §6.3       | Error taxonomy describes the reduced-scope checkpoint                                                      | Re-run against the reported checkpoint's own outputs.                                                                                                        |
| §6.4       | Document checkpoint cited as "best_checkpoint, epoch 1.0, checkpoint-2500" — those are not the same object | Cite by step number from the new `train_*.json`.                                                                                                             |
| §6.6       | Caveats for mixed hardware, mixed precision, seeded sampling, LENS absence                                 | Most retired by regeneration. Replace with the ones that remain: contamination, single-rater qualitative work, one-article latency findings.                 |
| §6.6       | No contamination figures anywhere                                                                          | `overlap_audit.json`: five split pairs, with the 17.9% validation overlap and what it does and does not affect.                                              |
| §7.1       | Natural-case retraining proposed as future work                                                            | If the stretch stage runs, it moves to Chapter 4/6 as a result.                                                                                              |
| Ch. 4/6    | No figures                                                                                                 
Then, and this is the part that is easy to skip: **fold `RESULTS_GPU.md` into `RESULTS.md` by
hand.** Deciding which number supersedes which is editorial. RESULTS.md's own history includes
retiring a row whose BLEU pair turned out not to reproduce, and no merge tool would have caught
that.

---

## 8. Files here

| File                                            | Purpose                                                                                              |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `preflight.py`                                  | **Read this first.** Refuses to start while a known defect would corrupt output.                     |
| `pipeline.py`                                   | Staged, resumable orchestrator. `--list`, `--dry-run`, `--from`, `--only`.                           |
| `train.py`                                      | The canonical training entry point. Resumable, device-aware, writes a full manifest.                 |
| `dataio.py`                                     | Canonical corpus loading and cleaning; `--report` quantifies the filter divergence.                  |
| `denormalize_corpus.py`                         | Natural-case conversion for the stretch experiment. Reuses the backend's own de-normaliser.          |
| `lens_only.py`                                  | LENS over cached generations, in the isolated venv.                                                  |
| `collect_results.py`                            | All artifacts → `RESULTS_GPU.md`.                                                                    |
| `gpucheck.py`                                   | Card, driver, bf16, who else is on it, suggested batch sizes.                                        |
| `cleanup.py`                                    | Disk reclamation that refuses to delete irreplaceable weights.                                       |
| `common.py`                                     | Paths, provenance stamping, disk guard, stage state.                                                 |
| `checks/test_d_sari.py`                         | Differential test of the D-SARI port against upstream.                                               |
| `checks/check_overlap.py`                       | Train/eval contamination across all split pairs.                                                     |
| `profiles/beet.env`, `profiles/hpc.env`         | Host environment; source before anything.                                                            |
| `ssh/config.example`                            | Two-hop access via the `uzziel` gateway; copy to `~/.ssh/config`.                                    |
| `hpc/prefetch.sh`                               | Download everything on the one node with internet.                                                   |
| `hpc/*.pbs`                                     | PBSPro job scripts for training and evaluation.                                                      |
| `requirements-gpu.txt`, `requirements-lens.txt` | Pinned metrics, unpinned torch, and why.                                                             |

---

## 9. Troubleshooting

**`torch.cuda.is_available()` is False but `nvidia-smi` works.** The torch wheel does not match
the driver. Reinstall from the index for the CUDA version `nvidia-smi` reports (top right):
`pip install torch --index-url https://download.pytorch.org/whl/cu124`.

**`Failed to initialize NVML: Driver/library version mismatch`.** The documented fix in
`../gpus-at-cl-hhu-main/README.md` is a reboot, which needs an admin — David Arps or Kilian
Evang. Use another box meanwhile.

**Ollama returns HTTP 400 on every request.** Almost certainly `keep_alive` sent as a string:
Ollama parses a string as a Go duration and rejects `"-1"`. This is the bug that made the whole
prompted path dead while 16 mocked tests passed (§4.9). `profiles/beet.env` exports it as a
number.

**A stage says "unresolved input".** The stage that produces its input has not run. Run that
one, then `--from <stage>`.

**Training OOMs.** Lower `--batch-size`. Document mode at 512 tokens fits 32 on 24 GB and 16 on
16 GB; halve it and re-run with `--resume`.

**Host RAM OOM during tokenisation, no traceback, process just dies.** That is the OS OOM
killer, and it is a documented failure mode here: the full document split's transient
Python-list peak reached ~8–9 GB before the chunked tokeniser existed. `train.py` chunks at
2,000 documents; if it still happens, lower `chunk_size` in `tokenize()`.

**A stage fails with `git_dirty: true` in its manifest.** Not a failure, but the artifact's
recorded commit does not identify the code that produced it. Commit before the long stages.
