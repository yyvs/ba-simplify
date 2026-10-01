#!/usr/bin/env bash
# Download everything a compute job will need, from the ONE node that can reach the internet.
#
#     ssh <Kennung>@storage.hpc.rz.uni-duesseldorf.de
#     cd /gpfs/project/<Kennung>/ba-simplify/research
#     source remote/profiles/hpc.env
#     bash remote/hpc/prefetch.sh
#
# Per the HHU hpc-tutorial, only `storage.hpc.rz.uni-duesseldorf.de` "allows connections for
# loading huggingface models"; compute nodes do not. from_pretrained() on an uncached model
# there retries until walltime runs out. Everything is fetched once into HF_HOME on scratch.
#
# Idempotent; rerun after adding a model or metric.
#
# Also useful on beet (source profiles/beet.env instead): the ~15 GB download does not need
# the GPU reservation.

set -euo pipefail

if [[ ! -f evaluate_document.py ]]; then
	echo "Run from research/ (evaluate_document.py should be in the current directory)." >&2
	exit 1
fi
if [[ -z ${HF_HOME:-} ]]; then
	echo "HF_HOME is unset -- source remote/profiles/hpc.env first." >&2
	exit 1
fi

# the profile sets offline mode for jobs
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

echo "prefetching into HF_HOME=${HF_HOME}"
echo

python - <<'PY'
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = [
    ("facebook/bart-base", "the starting checkpoint, and the zero-shot RQ1 baseline"),
    ("yunvs/bart-base-wikilarge-simplification", "this project's sentence checkpoint"),
    ("yunvs/bart-base-dwikipedia-simplification-full", "this project's document checkpoint"),
    ("eilamc14/bart-large-text-simplification", "the off-the-shelf comparison baseline"),
]
for model_id, why in MODELS:
    print(f"  {model_id}  -- {why}")
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForSeq2SeqLM.from_pretrained(model_id)

# BERTScore's default English model, picked internally by bert-score (never named in code),
# so easy to miss until a job fails at scoring.
print("  roberta-large  -- BERTScore's suggested default for English")
from transformers import AutoModel
AutoTokenizer.from_pretrained("roberta-large")
AutoModel.from_pretrained("roberta-large")
PY

echo
echo "datasets"
python - <<'PY'
from datasets import load_dataset

# ASSET: the sentence-level benchmark (359 test / 2000 validation sentences, 10 references).
for split in ("test", "validation"):
    ds = load_dataset("facebook/asset", "simplification", split=split)
    print(f"  facebook/asset {split}: {len(ds)} rows")

# WikiLarge: the training corpus. Scope slices are views over the same cached download.
ds = load_dataset("eilamc14/wikilarge-clean")
print(f"  eilamc14/wikilarge-clean: { {k: len(v) for k, v in ds.items()} }")
PY

echo
echo "D-Wikipedia (plain files from GitHub, not the Hub -- cached under scratch/)"
python - <<'PY'
import sys
sys.path.insert(0, "remote")
from dataio import load_d_wikipedia

# scope="full" downloads and extracts every split, including the .7z train archive
data = load_d_wikipedia(scope="full")
print(f"  splits: { {k: len(v['source']) for k, v in data.items() if k != 'stats'} }")
PY

echo
echo "NLTK punkt (D-SARI's sentence-count penalty needs a sentence splitter)"
python - <<'PY'
import nltk
for package in ("punkt", "punkt_tab"):
    nltk.download(package, quiet=True)
print("  done")
PY

echo
echo "the D-SARI reference implementation, for the differential test"
python remote/checks/test_d_sari.py --n 20 ||
	echo "  (test reported a problem -- read it now, on a node that can still download)"

cat <<'EOF'

== prefetch complete ==

The compute nodes can now run offline. Submit with:
  qsub remote/hpc/train_document_full.pbs

Two things worth knowing before you do:

  - The prompted-LLM conditions need an Ollama server, which means a *userspace* install and
    a model pull -- also only possible from this node. If you are running the LLM conditions
    on the cluster rather than on beet, do that now too (see setup.sh's closing notes), with
    OLLAMA_MODELS pointing at scratch.
  - Files on /gpfs/scratch older than 60 days are deleted automatically. This cache is on
    scratch by design (it is large and regenerable), but do not leave a checkpoint there as
    its only copy.
EOF
