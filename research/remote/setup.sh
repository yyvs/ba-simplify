#!/usr/bin/env bash
# Bootstrap the GPU environments. Run from research/ on the target machine.
#
#   bash remote/setup.sh                  # main venv (training + evaluation)
#   bash remote/setup.sh --lens           # the separate LENS venv, on top
#   bash remote/setup.sh --lens --lens-python python3.10   # older interpreter for LENS
#   bash remote/setup.sh --cuda cu121     # pick the torch wheel index explicitly
#   bash remote/setup.sh --hpc            # HHU HPC cluster: module load + pip mirror
#
# Two things this handles that a bare `pip install -r` does not, and both bite on these
# machines specifically:
#
#   1. torch is installed FIRST, from an index matched to the host driver, before the pinned
#      requirements file. Get this backwards and pip resolves a torch that does not match the
#      driver, and you find out when torch.cuda.is_available() is False after everything else
#      is already installed.
#   2. The HuggingFace cache is moved off $HOME. Home directories here are small -- the HPC
#      cluster documents a 50 GB soft quota and says outright "don't work here" -- and a
#      couple of BART checkpoints plus roberta-large for BERTScore plus the LENS checkpoint
#      will fill it.

set -euo pipefail

CUDA_TAG=""
WITH_LENS=0
HPC=0
PYTHON_BIN="${PYTHON_BIN:-python3}"
LENS_PYTHON="${LENS_PYTHON:-}"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--lens)
		WITH_LENS=1
		shift
		;;
	--cuda)
		CUDA_TAG="$2"
		shift 2
		;;
	--hpc)
		HPC=1
		shift
		;;
	--python)
		PYTHON_BIN="$2"
		shift 2
		;;
	--lens-python)
		LENS_PYTHON="$2"
		shift 2
		;;
	-h | --help)
		sed -n '2,30p' "$0"
		exit 0
		;;
	*)
		echo "unknown option: $1" >&2
		exit 1
		;;
	esac
done

if [[ ! -f evaluate_document.py ]]; then
	echo "Run this from the research/ directory (evaluate_document.py should be here)." >&2
	exit 1
fi

# HPC cluster: modules and the local pip mirror
if [[ ${HPC} == "1" ]]; then
	echo "== HPC mode: loading modules =="
	module load Python/3.10.12 || echo "  (module load failed -- are you on a login/storage node?)"
	# The cluster reaches PyPI only through its own mirror.
	export PIP_CONFIG_FILE=/software/python/pip.conf
	echo "  PIP_CONFIG_FILE=${PIP_CONFIG_FILE}"
fi

# Python floor: 3.10 is the oldest the pinned stack resolves on (requirements-gpu.txt has a
# numpy marker for 3.10 vs. 3.12+). Checked up front; pip would only fail after downloading torch.
MIN_PY="3.10"
PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")"
if [[ "$(printf '%s\n%s\n' "${MIN_PY}" "${PY_VERSION}" | sort -V | head -1 || true)" != "${MIN_PY}" ]]; then
	cat >&2 <<EOF
${PYTHON_BIN} is Python ${PY_VERSION}, and this environment needs ${MIN_PY} or newer.

The cl-hhu boxes and the HPC cluster's Python/3.10.12 module both satisfy that. If this
interpreter is older (Ubuntu 22.04 also ships a 3.8), point at a newer one:

    bash remote/setup.sh --lens --python /usr/bin/python3.10

Or install one in userspace -- uv is present at /snap/bin/uv on the cl-hhu machines:

    uv python install 3.12
    bash remote/setup.sh --lens --python "\$(uv python find 3.12)"
EOF
	exit 1
fi
echo "  python floor: ${PY_VERSION} >= ${MIN_PY}"

# detect the driver's CUDA version, so the torch wheel matches it
detect_cuda_tag() {
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		echo ""
		return
	fi
	# the header's "CUDA Version" is the driver's maximum runtime; the wheel must not exceed it
	local version
	version="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
	# cu124 even for 12.1-12.3 drivers: minor-version compatibility runs it on any 12.x, and
	# cu121 stops at torch 2.5.1, which transformers 5.2 refuses to `torch.load` from
	# (CVE-2025-32434), breaking --resume. Verified on beet: driver 535.309 (CUDA 12.2) runs
	# torch 2.6.0+cu124, bf16 matmul included.
	case "${version}" in
	12.* | 13.*) echo "cu124" ;;
	11.*) echo "cu118" ;;
	*) echo "" ;;
	esac
}

if [[ -z ${CUDA_TAG} ]]; then
	CUDA_TAG="$(detect_cuda_tag)"
fi

echo "== environment =="
echo "  python:     $(${PYTHON_BIN} --version 2>&1 || true)"
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  gpu:        /'
else
	echo "  gpu:        nvidia-smi not found (CPU-only host?)"
fi
echo "  torch index: ${CUDA_TAG:-default PyPI (no CUDA detected)}"

# HuggingFace cache off $HOME
CACHE_ROOT="${HHU_CACHE_ROOT:-${PWD}/scratch/hf}"
mkdir -p "${CACHE_ROOT}"
echo "  HF cache:   ${CACHE_ROOT}"

# main venv
if [[ ! -d venv-gpu ]]; then
	echo "== creating venv-gpu =="
	"${PYTHON_BIN}" -m venv venv-gpu
fi
# shellcheck disable=SC1091
source venv-gpu/bin/activate
pip install --upgrade pip wheel >/dev/null

echo "== installing torch =="
if [[ -n ${CUDA_TAG} ]]; then
	pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
	echo "  no CUDA detected; installing the default wheel (CPU). Training will be unusably"
	echo "  slow -- check nvidia-smi before going further."
	pip install torch
fi

echo "== installing pinned requirements =="
pip install -r remote/requirements-gpu.txt

echo "== NLTK punkt (D-SARI's sentence-count penalty needs a sentence splitter) =="
python - <<'PY'
import nltk
for package in ("punkt", "punkt_tab"):
    try:
        nltk.download(package, quiet=True)
    except Exception as exc:
        print(f"  {package}: {exc}")
print("  done")
PY

echo "== verifying =="
python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda_build={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB  "
          f"cc={'.'.join(map(str, torch.cuda.get_device_capability(0)))}  "
          f"bf16={torch.cuda.is_bf16_supported()}")
else:
    print("  NO CUDA DEVICE VISIBLE TO TORCH -- see remote/gpucheck.py for what to check")
for module in ("transformers", "datasets", "sacrebleu", "easse", "bert_score", "pysbd", "py7zr"):
    try:
        __import__(module)
        print(f"  ok   {module}")
    except Exception as exc:
        print(f"  FAIL {module}: {exc}")
PY

deactivate

# LENS venv, separate on purpose; see requirements-lens.txt
if [[ ${WITH_LENS} == "1" ]]; then
	echo
	# lens-metric pins pandas<2.0 (wheels up to cp311); on 3.12+ pip builds the 1.5.3 sdist,
	# whose setup.py needs pkg_resources (removed in setuptools 81). The two venvs may use
	# different Pythons: lens_only.py only reads generations from disk.
	if [[ -z ${LENS_PYTHON} ]]; then
		LENS_PYTHON="${PYTHON_BIN}"
		if [[ "$(printf '%s\n%s\n' "3.12" "${PY_VERSION}" | sort -V | head -1 || true)" == "3.12" ]]; then
			for candidate in python3.11 python3.10; do
				if command -v "${candidate}" >/dev/null 2>&1; then
					LENS_PYTHON="$(command -v "${candidate}")"
					echo "  lens venv: ${candidate} rather than ${PY_VERSION} (pandas<2 has no cp312 wheel)"
					break
				fi
			done
			if [[ ${LENS_PYTHON} == "${PYTHON_BIN}" ]]; then
				echo "  WARNING: no python3.11 or python3.10 on PATH. pandas 1.5.3 will be built from" >&2
				echo "  source under ${PY_VERSION} and will probably fail; pass --lens-python instead." >&2
			fi
		fi
	fi

	echo "== creating venv-lens (isolated: lens-metric's pins conflict with the service's) =="
	if [[ ! -d venv-lens ]]; then
		"${LENS_PYTHON}" -m venv venv-lens
	fi
	# shellcheck disable=SC1091
	source venv-lens/bin/activate
	pip install --upgrade pip wheel >/dev/null
	if [[ -n ${CUDA_TAG} ]]; then
		pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
	else
		pip install torch
	fi
	pip install -r remote/requirements-lens.txt
	python -c "import lens; print('  lens importable')" ||
		echo "  lens still not importable -- read the error above before running the lens stage"
	deactivate
fi

cat <<EOF

== done ==

Next:
  source remote/profiles/beet.env         # HF_HOME, thread limits, venv on PATH
  python remote/gpucheck.py               # is the card free, and is it the one you expect
  python remote/preflight.py              # correctness gate -- read this before spending hours
  python remote/pipeline.py --list        # the plan, with estimates

Ollama, for the prompted-LLM conditions (no root needed):
  # ollama.com/download/ollama-linux-amd64.tgz now 404s, and the release asset is zstd
  # rather than gzip -- so both the URL and the tar flags below differ from the old recipe.
  curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \\
       -o /tmp/ollama.tar.zst
  mkdir -p "\$HOME/.local" && tar --zstd -xf /tmp/ollama.tar.zst -C "\$HOME/.local"
  export PATH="\$HOME/.local/bin:\$PATH"
  export OLLAMA_MODELS="\$PWD/scratch/ollama"     # keep 5+ GB of weights off \$HOME
  ollama serve &
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ollama pull qwen2.5:3b-instruct-q4_K_M

Keep Ollama rather than switching to vLLM even though vLLM would be faster: the thesis
pins these tags by digest and describes an Ollama sidecar as the served stack, and on a GPU
speed has stopped being the constraint. Changing the serving stack would make the numbers
describe a system the extension does not run.
EOF
