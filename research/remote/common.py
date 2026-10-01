"""Shared plumbing for the HHU GPU runs: paths, host provenance, run manifests, disk guard.

Every script here writes output through `stamp()` and `write_artifact()`, so every number
can name the machine, driver, commit and seed that produced it. (Earlier numbers came from
four machines with different filters and decoding; see the caveats in thesis §6.6.)

Nothing from ../ is re-implemented: evaluation scripts, metrics and prompt templates are
invoked, never copied. dataio.py is the one exception (see its docstring).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Paths resolve relative to research/, like the existing scripts' `scratch/` and `results/`.
HHU_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = HHU_DIR.parent
REPO_DIR = RESEARCH_DIR.parent

SCRATCH = RESEARCH_DIR / 'scratch'
RESULTS = RESEARCH_DIR / 'results'
# Own tree: a session can be reviewed or discarded as a unit and never overwrites the
# committed results the thesis cites.
GPU_RESULTS = RESULTS / 'remote'
GPU_LOGS = GPU_RESULTS / 'logs'
STATE_FILE = GPU_RESULTS / 'pipeline_state.json'

MIN_FREE_GB = 40.0
"""Refuse to start a training stage below this. A 5-epoch document run keeps ~5
checkpoints at ~1.6 GB each (optimizer + scheduler + RNG state), plus the best-checkpoint
copy and the HF cache. Hard gate: filling a shared machine's disk damages others' work.
"""


def ensure_dirs() -> None:
    for d in (SCRATCH, RESULTS, GPU_RESULTS, GPU_LOGS):
        d.mkdir(parents=True, exist_ok=True)


# provenance
def git_commit() -> Optional[str]:
    """Short HEAD, or None outside a checkout (D2 has `git_commit: null`: it ran from a Colab upload)."""
    try:
        out = subprocess.run(
            ['git', '-C', str(REPO_DIR), 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def git_dirty() -> Optional[bool]:
    """Whether the tree has uncommitted changes, i.e. `git_commit` does not identify the code."""
    try:
        out = subprocess.run(
            ['git', '-C', str(REPO_DIR), 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def nvidia_smi(query: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        return (
            ', '.join(
                line.strip() for line in out.stdout.strip().splitlines() if line.strip()
            )
            or None
        )
    except Exception:
        return None


def torch_info() -> Dict[str, Any]:
    """Torch/CUDA facts, or a reason why torch can't see a GPU.

    Never raises, so preflight can report a broken CUDA setup: "Failed to initialize NVML:
    Driver/library version mismatch" is a documented failure on these machines
    (gpus-at-cl-hhu README) that needs a reboot.
    """
    info: Dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:
        return {'torch': None, 'error': f'torch not importable: {exc}'}

    info['torch'] = torch.__version__
    info['torch_cuda_build'] = getattr(torch.version, 'cuda', None)
    try:
        info['cuda_available'] = bool(torch.cuda.is_available())
        if info['cuda_available']:
            info['device_name'] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info['compute_capability'] = f'{major}.{minor}'
            info['device_count'] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info['vram_gb'] = round(props.total_memory / 1024**3, 1)
            info['bf16_supported'] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        info['cuda_available'] = False
        info['error'] = f'{type(exc).__name__}: {exc}'
    return info


def host_stamp() -> Dict[str, Any]:
    """The block every artifact in this session carries."""
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'cpu_count': os.cpu_count(),
        'nvidia_driver': nvidia_smi('driver_version'),
        'gpu_name': nvidia_smi('name'),
        'gpu_memory_total': nvidia_smi('memory.total'),
        **torch_info(),
    }


def stamp(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A full provenance block: when, where, which code, which environment."""
    block = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_commit(),
        'git_dirty': git_dirty(),
        'host': host_stamp(),
        'hf_home': os.environ.get('HF_HOME'),
        'profile': os.environ.get('HHU_PROFILE'),
    }
    if extra:
        block.update(extra)
    return block


def package_versions(*names: str) -> Dict[str, Optional[str]]:
    """Installed versions of the packages that can change a reported number.

    transformers, datasets, easse, sacrebleu and bert-score are pinned in
    requirements-gpu.txt because they move metrics; torch is not, since the wheel must match
    the host driver.
    """
    from importlib import metadata

    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = None
    return out


# artifacts
def rel(path: Path) -> str:
    """`path` as a research/-relative string, or unchanged if it is outside the tree.

    Guarded because Path.relative_to raises on a relative path vs. absolute RESEARCH_DIR
    (e.g. `--outdir results/...`); that once crashed a finished LENS run after writing.
    """
    try:
        return str(Path(path).resolve().relative_to(Path(RESEARCH_DIR).resolve()))
    except ValueError:
        return str(path)


def write_artifact(path: Path, payload: Dict[str, Any], *, quiet: bool = False) -> Path:
    """Write JSON with a provenance block merged in, atomically.

    Runs get interrupted, and a half-written JSON that parses would look like a result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {'provenance': stamp(), **payload}
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(body, indent=2, ensure_ascii=False))
    tmp.replace(path)
    if not quiet:
        print(f'  wrote {rel(path)}')
    return path


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# disk guard
def free_gb(where: Path = SCRATCH) -> float:
    target = where if where.exists() else where.parent
    return shutil.disk_usage(target).free / 1024**3


def require_disk(min_gb: float = MIN_FREE_GB, where: Path = SCRATCH) -> None:
    free = free_gb(where)
    if free < min_gb:
        raise SystemExit(
            f'Only {free:.1f} GB free at {where}; this stage needs about {min_gb:.0f} GB.\n'
            f'These machines are shared and the motd asks you to delete large files when '
            f"you're done. Reclaim space with:\n"
            f'  python {HHU_DIR.name}/cleanup.py --dry-run\n'
            f"then rerun without --dry-run once you've read what it would remove."
        )


# pipeline state (stage completion, for resume)
def load_state() -> Dict[str, Any]:
    return read_json(STATE_FILE) or {'stages': {}}


def save_state(state: Dict[str, Any]) -> None:
    ensure_dirs()
    tmp = STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def mark_stage(name: str, status: str, **fields: Any) -> None:
    state = load_state()
    state['stages'][name] = {
        'status': status,
        'updated_at': datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    save_state(state)


def stage_status(name: str) -> Optional[str]:
    return (load_state()['stages'].get(name) or {}).get('status')


def human_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h{m:02d}m' if h else (f'{m}m{s:02d}s' if m else f'{s}s')
