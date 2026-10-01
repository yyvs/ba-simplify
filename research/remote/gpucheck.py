"""Is this GPU actually free, and is it the one you think it is?

Two jobs, both of which exist because these are *shared* machines booked through a calendar
rather than a scheduler:

  1. **Etiquette.** There is no queue system on the cl-hhu boxes -- nothing stops two people
     training at once except one of them looking first. The motd asks you to reserve a slot
     (https://calendar.online/5afa284259e0c92ff6a2); this prints who else is currently
     resident on the card so you can tell an idle machine from a booked one before you fill
     its VRAM.
  2. **Capability.** Prints the facts that decide hyperparameters -- VRAM, compute
     capability, whether bf16 is real -- and does a short allocation and matmul so a broken
     driver surfaces here rather than at hour four. "Failed to initialize NVML:
     Driver/library version mismatch" is documented in the gpus-at-cl-hhu README as needing
     a reboot, which needs an admin, which needs knowing early.

    python remote/gpucheck.py
    python remote/gpucheck.py --benchmark    # add a throughput estimate
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import time
from typing import List, Optional

from common import GPU_RESULTS, free_gb, host_stamp, write_artifact

# The cl-hhu fleet, from the gpus-at-cl-hhu README.
FLEET = {
    'aker': ('isi-7.phil.hhu.de', 'RTX 3090', '24 GB', 'Ampere, bf16'),
    'beet': ('isi-8.phil.hhu.de', 'RTX A5000', '24 GB', 'Ampere, bf16'),
    'carrot': (
        'sfb991-35.phil-fak.uni-duesseldorf.de',
        'Quadro P5000',
        '16 GB',
        'Pascal, fp16 only',
    ),
    'shai': (
        'asw-3.phil-fak.uni-duesseldorf.de',
        'Titan Xp',
        '12 GB',
        'Pascal, fp16 only',
    ),
    'turnip': ('sfb991-7.phil.hhu.de', 'TITAN Xp', '12 GB', 'Pascal, fp16 only'),
}


def compute_processes() -> List[str]:
    """Who is using the card right now."""
    try:
        out = subprocess.run(
            [
                'nvidia-smi',
                '--query-compute-apps=pid,used_memory,process_name',
                '--format=csv,noheader',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return [
            line.strip() for line in out.stdout.strip().splitlines() if line.strip()
        ]
    except Exception:
        return []


def owner_of(pid: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ['ps', '-o', 'user=', '-p', pid], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def benchmark() -> Optional[dict]:
    """A rough sustained-matmul figure, for sanity rather than for reporting: catches a
    throttled or already-busy GPU."""
    import torch

    if not torch.cuda.is_available():
        return None
    torch.backends.cuda.matmul.allow_tf32 = True
    size = 4096
    a = torch.randn(
        size,
        size,
        device='cuda',
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    b = torch.randn_like(a)
    for _ in range(3):  # warm-up: compilation, clock ramp
        _ = a @ b
    torch.cuda.synchronize()
    started = time.time()
    iterations = 30
    for _ in range(iterations):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - started
    tflops = (2 * size**3 * iterations) / elapsed / 1e12
    return {
        'matmul_size': size,
        'dtype': str(a.dtype),
        'iterations': iterations,
        'seconds': round(elapsed, 3),
        'approx_tflops': round(tflops, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--no-artifact', action='store_true')
    args = ap.parse_args()

    stamp = host_stamp()
    short = (stamp.get('hostname') or '').split('.')[0]

    print('\nHOST')
    print(f"  hostname        {stamp.get('hostname')}")
    if short in FLEET:
        dns, gpu, vram, arch = FLEET[short]
        print(f'  known machine   {short} — {gpu}, {vram} ({arch})')
    print(f"  python          {stamp.get('python')}    cpus {stamp.get('cpu_count')}")
    print(f'  free disk       {free_gb():.1f} GB at research/scratch')
    print(
        f"  HF_HOME         {os.environ.get('HF_HOME') or 'unset (will use ~/.cache)'}"
    )

    print('\nGPU')
    print(f"  driver          {stamp.get('nvidia_driver')}")
    print(
        f"  reported        {stamp.get('gpu_name')} / {stamp.get('gpu_memory_total')}"
    )
    print(
        f"  torch           {stamp.get('torch')} (cuda build {stamp.get('torch_cuda_build')})"
    )
    if stamp.get('cuda_available'):
        print(
            f"  visible to torch {stamp.get('device_name')} · {stamp.get('vram_gb')} GB · "
            f"cc {stamp.get('compute_capability')} · bf16 {stamp.get('bf16_supported')}"
        )
    else:
        print(
            f"  visible to torch NO — {stamp.get('error', 'torch.cuda.is_available() is False')}"
        )
        print(
            "\n  If nvidia-smi itself reports 'Failed to initialize NVML: Driver/library"
        )
        print("  version mismatch', the machine needs a reboot and that needs an admin")
        print('  (David Arps or Kilian Evang, per the gpus-at-cl-hhu README).')

    print('\nWHO ELSE IS ON THIS CARD')
    procs = compute_processes()
    me = getpass.getuser()
    if not procs:
        print('  nothing resident — the card is idle')
    else:
        foreign = 0
        for line in procs:
            pid = line.split(',')[0].strip()
            user = owner_of(pid) or '?'
            mine = user == me
            foreign += not mine
            print(f"  {'(yours)' if mine else '(SOMEONE ELSE)'} {line}  user={user}")
        if foreign:
            print(
                f'\n  {foreign} process(es) belong to another user. These machines have no\n'
                f'  queue system, so nothing will stop you from competing with them for VRAM\n'
                f'  except this message. Check the reservation calendar before you start:\n'
                f'    https://calendar.online/5afa284259e0c92ff6a2'
            )

    bench = benchmark() if args.benchmark else None
    if bench:
        print('\nTHROUGHPUT (sanity check, not a reportable measurement)')
        print(
            f"  {bench['matmul_size']}^2 {bench['dtype']} matmul × {bench['iterations']}: "
            f"{bench['seconds']}s → ~{bench['approx_tflops']} TFLOP/s"
        )
        print(
            "  Far below the card's spec usually means someone else is on it, or it is "
            'thermally throttled.'
        )

    print('\nSUGGESTED SETTINGS FOR THIS CARD')
    vram = stamp.get('vram_gb') or 0
    if not stamp.get('cuda_available'):
        print('  — resolve the CUDA problem above first')
    elif vram >= 22:
        print('  document training: --batch-size 32   (512 tokens, bf16)')
        print('  document eval:     --batch-size 16')
        print('  sentence training: --batch-size 64   (64 tokens)')
    elif vram >= 15:
        print("  document training: --batch-size 16   (matches the project's T4 runs)")
        print('  document eval:     --batch-size 8')
        print('  sentence training: --batch-size 32')
    else:
        print('  12 GB card (turnip/shai): document training --batch-size 8, eval 4.')
        print('  Pascal has no bf16, so train.py will select fp16 automatically.')
        print(
            '  A full-scope document run here is viable but slow; prefer beet or aker.'
        )

    if not args.no_artifact:
        write_artifact(
            GPU_RESULTS / 'gpucheck.json',
            {'host': stamp, 'compute_processes': procs, 'benchmark': bench},
            quiet=True,
        )
    print()


if __name__ == '__main__':
    main()
