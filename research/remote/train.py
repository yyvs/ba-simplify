"""Resumable seq2seq fine-tuning for the HHU GPU boxes -- the canonical training entry point.

**Why a script and not the notebook.** Every real training run in this project came from
`notebooks/train_sentence_and_document_pipeline.ipynb`, and the logic below is a faithful
port of its `run_fine_tuning()` (same optimiser settings, same fixed-length padding, same
chunked tokenisation, same step-based eval/save cadence, same early stopping). It is a
script here for four reasons that all bite on a shared, reservation-scheduled machine:

  1. A 6-hour `nbconvert --execute` that dies at hour 5 loses everything; `--resume` costs
     one checkpoint interval.
  2. The notebook's Colab cells (`drive.mount`, `google.colab.userdata`) fail outside Colab.
  3. `nohup`/`tmux` + a log file is how you survive a dropped SSH; a kernel is not.
  4. The thesis §4.8 asks the project to "decide, and state explicitly, which one is
     the canonical/reproducible entry point". `research/fine_tune.py` cannot be it -- it
     reads columns "complex"/"simple" from a dataset whose columns are "source"/"target",
     so it raises KeyError on the first row. This file is the answer, and the notebook
     becomes the exploratory/illustrative copy.

**What differs from the notebook's CUDA path, and why.** All four differences exist because
the target hardware is an Ampere card with 24 GB and no session limit, where the notebook's
CUDA settings were tuned for a 16 GB Turing T4 inside a 5h20m Colab session:

  - **bf16 instead of fp16.** The notebook sets `fp16=torch.cuda.is_available()` and
    comments that T4s "predate hardware bf16 support entirely". The A5000 (sm_86) does not:
    it has real bf16 throughput, and bf16's wider dynamic range removes loss scaling as a
    failure mode. Gated on `torch.cuda.is_bf16_supported()` so a Titan Xp box (turnip/shai,
    sm_61) still gets fp16.
  - **TF32 matmuls enabled.** Free on Ampere, off by default in recent torch.
  - **No generation during evaluation.** The notebook runs `predict_with_generate=True`,
    which calls `generate()` over the whole validation split at every eval step -- but
    `metric_for_best_model="loss"`, so not one of those generations affects checkpoint
    selection. Turning it off is the largest single speedup available here and changes
    nothing about which checkpoint wins. `--generate-during-eval` restores the old
    behaviour if you want sample output in the log.
  - **Document full scope defaults to 5 epochs, not 2.** The notebook's 2-epoch cap is
    explicitly an artefact of the Colab session ceiling ("an experimental-design constraint
    imposed by the compute environment, not a modeling decision" -- §4.5). Removing that
    ceiling is the main reason this session exists, so the cap goes and early stopping
    decides when to stop instead.

Run it under tmux. Reserve the GPU first: https://calendar.online/5afa284259e0c92ff6a2

    tmux new -s train
    python train.py --structure document --scope full 2>&1 | tee -a \
        ../results/remote/logs/train_document_full.log
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from common import (
    GPU_RESULTS,
    SCRATCH,
    human_seconds,
    mark_stage,
    package_versions,
    require_disk,
    write_artifact,
)
from dataio import FILTERS, load_corpus

# hf_revisions lives in research/; `python remote/x.py` only puts remote/ on sys.path
# (same shim as lens_only.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hf_revisions import HF_REVISIONS  # noqa: E402

MAX_LENGTH = {'sentence': 64, 'document': 512}
"""Token ceilings, same as the notebook and backend/main.py `SEQ2SEQ_MAX_LENGTH`.

From measured percentiles, not BART's 1024 limit: WikiLarge sources mean ~33 / p90 ~52;
D-Wikipedia mean ~172 / p95 ~499 / p99 ~742 (512 covers ~95%, a quarter of 1024's
attention cost). Do not raise for runs meant to be comparable: a train/inference length
mismatch is the defect §5.6 records fixing.
"""


def pick_device(requested: str = 'auto') -> str:
    import torch

    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def default_batch_size(structure: str, scope: str, device: str, vram_gb: float) -> int:
    """Batch size per device, sized to the card rather than to a fixed constant.

    Measured on Colab: document mode at 512 tokens, batch 16 used 6.5 of 16 GB on a T4, and
    8 -> 16 raised throughput from 19.8 to 22.1 examples/sec; so 32 fits a 24 GB A5000.
    Sentence mode was never memory-constrained (batch 64 is fine on 12 GB+).
    """
    if scope == 'prove_loop':
        return 2 if structure == 'document' else 8
    if device != 'cuda':
        # MPS shares memory with the host: the notebook's small batch + accumulation
        return 2 if structure == 'document' else 16
    if structure == 'document':
        if vram_gb >= 22:
            return 32
        if vram_gb >= 15:
            return 16
        return 8  # 12 GB cards: turnip, shai
    return 64 if vram_gb >= 22 else 32


class SimplificationDataset:
    """Minimal map-style dataset over pre-tokenised, fixed-length arrays.

    Fixed-length padding, from the notebook: MPS's caching allocator never reuses
    differently-shaped blocks, so per-batch shapes grew reserved memory to 13+ GiB within
    ~100 steps. Slightly wasteful on CUDA but one code path; `pad_to_multiple_of=8` keeps
    shapes Tensor-Core-friendly.
    """

    def __init__(self, encodings: Dict[str, np.ndarray]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {key: value[idx].tolist() for key, value in self.encodings.items()}


def tokenize(
    tokenizer,
    sources: List[str],
    targets: List[str],
    max_length: int,
    chunk_size: int = 2000,
) -> Dict[str, np.ndarray]:
    """Tokenise in bounded chunks, converting each chunk to packed int32 immediately.

    A fast tokenizer builds nested Python lists first (~30-50 bytes/token): ~8-9 GB peak for
    D-Wikipedia's 132k-document train split at 512 tokens, which OOM-killed a 12.7 GB Colab
    session without a traceback. Chunking bounds the peak, int32 the result. Chunked and
    unchunked output are bit-identical (verified) because padding is to a fixed max_length.
    """
    out = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for start in range(0, len(sources), chunk_size):
        batch = tokenizer(
            sources[start : start + chunk_size],
            text_target=targets[start : start + chunk_size],
            max_length=max_length,
            truncation=True,
            padding='max_length',
        )
        for key in out:
            out[key].append(np.array(batch[key], dtype=np.int32))
    return {key: np.concatenate(arrays, axis=0) for key, arrays in out.items()}


def find_resume_checkpoint(output_dir: Path) -> Optional[str]:
    """Highest-numbered `checkpoint-N` in `output_dir`, or None.

    Done here rather than via `resume_from_checkpoint=True` so the manifest records it.
    """
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        (
            p
            for p in output_dir.iterdir()
            if p.is_dir() and p.name.startswith('checkpoint-')
        ),
        key=lambda p: int(p.name.split('-')[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--structure', choices=['sentence', 'document'], required=True)
    ap.add_argument(
        '--scope', choices=['prove_loop', 'reduced', 'full'], default='full'
    )
    ap.add_argument('--model', default='facebook/bart-base', help='starting checkpoint')
    ap.add_argument(
        '--filter',
        dest='mode',
        choices=list(FILTERS),
        default='strict',
        help='mojibake filter; recorded in the manifest (see dataio.py)',
    )
    ap.add_argument(
        '--epochs', type=int, default=None, help='override the scope default'
    )
    ap.add_argument(
        '--batch-size', type=int, default=None, help='override the card-sized default'
    )
    ap.add_argument('--grad-accum', type=int, default=None)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    ap.add_argument(
        '--output-dir',
        default=None,
        help='default: scratch/simplification_results_<structure>',
    )
    ap.add_argument(
        '--save-total-limit',
        type=int,
        default=3,
        help='checkpoints to keep (each ~1.6 GB with optimiser state). The '
        'metric trajectory survives pruning regardless -- it goes to '
        'training_history.json, which is what plotting needs.',
    )
    ap.add_argument(
        '--early-stopping-patience',
        type=int,
        default=4,
        help='in eval steps, i.e. quarter-epochs; 4 = roughly one epoch',
    )
    ap.add_argument(
        '--generate-during-eval',
        action='store_true',
        help="restore the notebook's predict_with_generate=True. Costs a full "
        'generate() over the validation split at every eval step and '
        'cannot affect checkpoint selection, which is on loss.',
    )
    ap.add_argument(
        '--resume', action='store_true', help='continue from the latest checkpoint'
    )
    ap.add_argument(
        '--natural-case',
        action='store_true',
        help='document only: train on natural-case, normally-punctuated text '
        "instead of D-Wikipedia's lowercased PTB convention. This is the "
        "stretch experiment (§7.1's 'highest-value follow-up') -- it "
        'de-normalises the corpus so the checkpoint matches the prose the '
        'extension actually sends, removing the need for '
        "document_text.py's normalize/de-normalize workaround.",
    )
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    device = pick_device(args.device)
    vram_gb = 0.0
    if device == 'cuda':
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # free on Ampere+, off by default in recent torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Shuffling and dropout. `set_seed` does not reach the MPS generator. Two otherwise
    # identical unseeded runs differed by 2.0 SARI (rows S1 vs S2): the noise floor.
    set_seed(args.seed)
    if device == 'mps':
        torch.mps.manual_seed(args.seed)

    run_tag = f'{args.structure}_{args.scope}'
    if args.natural_case:
        run_tag += '_naturalcase'
    # separate checkpoint trees per structure / natural-case variant
    default_name = f'simplification_results_{args.structure}'
    if args.natural_case:
        default_name += '_naturalcase'
    output_dir = Path(args.output_dir) if args.output_dir else SCRATCH / default_name
    require_disk(where=SCRATCH)
    mark_stage(f'train_{run_tag}', 'running')

    # data
    print(
        f'[1/4] loading corpus (structure={args.structure}, scope={args.scope}, filter={args.mode})'
    )
    data = load_corpus(args.structure, args.scope, args.mode)

    if args.natural_case:
        if args.structure != 'document':
            raise SystemExit('--natural-case only applies to the document corpus')
        from denormalize_corpus import denormalize_split

        print('      de-normalising D-Wikipedia into natural-case prose')
        for split in ('train', 'validation', 'test'):
            data[split] = denormalize_split(data[split])

    max_length = MAX_LENGTH[args.structure]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=HF_REVISIONS.get(args.model)
    )

    print(f'[2/4] tokenising at max_length={max_length}')
    train_tokens = tokenize(
        tokenizer, data['train']['source'], data['train']['target'], max_length
    )
    val_tokens = tokenize(
        tokenizer,
        data['validation']['source'],
        data['validation']['target'],
        max_length,
    )
    train_dataset = SimplificationDataset(train_tokens)
    val_dataset = SimplificationDataset(val_tokens)

    # hyperparameters
    if args.epochs is not None:
        epochs = args.epochs
    elif args.scope == 'prove_loop':
        epochs = 1
    elif args.scope == 'reduced':
        epochs = 2
    else:
        # The notebook's 2 for document/full was only Colab's 5h20m cap; early stopping decides.
        epochs = 5

    batch_size = args.batch_size or default_batch_size(
        args.structure, args.scope, device, vram_gb
    )
    grad_accum = args.grad_accum or (
        2 if device == 'mps' and args.structure == 'document' else 1
    )
    use_grad_checkpointing = args.structure == 'document' and device != 'cuda'

    use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    use_fp16 = device == 'cuda' and not use_bf16

    steps_per_epoch = max(1, math.ceil(len(train_dataset) / (batch_size * grad_accum)))
    eval_steps = max(1, round(steps_per_epoch * 0.25))
    # ~once per epoch (saves include optimiser state); must be a multiple of eval_steps
    # with load_best_model_at_end=True
    save_steps = eval_steps * 4

    print(
        f'[3/4] {len(train_dataset)} train / {len(val_dataset)} val examples\n'
        f'      device={device} vram={vram_gb:.0f}GB batch={batch_size} accum={grad_accum} '
        f"precision={'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}\n"
        f'      {steps_per_epoch} steps/epoch x {epochs} epochs; eval every {eval_steps}, '
        f'save every {save_steps}\n'
        f"      generation during eval: {'ON' if args.generate_during_eval else 'OFF (selection is on loss)'}"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, revision=HF_REVISIONS.get(args.model)
    )
    model.to(device)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,  # padding excluded from the loss
        pad_to_multiple_of=8 if device == 'cuda' else None,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy='steps',
        eval_steps=eval_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_checkpointing,
        weight_decay=args.weight_decay,
        num_train_epochs=epochs,
        predict_with_generate=args.generate_during_eval,
        logging_strategy='steps',
        logging_steps=eval_steps,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_pin_memory=(device == 'cuda'),
        dataloader_num_workers=4 if device == 'cuda' else 0,
        seed=args.seed,
        report_to='none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        # duck-typed Dataset, not a subclass
        train_dataset=train_dataset,  # pyright: ignore[reportArgumentType]
        eval_dataset=val_dataset,  # pyright: ignore[reportArgumentType]
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    resume_from = find_resume_checkpoint(output_dir) if args.resume else None
    if resume_from:
        print(f'      resuming from {resume_from}')

    # train
    print('[4/4] training')
    started = time.time()
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        # reservation ended or Ctrl-C; the last periodic checkpoint is intact
        elapsed = time.time() - started
        mark_stage(f'train_{run_tag}', 'interrupted', elapsed_seconds=elapsed)
        raise SystemExit(
            f'\nInterrupted after {human_seconds(elapsed)}. Resume with the same command '
            f'plus --resume; it restarts from the newest checkpoint in {output_dir}.'
        ) from None
    elapsed = time.time() - started

    # weights only (~535 MB) vs. ~1.6 GB intermediate checkpoints with optimiser state
    best_dir = output_dir / 'best_checkpoint'
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Saved separately so the trajectory survives save_total_limit pruning (§6.6: earlier
    # runs lost it to save_total_limit=2).
    history = trainer.state.log_history

    manifest = {
        'run': {
            'tag': run_tag,
            'structure': args.structure,
            'scope': args.scope,
            'natural_case': args.natural_case,
            'base_model': args.model,
            'seed': args.seed,
            'elapsed_seconds': round(elapsed, 1),
            'elapsed_human': human_seconds(elapsed),
            'resumed_from': resume_from,
        },
        'corpus': {
            'mojibake_filter': args.mode,
            'split_stats': data['stats'],
            'note': (
                'kept_rows here is the corpus this checkpoint actually trained on. Quote it '
                'in the thesis rather than data/stats.json, which was produced by '
                'prepare_data.py and may use a different filter -- see '
                'results/remote/corpus_filter_divergence.json.'
            ),
        },
        'hyperparameters': {
            'max_length': max_length,
            'epochs': epochs,
            'per_device_batch_size': batch_size,
            'gradient_accumulation_steps': grad_accum,
            'effective_batch_size': batch_size * grad_accum,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'precision': 'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32',
            'gradient_checkpointing': use_grad_checkpointing,
            'steps_per_epoch': steps_per_epoch,
            'eval_steps': eval_steps,
            'save_steps': save_steps,
            'save_total_limit': args.save_total_limit,
            'early_stopping_patience': args.early_stopping_patience,
            'predict_with_generate': args.generate_during_eval,
            'tf32': device == 'cuda',
        },
        'selection': {
            'metric_for_best_model': 'loss',
            'best_model_checkpoint': trainer.state.best_model_checkpoint,
            'best_metric': trainer.state.best_metric,
            'global_step': trainer.state.global_step,
            'epoch': trainer.state.epoch,
            'best_checkpoint_dir': str(best_dir),
            'caveat': (
                "Validation loss is measured on the corpus's own validation split. For "
                'WikiLarge, 71 of 397 validation sources (17.9%) also appear in the train '
                'split, so absolute validation losses are optimistically biased; the '
                'relative comparison between checkpoints is unaffected because every '
                'checkpoint faces the same set. See results/remote/overlap_audit.json.'
            ),
        },
        'packages': package_versions(
            'torch', 'transformers', 'datasets', 'tokenizers', 'accelerate', 'numpy'
        ),
        'log_history': history,
    }
    write_artifact(GPU_RESULTS / f'train_{run_tag}.json', manifest)
    (GPU_RESULTS / f'training_history_{run_tag}.json').write_text(
        json.dumps(history, indent=2)
    )

    mark_stage(
        f'train_{run_tag}',
        'done',
        best_checkpoint=str(best_dir),
        elapsed_seconds=round(elapsed, 1),
    )
    print(
        f'\nDone in {human_seconds(elapsed)}.\n'
        f'  best checkpoint: {best_dir}\n'
        f'  best val loss:   {trainer.state.best_metric}\n'
        f'  manifest:        results/remote/train_{run_tag}.json'
    )


if __name__ == '__main__':
    main()
