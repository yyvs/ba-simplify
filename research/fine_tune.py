import argparse
import logging
import os
import re

import numpy as np
import torch
from datasets import Dataset, load_dataset
from hf_revisions import HF_REVISIONS
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_whitespace(text: str) -> str:
    """
    Normalizes whitespace identically to the backend's extract_and_clean() function.
    Collapses multiple whitespace characters and strips leading/trailing spaces.
    """
    if not text:
        return ''
    return re.sub(r'\s+', ' ', text).strip()


def contains_mojibake(text: str) -> bool:
    """
    Detects mojibake artifacts inherited from WikiLarge's original alignment pipeline.
    Typically, multi-byte UTF-8 sequences (like accented characters or en-dashes) are garbled
    into byte fragments split by a stray space, e.g. '1809 â '' 11' instead of '1809 - 11'.
    As legitimate English accented text essentially never contains a bare 'â' or 'Ã',
    dropping rows containing these characters removes approximately 5% of garbled rows
    while preserving correct accented words like 'São Tomé' or 'Atlético Madrid'.
    """
    return 'â' in text or 'Ã' in text


def preprocess_and_filter(dataset) -> Dataset:
    """
    Applies whitespace normalization and filters out mojibake rows.
    """
    logger.info('Starting dataset preprocessing and filtering...')
    cleaned_rows = []

    total_rows = len(dataset)
    dropped_mojibake = 0

    for row in dataset:
        src = row['complex']
        tgt = row['simple']

        if contains_mojibake(src) or contains_mojibake(tgt):
            dropped_mojibake += 1
            continue

        src_clean = clean_whitespace(src)
        tgt_clean = clean_whitespace(tgt)

        cleaned_rows.append({'complex': src_clean, 'simple': tgt_clean})

    logger.info(
        f'Processed {total_rows} rows. Dropped {dropped_mojibake} rows due to mojibake. Remaining: {len(cleaned_rows)}'
    )
    return Dataset.from_list(cleaned_rows)


def load_asset_split(split: str = 'validation'):
    """
    Loads ASSET (Alva-Manchego et al., 2020): a crowdsourced, multi-reference (10 human
    simplifications per sentence) benchmark, used here in place of WikiLarge's validation
    split. ASSET has no train split -- it's evaluation-only, so it doesn't touch fine-tuning,
    only what gets reported as eval loss/SARI/BLEU/FKGL during training.
    """
    logger.info(f'Loading ASSET ({split!r} split) for evaluation...')
    raw = load_dataset(
        'facebook/asset',
        'simplification',
        split=split,
        revision=HF_REVISIONS.get('facebook/asset'),
    )
    sources = [clean_whitespace(s) for s in raw['original']]
    references = [
        [clean_whitespace(r) for r in refs] for refs in raw['simplifications']
    ]
    return sources, references


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune a BART or T5 model for English Text Simplification.'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='facebook/bart-base',
        help='Base model to fine-tune (e.g. facebook/bart-base or t5-base)',
    )
    parser.add_argument(
        '--prove_loop',
        action='store_true',
        help='Run a quick end-to-end loop with 200 examples for 1 epoch to validate the pipeline.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save model checkpoints and logs.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs for full training.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training and evaluation.',
    )
    parser.add_argument(
        '--learning_rate', type=float, default=3e-5, help='Learning rate.'
    )

    args = parser.parse_args()

    # WikiLarge (clean) is training data only. ASSET replaces its validation split: a
    # clean multi-reference benchmark, not the same noisy automatic alignment the
    # training pairs come from.
    logger.info("Loading 'eilamc14/wikilarge-clean' dataset...")
    raw_datasets = load_dataset(
        'eilamc14/wikilarge-clean',
        revision=HF_REVISIONS.get('eilamc14/wikilarge-clean'),
    )

    logger.info('Preprocessing train split...')
    train_dataset = preprocess_and_filter(raw_datasets['train'])
    asset_sources, asset_references = load_asset_split(split='validation')

    if args.prove_loop:
        logger.info('--- PROVE LOOP MODE ENABLED ---')
        logger.info(
            'Running on a tiny subset of 200 training and 50 validation examples for 1 epoch.'
        )
        train_dataset = train_dataset.select(range(min(200, len(train_dataset))))
        asset_sources = asset_sources[:50]
        asset_references = asset_references[:50]
        args.epochs = 1
        args.output_dir = './prove_loop_results'

    # First ASSET reference is the loss target; all 10 (`asset_references`) are used for
    # SARI/BLEU in compute_metrics.
    val_dataset = Dataset.from_dict(
        {
            'complex': asset_sources,
            'simple': [refs[0] for refs in asset_references],
        }
    )

    logger.info(f'Loading tokenizer and model for {args.model_name_or_path}...')
    revision = HF_REVISIONS.get(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, revision=revision
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, revision=revision
    )

    # T5 requires a task prefix
    is_t5 = 't5' in args.model_name_or_path.lower()
    prefix = 'simplify: ' if is_t5 else ''

    max_input_length = 512

    def tokenize_function(examples):
        inputs = [prefix + doc for doc in examples['complex']]
        # text_target= tokenizes targets with matching special tokens/vocab IDs
        # (as_target_tokenizer() was removed in transformers 5.x)
        model_inputs = tokenizer(
            inputs,
            text_target=examples['simple'],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        return model_inputs

    logger.info('Tokenizing datasets...')
    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, remove_columns=['complex', 'simple']
    )
    tokenized_val = val_dataset.map(
        tokenize_function, batched=True, remove_columns=['complex', 'simple']
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    eval_strategy = 'steps' if args.prove_loop else 'epoch'
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=eval_strategy,
        save_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=None,  # keep every checkpoint
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=10 if args.prove_loop else 100,
        eval_steps=20 if args.prove_loop else None,
        fp16=torch.cuda.is_available(),
        report_to='none',  # no wandb etc., for local predictability
    )

    # SARI/BLEU/FKGL via EASSE if installed, else sacrebleu BLEU only.
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # -100 can't be decoded
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        metrics = {}
        try:
            # EASSE's public API is corpus_sari/corpus_bleu/corpus_fkgl. An earlier
            # version imported get_sari/get_bleu/get_fkgl (in no release), so the
            # ImportError fallback ran silently and SARI/FKGL were never computed during
            # training.
            from easse.bleu import corpus_bleu
            from easse.fkgl import corpus_fkgl
            from easse.sari import corpus_sari

            # Without `prefix`: SARI compares against the original sentence, and the T5
            # task prefix is an input convention, not part of it.
            sources = list(val_dataset['complex'])

            # (n_samples, 10 refs) -> (n_references, n_samples), as SARI/BLEU expect
            references = [list(refs) for refs in zip(*asset_references, strict=True)]

            metrics['sari'] = corpus_sari(
                orig_sents=sources, sys_sents=decoded_preds, refs_sents=references
            )
            metrics['bleu'] = corpus_bleu(
                sys_sents=decoded_preds, refs_sents=references
            )
            # FKGL is a ratio of corpus totals; a mean of per-sentence FKGL is a
            # different, noisier quantity.
            metrics['fkgl'] = corpus_fkgl(sentences=decoded_preds)

        except ImportError:
            logger.warning(
                "EASSE library not found. Skipping SARI/FKGL computation in logging. Please run 'pip install easse' or check step-by-step-v2."
            )
            # Fallback: sacrebleu BLEU if available
            try:
                import sacrebleu

                bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
                metrics['bleu'] = bleu.score
            except ImportError:
                pass

        return metrics

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        # transformers types eval_dataset as a torch Dataset while typing
        # train_dataset to also accept this one; both take a datasets.Dataset.
        eval_dataset=tokenized_val,  # pyright: ignore[reportArgumentType]
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics if not args.prove_loop else None
        ),
    )

    logger.info('Starting training loop...')
    trainer.train()

    logger.info(
        f'Saving final fine-tuned model and tokenizer to {args.output_dir}/final_model...'
    )
    trainer.save_model(os.path.join(args.output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    logger.info('Training pipeline execution completed successfully!')


if __name__ == '__main__':
    main()
