"""Load WikiLarge, clean it, and write train/validation/test splits for
fine-tuning (Phase 3, Step 2 — see docs/roadmap.md and docs/step-by-step-v2.md).

WikiSmall is deliberately not used: its classic form replaces named entities
with placeholder tags (PERSON@1, LOCATION@1, ...) baked in by the original
2010 alignment pipeline, which would teach the model to expect/produce those
tags on real text. See data/README.md for the full writeup.

Usage:
    cd training
    ../backend/venv/bin/python3 -m pip install -r requirements.txt
    ../backend/venv/bin/python3 prepare_data.py
"""

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence, cast

from datasets import load_dataset
from hf_revisions import HF_REVISIONS

DATASET_NAME = 'eilamc14/wikilarge-clean'
SPLITS = ('train', 'validation', 'test')
DATA_DIR = Path(__file__).resolve().parent / 'data'

# WikiLarge and similar Wikipedia-derived corpora of that era's alignment tooling
# garble multi-byte UTF-8 (accents, en-dashes) into fragments split by a stray space,
# e.g. "1809 â '' 11" for "1809 - 11", "TÃ xi" for "Táxi". ftfy can't repair it (the
# byte sequence is no longer contiguous). Legitimate English essentially never has a
# bare "â" or "Ã", so such rows are dropped.
MOJIBAKE_MARKERS = ('â', 'Ã')


def clean_text(text: str) -> str:
    """Match backend/main.py's extract_and_clean() whitespace collapsing so
    train-time and inference-time preprocessing agree."""
    return re.sub(r'\s+', ' ', text).strip()


def is_clean(source: str, target: str) -> bool:
    combined = source + target
    return not any(marker in combined for marker in MOJIBAKE_MARKERS)


def word_length_stats(texts):
    lengths = [len(t.split()) for t in texts]
    return {
        'mean': round(statistics.mean(lengths), 1),
        'median': statistics.median(lengths),
        'min': min(lengths),
        'max': max(lengths),
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, revision=HF_REVISIONS.get(DATASET_NAME))

    all_stats = []
    sample_for_review = None

    for split in SPLITS:
        # cast: DatasetDict.__getitem__ is a union over streaming/non-streaming types
        # and Dataset.__iter__ is inferred as yielding lists too; a plain
        # load_dataset() gives a sized sequence of dict rows.
        raw_rows = cast(Sequence[Dict[str, Any]], dataset[split])
        kept = []
        dropped_mojibake = 0

        for ex in raw_rows:
            source = clean_text(ex['source'])
            target = clean_text(ex['target'])
            if not is_clean(source, target):
                dropped_mojibake += 1
                continue
            kept.append({'source': source, 'target': target})

        out_path = DATA_DIR / f'{split}.jsonl'
        with out_path.open('w', encoding='utf-8') as f:
            for pair in kept:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

        vocab = Counter()
        for pair in kept:
            vocab.update(pair['source'].lower().split())
            vocab.update(pair['target'].lower().split())

        stats = {
            'split': split,
            'raw_rows': len(raw_rows),
            'dropped_mojibake_rows': dropped_mojibake,
            'kept_rows': len(kept),
            'vocab_size': len(vocab),
            'source_length_words': word_length_stats([p['source'] for p in kept]),
            'target_length_words': word_length_stats([p['target'] for p in kept]),
        }
        all_stats.append(stats)
        print(
            f"{split}: kept {stats['kept_rows']}/{stats['raw_rows']} rows "
            f"(dropped {stats['dropped_mojibake_rows']} mojibake) -> {out_path}"
        )

        if split == 'train':
            random.seed(42)
            sample_for_review = random.sample(kept, 10)

    stats_path = DATA_DIR / 'stats.json'
    with stats_path.open('w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'\nwrote stats to {stats_path}')

    print('\n--- sanity check: 10 random train pairs (read these by hand) ---')
    for pair in sample_for_review or []:
        print(f"  src: {pair['source']}")
        print(f"  tgt: {pair['target']}")
        print()


if __name__ == '__main__':
    main()
