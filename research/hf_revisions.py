"""Hugging Face Hub revisions, pinned to what the reported runs actually resolved.

`load_dataset("x")` and `from_pretrained("x")` fetch whatever `main` points at on the
day they run, and a dataset or checkpoint can be re-uploaded under the same name with
no trace in a results file (the easse commit pin in requirements.txt solves the same
problem one layer up).

Provenance: nothing in research/results/ records a revision (only
`source: "huggingface_hub"` and the model id). These SHAs were recovered on 2026-09-16
from the local Hugging Face cache (~/.cache/huggingface/hub/<repo>/refs/main), which
stores the commit each download resolved to. All were downloaded between 2026-07-10 and
2026-08-24, before the 2026-08-26 results freeze, and verified on 2026-09-16 to equal
the Hub's current HEAD. Pinning is therefore a no-op today; it only stops a future
re-upload from silently changing a number.

Caveat: this is the cache on the laptop the thesis is written on. The GPU runs
happened on beet with its own cache, and nothing recorded what that resolved. For these
repos (stable public datasets and this project's own uploads) the two agreeing is the
strong expectation, not a proven fact.

Usage -- look the id up rather than hardcoding at the call site:

    from_pretrained(model_id, revision=HF_REVISIONS.get(model_id))

`revision=None` is from_pretrained's and load_dataset's own default, so an id not in
this table (a local checkpoint path, a model added later) behaves as before.

Not pinned: FacebookAI/roberta-large. bert-score downloads it internally and takes no
revision argument from us. Its cached revision is
722cf37b1afa9454edce342e7895e588b6ff1d59, recorded here for the write-up.
"""

from __future__ import annotations

from typing import Dict

HF_REVISIONS: Dict[str, str] = {
    # --- datasets ---
    # ASSET (Alva-Manchego et al., 2020): 10 references per sentence, the sentence-level
    # test set behind every SARI/BLEU number.
    'facebook/asset': 'c7f2fa4bae55ae656091805d4416c1374582bb4e',
    # WikiLarge, the cleaned re-upload this project trains the sentence models on.
    'eilamc14/wikilarge-clean': '216fedb399e10141b390c8b89039737c934d43be',
    # --- models ---
    # The starting checkpoint for every fine-tuning run.
    'facebook/bart-base': 'aadd2ab0ae0c8268c7c9693540e9904811f36177',
    # This project's own uploads, served by the backend under the llm_*/seq2seq keys.
    'yunvs/bart-base-dwikipedia-simplification-full': (
        'd5d1ed8b2809d9dd982775d27d7765f5b5ed271c'
    ),
    'yunvs/bart-base-wikilarge-simplification': (
        '050112d6e53fbde261fdc120ce19a687a9e13086'
    ),
    # The published comparison checkpoint the results are measured against.
    'eilamc14/bart-large-text-simplification': (
        'ad691172de3ad6e8d5f21b71c10558c54886718f'
    ),
}
