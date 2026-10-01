"""Verify the D-SARI port -- by differential test against the upstream implementation.

D-SARI carries the whole document-level claim. This downloads the upstream `D_SARI.py` from
the paper's own repository, calls both implementations on the same inputs, and asserts they
agree to full float precision.

    python remote/checks/test_d_sari.py                 # from research/
    python remote/checks/test_d_sari.py --n 200         # more sampled documents
    python remote/checks/test_d_sari.py --offline       # invariants only, no download

Exit codes: 0 = everything checked passed. 1 = a mismatch or a broken invariant. The final
line always states whether the upstream comparison actually ran, because "invariants pass"
and "verified against upstream" are different claims and preflight.py reports which one you
have.

Two documented quirks are deliberately preserved by the port and must therefore also be
preserved here: the "delete" n-gram score returns precision only rather than an F1 like
keep/add do, and one intermediate recall term is computed but unused. They look like bugs
and are kept so scores stay comparable to the metric as the literature uses it.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import random
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

RESEARCH = Path(__file__).resolve().parent.parent.parent
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The paper's repo has moved files around; try each plausible path.
UPSTREAM_CANDIDATES = [
    'https://raw.githubusercontent.com/RLSNLP/Document-level-text-simplification/main/D_SARI.py',
    'https://raw.githubusercontent.com/RLSNLP/Document-level-text-simplification/master/D_SARI.py',
    'https://raw.githubusercontent.com/RLSNLP/Document-level-text-simplification/main/Metric/D_SARI.py',
    'https://raw.githubusercontent.com/RLSNLP/Document-level-text-simplification/main/metric/D_SARI.py',
    'https://raw.githubusercontent.com/RLSNLP/Document-level-text-simplification/main/evaluation/D_SARI.py',
]

CACHE = RESEARCH / 'scratch' / 'd_sari_upstream' / 'D_SARI.py'

FAILURES: List[str] = []
CHECKS_RUN = 0


def check(condition: bool, label: str, detail: str = '') -> None:
    global CHECKS_RUN
    CHECKS_RUN += 1
    if condition:
        print(f'  PASS  {label}')
    else:
        print(f'  FAIL  {label}' + (f' -- {detail}' if detail else ''))
        FAILURES.append(label)


# the port under test
def load_port() -> Callable[[str, str, List[str]], float]:
    """Import `d_sari_document` from evaluate_document.py without running its CLI."""
    spec = importlib.util.spec_from_file_location(
        '_evaluate_document', RESEARCH / 'evaluate_document.py'
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load evaluate_document.py from {RESEARCH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.d_sari_document


# upstream
def fetch_upstream() -> Optional[Path]:
    if CACHE.exists():
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for url in UPSTREAM_CANDIDATES:
        try:
            print(f'  fetching {url}')
            with urllib.request.urlopen(url, timeout=30) as response:
                body = response.read().decode('utf-8')
            CACHE.write_text(body)
            print(f'  cached -> {CACHE.relative_to(RESEARCH)}')
            return CACHE
        except urllib.error.HTTPError as exc:
            print(f'    {exc.code} {exc.reason}')
        except Exception as exc:
            print(f'    {type(exc).__name__}: {exc}')
    return None


def load_upstream(path: Path) -> Tuple[Optional[Callable], str]:
    """Import upstream and find its per-document scoring function.

    Upstream may run example code or read files at module scope, so the import is guarded.
    The entry point is found by name, then signature: copies call it `D_SARIsent`, `D_SARI`,
    `SARIsent` or similar.
    """
    spec = importlib.util.spec_from_file_location('_upstream_d_sari', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load an upstream D-SARI module from {path}')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return (
            None,
            f'upstream module did not import cleanly: {type(exc).__name__}: {exc}',
        )

    preferred = ('D_SARIsent', 'D_SARI_sent', 'D_SARI', 'SARIsent', 'd_sari')
    for name in preferred:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn, name

    # Fall back to any 3-argument module-level function.
    for name, obj in vars(module).items():
        if (
            isinstance(obj, types.FunctionType)
            and len(inspect.signature(obj).parameters) == 3
        ):
            return obj, name
    return None, 'no 3-argument scoring function found in upstream module'


def call_upstream(fn: Callable, source: str, candidate: str, references: List[str]):
    """Try references as a list, then as a single string; upstream copies differ.

    A signature difference is reported as such, not as a numerical mismatch.
    """
    attempts = (
        (source, candidate, references),
        (source, candidate, references[0] if references else ''),
    )
    errors = []
    for args in attempts:
        try:
            value = fn(*args)
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
            continue
        if isinstance(value, (int, float)):
            return float(value), None
        if (
            isinstance(value, (tuple, list))
            and value
            and isinstance(value[0], (int, float))
        ):
            # some copies return (d_sari, keep, del, add)
            return float(value[0]), None
    return None, '; '.join(errors) or 'returned no numeric value'


# fixtures
EDGE_CASES: List[Tuple[str, str, str, List[str]]] = [
    (
        'identity: prediction equals the reference',
        'the village had a population of 13,579 people as of the 2010 census .',
        'the village had a population of 13,579 people .',
        ['the village had a population of 13,579 people .'],
    ),
    (
        'copy: prediction equals the source',
        'the village had a population of 13,579 people as of the 2010 census .',
        'the village had a population of 13,579 people as of the 2010 census .',
        ['the village had a population of 13,579 people .'],
    ),
    (
        'over-deletion: a long source reduced to one clause',
        'achtkarspelen is a municipality in the province of friesland in the northern '
        'netherlands . it has a population of about 28,000 people . the municipality was '
        'formed in 1851 .',
        'achtkarspelen is a municipality .',
        [
            'achtkarspelen is a municipality in friesland . about 28,000 people live there .'
        ],
    ),
    (
        'sentence split: one sentence becomes two',
        'the commission determined that the allegations were unsubstantiated and dismissed '
        'the complaint .',
        'the commission looked at the claims . it found no proof and said no .',
        ['the group looked at the claims . they found no proof , so they said no .'],
    ),
    (
        'single token prediction',
        'typesetting is the composition of text by means of arranging physical types .',
        'typesetting .',
        ['typesetting is arranging letters to make text .'],
    ),
    (
        'multiple references',
        'the string can vibrate in different modes .',
        'the string can move in different ways .',
        [
            'the string can move in different ways .',
            'a string vibrates in several modes .',
        ],
    ),
]


def sampled_documents(n: int, seed: int = 1234):
    """Real (source, reference) pairs from D-Wikipedia's test split, with synthetic predictions.

    The length and sentence-count penalties only get exercised on real document shapes.
    Predictions are mechanical mutations of reference/source: a spread of keep/delete/add
    behaviour without a model.
    """
    from dataio import load_d_wikipedia

    data = load_d_wikipedia(scope='reduced')
    sources = data['test']['source']
    targets = data['test']['target']
    rng = random.Random(seed)
    indices = rng.sample(range(len(sources)), min(n, len(sources)))

    cases = []
    for i in indices:
        source, reference = sources[i], targets[i]
        words = reference.split(' ')
        variants = {
            'reference': reference,
            'source_copy': source,
            'truncated_half': ' '.join(words[: max(1, len(words) // 2)]),
            'dropped_every_fifth': ' '.join(w for j, w in enumerate(words) if j % 5),
            'shuffled_tail': ' '.join(
                words[: len(words) // 2] + words[len(words) // 2 :][::-1]
            ),
        }
        kind = rng.choice(sorted(variants))
        cases.append((f'doc[{i}] {kind}', source, variants[kind], [reference]))
    return cases


# invariants
def run_invariants(port: Callable) -> None:
    print('\nInvariants (hold regardless of upstream availability)')

    for label, source, candidate, references in EDGE_CASES:
        try:
            score = port(source, candidate, references)
        except Exception as exc:
            check(
                False, f'scores without error: {label}', f'{type(exc).__name__}: {exc}'
            )
            continue
        check(
            isinstance(score, float) and 0.0 <= score <= 1.0 and not math.isnan(score),
            f'in [0,1] and finite: {label}',
            f'got {score!r}',
        )

    # Reference must beat source copy; an inverted penalty term would pass the bounds checks.
    _, src, _, refs = EDGE_CASES[0]
    as_reference = port(src, refs[0], refs)
    as_copy = port(src, src, refs)
    check(
        as_reference > as_copy,
        'reproducing the reference scores above copying the source',
        f'reference={as_reference:.6f} copy={as_copy:.6f}',
    )

    check(
        port(src, refs[0], refs) == as_reference,
        'deterministic across repeated calls',
    )

    # sentence-count penalty: same words, different segmentation
    long_source = (
        'the group looked at the claims and found no proof and dismissed the complaint '
        'entirely after reviewing the documents .'
    )
    reference = 'the group looked at the claims . it found no proof . it said no .'
    matched = port(long_source, reference, [reference])
    one_sentence = port(long_source, reference.replace(' . ', ' and '), [reference])
    check(
        matched > one_sentence,
        "sentence-count penalty favours matching the reference's segmentation",
        f'matched={matched:.6f} single-sentence={one_sentence:.6f}',
    )


# differential test
def run_differential(port: Callable, n_docs: int) -> bool:
    print('\nDifferential test against the upstream implementation')
    path = fetch_upstream()
    if path is None:
        print(
            '  SKIP  upstream D_SARI.py could not be downloaded from any candidate path.\n'
            "        Fetch it by hand from the paper's repo\n"
            '        (RLSNLP/Document-level-text-simplification), save it to\n'
            f'        {CACHE.relative_to(RESEARCH)} and rerun. Until then the port is\n'
            "        checked only against invariants, and §2.6's 'numerically identical'\n"
            '        claim is not backed by anything in this repository.'
        )
        return False

    upstream, name = load_upstream(path)
    if upstream is None:
        print(f'  SKIP  {name}')
        return False
    print(f'  using upstream function: {name}()')

    cases = list(EDGE_CASES)
    try:
        cases += sampled_documents(n_docs)
    except Exception as exc:
        print(
            f'  note: could not sample D-Wikipedia documents ({exc}); edge cases only'
        )

    mismatches = 0
    compared = 0
    signature_errors = 0
    worst = (0.0, '')
    for label, source, candidate, references in cases:
        theirs, err = call_upstream(upstream, source, candidate, references)
        if theirs is None:
            signature_errors += 1
            if signature_errors <= 2:
                print(f'  note: upstream refused {label}: {err}')
            continue
        ours = port(source, candidate, references)
        compared += 1
        # exact comparison first; a tiny delta is a different problem from a wrong formula
        if ours != theirs:
            delta = abs(ours - theirs)
            if delta > worst[0]:
                worst = (delta, label)
            if math.isclose(ours, theirs, rel_tol=0, abs_tol=1e-12):
                continue  # float association order, not a behavioural difference
            mismatches += 1
            if mismatches <= 5:
                print(
                    f'  FAIL  {label}: ours={ours!r} upstream={theirs!r} delta={delta:.3e}'
                )

    if compared == 0:
        print(
            '  SKIP  no case could be scored by both implementations (signature mismatch)'
        )
        return False

    check(
        mismatches == 0,
        f'bit-identical to upstream across {compared} cases',
        f'{mismatches} mismatch(es), worst delta {worst[0]:.3e} on {worst[1]}',
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--n', type=int, default=60, help='D-Wikipedia test documents to sample'
    )
    ap.add_argument(
        '--offline', action='store_true', help='invariants only; skip the download'
    )
    args = ap.parse_args()

    print('D-SARI port verification')
    port = load_port()
    run_invariants(port)
    verified = False if args.offline else run_differential(port, args.n)

    print(f'\n{CHECKS_RUN} checks, {len(FAILURES)} failed')
    print(
        'REFERENCE VERIFICATION: upstream comparison PASSED'
        if verified and not FAILURES
        else (
            'REFERENCE VERIFICATION: ABSENT (invariants only -- see the note above)'
            if not verified
            else 'REFERENCE VERIFICATION: upstream comparison FAILED'
        )
    )
    raise SystemExit(1 if FAILURES else 0)


if __name__ == '__main__':
    main()
