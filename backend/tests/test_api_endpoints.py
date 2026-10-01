"""The backend through its HTTP surface: /simplify, /health, and the two cache
endpoints, plus the seq2seq internals that only show up from the request path.

The checkpoints are faked at the `transformers` level (a tokenizer that remembers its
last batch, a model that "generates" indices into it), so the whole suite runs offline
in seconds and never downloads weights. Everything around that fake is the real code --
splitting, batching, the cache and its page index, link extraction, the document
normalize/de-normalize round trip, and all three output guards.

The prompted-LLM path has its own file (test_ollama_backend.py); this one is the
seq2seq side and the endpoints both methods share.
"""

import asyncio
import logging
import sys
import threading
import time
from collections import OrderedDict
from typing import cast

import main
import pytest
import vocabulary
from fastapi.testclient import TestClient


class FakeTokenizer:
    """Stand-in for a HF tokenizer. Remembers the last batch of texts it was
    called with so `decode` can map a fake token id back to a deterministic
    "simplified" string, without touching real model weights.
    """

    def __init__(self):
        self._last_texts = []

    def __call__(self, texts, **kwargs):
        self._last_texts = texts
        return {'texts': texts}

    def decode(self, token_id, skip_special_tokens=True):
        return f'{self._last_texts[token_id]} [SIMPLIFIED]'


class FakeModel:
    """Stand-in for a HF seq2seq model. "Generates" one fake token id (just
    an index into the input batch) per input text.
    """

    def eval(self):
        return self

    def generate(self, texts, max_length=512):
        return list(range(len(texts)))


@pytest.fixture
def fake_model(monkeypatch):
    """Patch the HF loading calls used inside main.lifespan() so tests never
    download or run the real BART weights. Only the 'online' model loads
    (SIMPLIFIER_MODEL_FINETUNED and SIMPLIFIER_MODEL_DOCUMENT are explicitly forced
    empty, since both have truthy defaults) so tests are deterministic
    regardless of what's set in the developer's own shell environment.
    """
    monkeypatch.setenv('SIMPLIFIER_MODEL_FINETUNED', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_DOCUMENT', '')
    tokenizer = FakeTokenizer()
    model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name):
            return model

    monkeypatch.setattr('transformers.AutoTokenizer', FakeAutoTokenizer)
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM', FakeAutoModel)
    return tokenizer, model


@pytest.fixture
def client(fake_model):
    """TestClient that runs the app's real lifespan (model load + batch
    worker) against the faked tokenizer/model above.
    """
    with TestClient(main.app) as c:
        yield c


def online_simplifier():
    """The baseline simplifier, narrowed to the concrete seq2seq class.

    `main.simplifiers` is typed against the `Simplifier` protocol, which deliberately
    does not expose `tokenizer` -- that is an implementation detail of the seq2seq
    backend and meaningless for the Ollama-backed one. Tests that monkeypatch decoding
    are reaching past the protocol on purpose, so the narrowing is stated once here
    rather than silently assumed at each call site.
    """
    return cast(main.Seq2SeqSimplifier, main.simplifiers['online'])


def test_simplify_empty_text_returns_400(client):
    # Only an empty string reaches the handler's "Missing 'text' field" guard; an
    # omitted field fails Pydantic validation (422) first.
    response = client.post('/simplify', json={'text': ''})
    assert response.status_code == 400


def test_simplify_returns_503_before_model_load(monkeypatch):
    monkeypatch.setattr(main, 'simplifiers', {})
    # not entered as a context manager, so lifespan (and thus model
    # "loading") never runs — simplifiers stays empty regardless.
    client = TestClient(main.app)
    response = client.post('/simplify', json={'text': 'hello there'})
    assert response.status_code == 503


def test_health_reports_model_loaded(client):
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ok'
    assert body['model_loaded'] is True
    assert body['models_loaded'] == ['online']
    assert body['model_names'] == {'online': 'eilamc14/bart-large-text-simplification'}
    # a seq2seq checkpoint has no audience knob
    assert body['audience_models'] == []
    # the extension collects page text differently per granularity
    assert body['granularities'] == {
        'online': vocabulary.GRANULARITY_SENTENCE_BY_SENTENCE
    }
    # served so the extension needn't keep its own copy of the mapping
    assert body['methods'] == {'online': vocabulary.METHOD_FINE_TUNED_SEQ2SEQ}
    # per model; None for a sentence model
    assert body['document_max_tokens'] == {'online': None}


def test_health_reports_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, 'simplifiers', {})
    client = TestClient(main.app)
    response = client.get('/health')
    assert response.status_code == 200
    body = response.json()
    assert body['model_loaded'] is False
    assert body['models_loaded'] == []
    assert body['model_names'] == {}
    assert body['audience_models'] == []
    # static config, served even with no model up, so the dropdown can be built early
    assert body['audiences']


def test_simplify_rejects_unconfigured_model(client):
    response = client.post(
        '/simplify', json={'text': 'hello there', 'model': 'finetuned'}
    )
    assert response.status_code == 400
    assert 'finetuned' in response.json()['detail']


def test_looks_like_hallucination_true_for_short_unrelated_output():
    assert main._looks_like_hallucination('Published', 'Other websites') is True


def test_looks_like_hallucination_false_when_words_overlap():
    assert main._looks_like_hallucination('By', 'Written by') is False


def test_looks_like_hallucination_false_for_longer_input():
    original = 'This is a longer sentence with several words in it'
    assert (
        main._looks_like_hallucination(original, 'Something totally unrelated') is False
    )


def test_corpus_artifact_caught_regardless_of_input_length():
    """The case the short-input rule above misses: a four-word heading, well past
    its three-word ceiling, answered with the corpus's own boilerplate."""
    assert (
        main._looks_like_corpus_artifact(
            'Municipal recycling policy revisions', 'Other websites'
        )
        is True
    )
    assert (
        main._looks_like_corpus_artifact(
            'This is a much longer sentence with a good many ordinary words in it.',
            'Other websites.',
        )
        is True
    )


def test_corpus_artifact_ignores_the_phrase_inside_a_real_simplification():
    """Matched as the whole answer, not searched for: a sentence that talks about
    other websites is a simplification like any other."""
    assert (
        main._looks_like_corpus_artifact(
            'The page also enumerates supplementary external resources.',
            'The page also lists other websites you can visit.',
        )
        is False
    )


def test_corpus_artifact_catches_other_pages_the_overlap_test_cannot_see():
    """The four cases that reached the page before "other pages" joined the deny-list.

    Each defeats ``_looks_like_hallucination`` for one of two opposite reasons: the input
    shares the word "pages" with the output, so the word sets are not disjoint; or it runs
    past that rule's three-word ceiling. Kept as explicit rows because the phrase is the
    same in all four and only the reason it slipped differs.
    """
    for source in (
        'Helpful pages',
        'Popular pages',
        'Special pages',
        'You may also like',
    ):
        assert main._looks_like_hallucination(source, 'Other pages') is False, source
        assert main._looks_like_corpus_artifact(source, 'Other pages') is True, source


def test_corpus_artifact_still_allows_a_real_simplification_naming_pages():
    """The deny-list matches whole answers only, so prose that mentions other pages is
    left alone -- the same guarantee the "other websites" case already had."""
    assert (
        main._looks_like_corpus_artifact(
            'The document enumerates supplementary pages.',
            'The document lists other pages you can read.',
        )
        is False
    )


def test_corpus_artifact_leaves_the_phrase_alone_as_an_input():
    """Nothing was fabricated when the input is the heading itself -- output equal
    to the input is reported as unchanged, not as a rejected generation."""
    assert main._looks_like_corpus_artifact('Other websites', 'Other websites') is False
    assert main._looks_like_corpus_artifact('Other pages', 'Other pages') is False


class HallucinatingTokenizer:
    """Stand-in that mimics the observed real-world bug: several unrelated
    short inputs ("Published", "on", "By") all came back as the same
    fabricated phrase ("Other websites"), regardless of what was asked.
    """

    def __call__(self, texts, **kwargs):
        return {'texts': texts}

    def decode(self, token_id, skip_special_tokens=True):
        return 'Other websites'


class HallucinatingModel:
    def eval(self):
        return self

    def generate(self, texts, max_length=512):
        return list(range(len(texts)))


@pytest.fixture
def hallucinating_client(monkeypatch):
    tokenizer = HallucinatingTokenizer()
    model = HallucinatingModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name):
            return model

    monkeypatch.setattr('transformers.AutoTokenizer', FakeAutoTokenizer)
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM', FakeAutoModel)
    with TestClient(main.app) as c:
        yield c


def test_simplify_falls_back_to_original_on_short_input_hallucination(
    hallucinating_client,
):
    response = hallucinating_client.post('/simplify', json={'text': 'Published'})
    assert response.status_code == 200
    assert response.json()['simplified'] == 'Published'


def test_simplify_falls_back_on_corpus_boilerplate_for_a_longer_input(
    hallucinating_client,
):
    """The short-input guard passes this one (the input has plenty of words), and the
    change guard passes it too (words did change) -- the deny-list is what catches it,
    and it has to be reachable through the request path, not just as a predicate."""
    text = 'This is a long enough sentence to not trigger the guard.'
    response = hallucinating_client.post('/simplify', json={'text': text})
    assert response.status_code == 200
    body = response.json()
    assert body['simplified'] == text
    assert body['fallback_reason'] == 'corpus_artifact'
    assert body['model_result'] == 'Other websites'


def test_simplify_keeps_unrelated_output_that_is_not_corpus_boilerplate(
    hallucinating_client, monkeypatch
):
    """The deny-list is a deny-list, not a general relatedness test: an unrelated
    rewrite of a full sentence is still served, since judging one is the offline
    evaluation's job rather than a request-time heuristic's."""
    monkeypatch.setattr(
        online_simplifier().tokenizer,
        'decode',
        lambda token_id, skip_special_tokens=True: 'Something else entirely.',
    )
    text = 'This is a long enough sentence to not trigger the guard.'
    body = hallucinating_client.post('/simplify', json={'text': text}).json()
    assert body['simplified'] == 'Something else entirely.'
    assert body['fallback_reason'] is None


def test_simplify_cache_hit_on_second_call(client):
    text = 'This sentence is long enough to be simplified for the test.'

    first = client.post('/simplify', json={'text': text})
    assert first.status_code == 200
    assert first.json()['cached'] is False

    second = client.post('/simplify', json={'text': text})
    assert second.status_code == 200
    assert second.json()['cached'] is True
    assert second.json()['simplified'] == first.json()['simplified']


def test_clear_cache_makes_the_next_identical_request_run_the_model_again(client):
    """What the extension's "Clear cache" button is for: after it, a repeat of
    already-simplified text is a real generation again rather than a cache hit."""
    text = 'This sentence is long enough to be simplified for the test.'

    assert client.post('/simplify', json={'text': text}).json()['cached'] is False
    assert client.post('/simplify', json={'text': text}).json()['cached'] is True

    cleared = client.post('/cache/clear')
    assert cleared.status_code == 200
    assert cleared.json()['cleared'] == 1
    assert len(main.cache) == 0

    assert client.post('/simplify', json={'text': text}).json()['cached'] is False


def test_clear_cache_on_an_empty_cache_reports_nothing_cleared(client):
    assert client.post('/cache/clear').json() == {'cleared': 0}


def test_clear_cache_also_forgets_which_pages_held_what(client):
    """The page index describes the cache, so the two are emptied together. Left
    behind, it would claim entries that no longer exist and a later page deletion
    would report dropping them a second time."""
    client.post(
        '/simplify', json={'text': 'This sentence belongs to one page.', 'page': 'p1'}
    )
    assert main.cache_pages['p1']

    client.post('/cache/clear')
    assert main.cache_pages == {}
    assert main.cache_key_pages == {}


def test_page_is_not_part_of_the_cache_key(client):
    """Two pages quoting the same sentence share one generation -- the page field is
    bookkeeping, not a cache dimension. Were it part of the key, every page would
    re-run the model over text the last one had already simplified."""
    text = 'This sentence is long enough to be simplified for the test.'

    assert (
        client.post('/simplify', json={'text': text, 'page': 'p1'}).json()['cached']
        is False
    )
    assert (
        client.post('/simplify', json={'text': text, 'page': 'p2'}).json()['cached']
        is True
    )
    assert len(main.cache) == 1


def test_deleting_a_page_drops_only_that_page_s_cached_units(client):
    """What the extension calls when a page falls out of its 20-page History window,
    and when one page's log entry is deleted: that page's cached work goes, the rest
    of the cache stays."""
    kept = 'This sentence stays cached because its page stays.'
    dropped = 'This sentence goes because its own page is deleted.'
    client.post('/simplify', json={'text': kept, 'page': 'keep-me'})
    client.post('/simplify', json={'text': dropped, 'page': 'drop-me'})
    assert len(main.cache) == 2

    result = client.post('/cache/pages/delete', json={'pages': ['drop-me']})
    assert result.status_code == 200
    assert result.json() == {'dropped': 1, 'pages': 1}

    assert (
        client.post('/simplify', json={'text': kept, 'page': 'keep-me'}).json()[
            'cached'
        ]
        is True
    )
    assert (
        client.post('/simplify', json={'text': dropped, 'page': 'drop-me'}).json()[
            'cached'
        ]
        is False
    )


def test_a_shared_cache_entry_survives_until_its_last_page_goes(client):
    """A sentence two pages both contain is one generation. Dropping it when the first
    of them is deleted would make the second page's repeat visit run the model again
    over text that is still very much in the log."""
    text = 'Both of these pages contain exactly this sentence.'
    client.post('/simplify', json={'text': text, 'page': 'first'})
    client.post('/simplify', json={'text': text, 'page': 'second'})

    assert (
        client.post('/cache/pages/delete', json={'pages': ['first']}).json()['dropped']
        == 0
    )
    assert (
        client.post('/simplify', json={'text': text, 'page': 'second'}).json()['cached']
        is True
    )

    assert (
        client.post('/cache/pages/delete', json={'pages': ['second']}).json()['dropped']
        == 1
    )
    assert len(main.cache) == 0


def test_a_page_that_only_read_the_cache_still_holds_what_it_read(client):
    """Attribution follows cache *hits* as well as generations. A page whose whole run
    came back from the cache depends on those entries exactly as much as the page that
    produced them -- so deleting the producer must not take them away."""
    text = 'The second page gets this sentence entirely from the cache.'
    client.post('/simplify', json={'text': text, 'page': 'producer'})
    assert (
        client.post('/simplify', json={'text': text, 'page': 'reader'}).json()['cached']
        is True
    )

    client.post('/cache/pages/delete', json={'pages': ['producer']})
    assert (
        client.post('/simplify', json={'text': text, 'page': 'reader'}).json()['cached']
        is True
    )


def test_unowned_cache_entries_are_not_dropped_by_page_deletion(client):
    """The toggle preflight and History's comparison runs send no page id. They belong
    to no page, so no page's deletion can take them, and CACHE_MAX's LRU governs them
    as it did before any of this existed."""
    text = 'This request named no page at all.'
    client.post('/simplify', json={'text': text})
    assert main.cache_key_pages == {}

    client.post('/cache/pages/delete', json={'pages': ['anything', 'at', 'all']})
    assert client.post('/simplify', json={'text': text}).json()['cached'] is True


def test_deleting_pages_that_hold_nothing_is_not_an_error(client):
    """A page simplified before this process started, or one whose every unit was too
    short to send, holds nothing. The two counts are reported separately so a caller
    can tell "nothing to drop" from "dropped nothing"."""
    result = client.post('/cache/pages/delete', json={'pages': ['never-seen']})
    assert result.status_code == 200
    assert result.json() == {'dropped': 0, 'pages': 0}


def test_page_deletion_covers_every_sentence_of_a_multi_sentence_page(client):
    """A sentence-granularity page caches one entry per sentence, so its deletion has
    to reach all of them rather than only the unit the request was made of."""
    text = 'This is the first sentence of the page. This is the second one of them.'
    client.post('/simplify', json={'text': text, 'page': 'p1'})
    assert len(main.cache) == 2

    assert (
        client.post('/cache/pages/delete', json={'pages': ['p1']}).json()['dropped']
        == 2
    )
    assert len(main.cache) == 0


def test_lru_eviction_keeps_the_page_index_in_step(client, monkeypatch):
    """An entry pushed out under CACHE_MAX pressure must leave the index too, or its
    page would keep naming it and a later deletion would count it as dropped."""
    monkeypatch.setattr(main, 'CACHE_MAX', 1)
    client.post(
        '/simplify',
        json={'text': 'The first sentence to be cached here.', 'page': 'p1'},
    )
    client.post(
        '/simplify',
        json={'text': 'The second sentence, which evicts the first.', 'page': 'p1'},
    )

    assert len(main.cache) == 1
    assert main.cache_pages['p1'] == set(main.cache_key_pages)
    assert (
        client.post('/cache/pages/delete', json={'pages': ['p1']}).json()['dropped']
        == 1
    )


def test_split_sentences_splits_on_sentence_boundaries():
    text = 'This is sentence one. This is sentence two.'
    assert main.split_sentences(text) == [
        'This is sentence one.',
        'This is sentence two.',
    ]


def test_split_sentences_keeps_abbreviations_together():
    # a naive regex split on ". " would wrongly break after "Dr." and "U.S."
    text = 'Dr. Smith went to the U.S. in 1998. He liked it.'
    assert main.split_sentences(text) == [
        'Dr. Smith went to the U.S. in 1998.',
        'He liked it.',
    ]


def test_split_sentences_falls_back_to_whole_text_when_no_boundary():
    assert main.split_sentences('no terminal punctuation here') == [
        'no terminal punctuation here'
    ]


def test_simplify_splits_multi_sentence_text_and_recombines(client):
    text = 'This is sentence one. This is sentence two.'
    response = client.post('/simplify', json={'text': text})
    assert response.status_code == 200
    # each sentence simplified independently (the fakes append "[SIMPLIFIED]" per
    # input) and rejoined with the original separators
    assert response.json()['simplified'] == (
        'This is sentence one. [SIMPLIFIED] This is sentence two. [SIMPLIFIED]'
    )


def test_simplify_preserves_linebreaks_and_tabs_between_sentences(client):
    text = 'A.\nB.\tC.'
    response = client.post('/simplify', json={'text': text})
    assert response.status_code == 200
    assert (
        response.json()['simplified']
        == 'A. [SIMPLIFIED]\nB. [SIMPLIFIED]\tC. [SIMPLIFIED]'
    )


def test_cosmetic_only_output_is_served_as_the_original(client, monkeypatch):
    """The change guard sits on the request path, not just in its own module.

    The fake model appends " [SIMPLIFIED]" to everything, which is a real edit as
    far as the guard is concerned -- so this one is made to return output that
    differs from the input by a single full stop, the case the guard exists for.
    """
    original = 'A sentence carrying rather more than a handful of ordinary words.'
    monkeypatch.setattr(
        online_simplifier().tokenizer,
        'decode',
        lambda token_id, skip_special_tokens=True: original.rstrip('.'),
    )
    response = client.post('/simplify', json={'text': original})
    assert response.status_code == 200
    body = response.json()
    # the model's own output is discarded and the input served in its place, which
    # is how every other rejected output is handled
    assert body['simplified'] == original
    assert body['fallback_reason'] == 'no_meaningful_change'


# --- model_result / sentence_split: the input -> model result -> output chain ---


def test_model_result_matches_the_output_when_nothing_was_rejected(client):
    text = 'This sentence is long enough to avoid the hallucination guard.'
    body = client.post('/simplify', json={'text': text}).json()
    assert body['fallback_reason'] is None
    assert body['model_result'] == body['simplified']


def test_model_result_keeps_the_rejected_output(hallucinating_client):
    """The reason the field exists: what the model said is reported even where it is
    not what gets served. Without it a fallback is indistinguishable from the model
    having declined to change anything."""
    body = hallucinating_client.post('/simplify', json={'text': 'Published'}).json()
    assert body['simplified'] == 'Published'
    assert body['fallback_reason'] == 'hallucination'
    assert body['model_result'] == 'Other websites'


def test_model_result_survives_a_cache_hit(hallucinating_client):
    """A repeat of a rejected unit has to report the same chain as the first request
    -- the rejected output is cached alongside what is served, not recomputed."""
    hallucinating_client.post('/simplify', json={'text': 'Published'})
    body = hallucinating_client.post('/simplify', json={'text': 'Published'}).json()
    assert body['cached'] is True
    assert body['model_result'] == 'Other websites'
    assert body['fallback_reason'] == 'hallucination'


def test_model_result_keeps_the_output_the_change_guard_rejected(client, monkeypatch):
    """The change guard runs after the model's own guard and rejects on different
    grounds, so it gets its own check: it too must leave `model_result` alone."""
    original = 'A sentence carrying rather more than a handful of ordinary words.'
    monkeypatch.setattr(
        online_simplifier().tokenizer,
        'decode',
        lambda token_id, skip_special_tokens=True: original.rstrip('.'),
    )
    body = client.post('/simplify', json={'text': original}).json()
    assert body['simplified'] == original
    assert body['fallback_reason'] == 'no_meaningful_change'
    assert body['model_result'] == original.rstrip('.')


def test_sentence_split_reports_the_parts_that_were_rejoined(client):
    body = client.post(
        '/simplify', json={'text': 'This is sentence one. This is sentence two.'}
    ).json()
    assert body['sentence_split'] == {
        'count': 2,
        'parts': [
            'This is sentence one. [SIMPLIFIED]',
            'This is sentence two. [SIMPLIFIED]',
        ],
    }
    # the parts are the output's own pieces: rejoined with the separators the source
    # text used, they are exactly `simplified`
    assert ' '.join(body['sentence_split']['parts']) == body['simplified']


def test_sentence_split_is_absent_for_a_single_sentence(client):
    """One sentence in, one sentence out is not a split. Reporting it as a one-part
    one would make every ordinary request look like it had been taken apart."""
    body = client.post(
        '/simplify', json={'text': 'This is one single sentence.'}
    ).json()
    assert body['sentence_split'] is None


def test_sentence_split_is_absent_for_a_document_model(client):
    main.simplifiers['online'].granularity = vocabulary.GRANULARITY_WHOLE_SECTIONS
    body = client.post(
        '/simplify', json={'text': 'This is sentence one. This is sentence two.'}
    ).json()
    assert body['sentence_split'] is None


def test_simplify_caches_sentences_independently_across_requests(client):
    first = client.post(
        '/simplify', json={'text': 'This is sentence one. This is sentence two.'}
    )
    assert first.status_code == 200
    assert first.json()['cached'] is False

    # a later request for just one of those sentences on its own hits the
    # per-sentence cache entry populated by the first request.
    second = client.post('/simplify', json={'text': 'This is sentence one.'})
    assert second.status_code == 200
    assert second.json()['cached'] is True
    assert second.json()['simplified'] == 'This is sentence one. [SIMPLIFIED]'


def test_document_granularity_model_does_not_split_sentences(client):
    # Mutates the registered fake "online" simplifier instead of loading "document":
    # only simplify_text's granularity routing is under test (the document key's
    # handling is covered by the two tests below).
    main.simplifiers['online'].granularity = vocabulary.GRANULARITY_WHOLE_SECTIONS
    text = 'This is sentence one. This is sentence two.'
    response = client.post('/simplify', json={'text': text})
    assert response.status_code == 200
    # the whole block went through generate() as a single unit
    assert response.json()['simplified'] == f'{text} [SIMPLIFIED]'


def test_document_model_normalizes_input_and_denormalizes_output(monkeypatch):
    """The document checkpoint's corpus is lowercased + PTB-pre-tokenized, so
    `/simplify` has to convert into that style on the way in and back out again --
    otherwise the model sees out-of-distribution text and its raw output
    ("achtkarspelen is a municipality .") renders as broken prose on the page.
    """
    monkeypatch.setenv('SIMPLIFIER_MODEL_ONLINE', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_FINETUNED', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_DOCUMENT', 'fake-document-checkpoint')

    seen = {}

    class RecordingTokenizer:
        """Records what the model was actually fed, and echoes back a corpus-style
        (lowercase, space-before-punctuation) response like the real one does."""

        def __call__(self, texts, **kwargs):
            seen['model_input'] = texts[0]
            return {'input_ids': [[1, 2, 3]]}

        def decode(self, token_id, skip_special_tokens=True):
            return 'achtkarspelen is a municipality in friesland .'

    class FakeDocModel:
        def eval(self):
            return self

        def generate(self, input_ids=None, max_length=512, **kwargs):
            return [0]

    monkeypatch.setattr(
        'transformers.AutoTokenizer',
        type(
            'T', (), {'from_pretrained': staticmethod(lambda n: RecordingTokenizer())}
        ),
    )
    monkeypatch.setattr(
        'transformers.AutoModelForSeq2SeqLM',
        type('M', (), {'from_pretrained': staticmethod(lambda n: FakeDocModel())}),
    )

    with TestClient(main.app) as client:
        body = client.get('/health').json()
        assert body['granularities'] == {
            'document': vocabulary.GRANULARITY_WHOLE_SECTIONS
        }

        text = (
            'Achtkarspelen is a municipality in Friesland, in the northern Netherlands.'
        )
        response = client.post('/simplify', json={'text': text, 'model': 'document'})
        assert response.status_code == 200

    # input reached the model lowercased with punctuation split off
    assert seen['model_input'] == (
        'achtkarspelen is a municipality in friesland , in the northern netherlands .'
    )
    # ...and the lowercase output came back capitalized, with proper nouns restored
    # from the request text and the space before "." removed
    assert (
        response.json()['simplified'] == 'Achtkarspelen is a municipality in Friesland.'
    )


def test_document_model_receives_whole_text_without_sentence_splitting(monkeypatch):
    """A document model must get the full multi-sentence input in one piece -- it's
    trained to rewrite a whole document (deleting/merging/reordering sentences), so
    splitting it per sentence would defeat the point."""
    monkeypatch.setenv('SIMPLIFIER_MODEL_ONLINE', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_FINETUNED', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_DOCUMENT', 'fake-document-checkpoint')

    calls = []

    class RecordingTokenizer:
        def __call__(self, texts, **kwargs):
            calls.append(texts[0])
            return {'input_ids': [[1]]}

        def decode(self, token_id, skip_special_tokens=True):
            return 'short output .'

    class FakeDocModel:
        def eval(self):
            return self

        def generate(self, input_ids=None, max_length=512, **kwargs):
            return [0]

    monkeypatch.setattr(
        'transformers.AutoTokenizer',
        type(
            'T', (), {'from_pretrained': staticmethod(lambda n: RecordingTokenizer())}
        ),
    )
    monkeypatch.setattr(
        'transformers.AutoModelForSeq2SeqLM',
        type('M', (), {'from_pretrained': staticmethod(lambda n: FakeDocModel())}),
    )

    with TestClient(main.app) as client:
        client.post(
            '/simplify',
            json={
                'text': 'First sentence here. Second sentence here. Third sentence here.',
                'model': 'document',
            },
        )

    # exactly one generate call, carrying all three sentences
    assert len(calls) == 1
    assert calls[0] == (
        'first sentence here . second sentence here . third sentence here .'
    )


def test_extract_and_clean_extracts_links():
    text = 'See <a href="https://example.com">this link</a> for more.'
    clean, links = main.extract_and_clean(text)
    assert clean == 'See this link for more.'
    assert links == [{'text': 'this link', 'url': 'https://example.com'}]


def test_extract_and_clean_preserves_structure():
    text = 'Line one\n\n   Line two\t\tLine three'
    clean, links = main.extract_and_clean(text)
    assert clean == 'Line one\n\n   Line two\t\tLine three'
    assert links == []


def test_split_sentences_with_whitespace_preserves_separators():
    text = 'A.\nB.\tC.'
    assert main.split_sentences_with_whitespace(text) == [
        ('A.', '\n'),
        ('B.', '\t'),
        ('C.', ''),
    ]

    text = 'A. \n\nB. C.'
    assert main.split_sentences_with_whitespace(text) == [
        ('A.', ' \n\n'),
        ('B.', ' '),
        ('C.', ''),
    ]

    text = 'A\nB\tC'
    assert main.split_sentences_with_whitespace(text) == [
        ('A', '\n'),
        ('B', '\t'),
        ('C', ''),
    ]

    text = 'A\n\nBC'
    assert main.split_sentences_with_whitespace(text) == [('A', '\n\n'), ('BC', '')]


def test_extract_and_clean_plain_text_passthrough():
    text = 'Already a single clean sentence.'
    clean, links = main.extract_and_clean(text)
    assert clean == text
    assert links == []


async def test_batch_worker_resolves_multiple_futures(monkeypatch, fake_model):
    tokenizer, model = fake_model
    simplifier = main.Seq2SeqSimplifier('fake-model', tokenizer, model)
    monkeypatch.setattr(main, 'simplifiers', {'online': simplifier})
    monkeypatch.setattr(main, 'cache', OrderedDict())
    monkeypatch.setattr(main, 'cache_lock', asyncio.Lock())
    queue = asyncio.Queue()
    monkeypatch.setattr(main, 'batch_queues', {'online': queue})

    audience = main.DEFAULT_AUDIENCE
    loop = asyncio.get_running_loop()
    fut1 = loop.create_future()
    fut2 = loop.create_future()
    await queue.put(('first paragraph', [], audience, fut1))
    await queue.put(('second paragraph', [], audience, fut2))

    try:
        # _batch_worker() loops forever; it processes both queued items in
        # one batch (well within BATCH_TIMEOUT) then blocks on the next
        # (now-empty) queue.get() until this timeout cuts it off.
        await asyncio.wait_for(main._batch_worker('online'), timeout=0.5)
    except asyncio.TimeoutError:
        pass

    # (served text, fallback reason, model result) -- nothing was rejected here, so
    # the served text and the model's own result are the same string.
    assert fut1.done() and fut1.result() == (
        'first paragraph [SIMPLIFIED]',
        None,
        'first paragraph [SIMPLIFIED]',
    )
    assert fut2.done() and fut2.result() == (
        'second paragraph [SIMPLIFIED]',
        None,
        'second paragraph [SIMPLIFIED]',
    )
    key = ('online', audience.value, 'first paragraph')
    assert main.cache[key][0] == 'first paragraph [SIMPLIFIED]'
    assert main.cache[('online', audience.value, 'second paragraph')][0] == (
        'second paragraph [SIMPLIFIED]'
    )


def test_seq2seq_model_ignores_a_requested_audience(client):
    """Only the prompted-LLM path can act on an audience. Asking a seq2seq
    checkpoint for one is accepted (rather than erroring, which would make the
    extension's dropdown fragile) but normalized away, so the cache doesn't
    fragment across audiences that produce identical output."""
    text = 'This sentence is long enough to avoid the hallucination guard.'
    response = client.post('/simplify', json={'text': text, 'audience': 'children'})
    assert response.status_code == 200
    assert response.json()['audience'] == main.DEFAULT_AUDIENCE.value

    # ...and the differing audience still hits the same cache entry
    again = client.post('/simplify', json={'text': text, 'audience': 'low_literacy'})
    assert again.json()['cached'] is True


def test_simplify_rejects_an_unknown_audience(client):
    response = client.post(
        '/simplify', json={'text': 'hello there', 'audience': 'wizards'}
    )
    assert response.status_code == 422


def test_simplify_routes_to_requested_model_and_caches_separately(monkeypatch):
    """Two distinct fake models loaded as 'online' and 'finetuned' should produce
    different output for the same input text, and each should only cache-hit
    against requests for its own model key.
    """
    monkeypatch.setenv('SIMPLIFIER_MODEL_FINETUNED', 'fake-finetuned-checkpoint')

    class TaggingTokenizer:
        """Like FakeTokenizer, but tags decoded output with which model_id it
        was constructed for, so online/finetuned are distinguishable."""

        def __init__(self, tag):
            self.tag = tag
            self._last_texts = []

        def __call__(self, texts, **kwargs):
            self._last_texts = texts
            return {'texts': texts}

        def decode(self, token_id, skip_special_tokens=True):
            return f'{self._last_texts[token_id]} [{self.tag}]'

    class TaggingModel:
        def eval(self):
            return self

        def generate(self, texts, max_length=512):
            return list(range(len(texts)))

    ONLINE_ID = main.MODEL_ENV_CONFIG['online'][1]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return TaggingTokenizer(tag='ONLINE' if name == ONLINE_ID else 'FINETUNED')

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name):
            return TaggingModel()

    monkeypatch.setattr('transformers.AutoTokenizer', FakeAutoTokenizer)
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM', FakeAutoModel)

    with TestClient(main.app) as client:
        text = 'This sentence is long enough to avoid the hallucination guard.'

        online_resp = client.post('/simplify', json={'text': text, 'model': 'online'})
        assert online_resp.status_code == 200
        assert online_resp.json()['simplified'] == f'{text} [ONLINE]'

        finetuned_resp = client.post(
            '/simplify', json={'text': text, 'model': 'finetuned'}
        )
        assert finetuned_resp.status_code == 200
        assert finetuned_resp.json()['simplified'] == f'{text} [FINETUNED]'

        # re-requesting "online" hits the "online" entry, not "finetuned"'s: the
        # cache key is per model
        online_again = client.post('/simplify', json={'text': text, 'model': 'online'})
        assert online_again.json()['simplified'] == f'{text} [ONLINE]'
        assert online_again.json()['cached'] is True


def test_truncation_warning_names_only_the_row_that_was_truncated(caplog):
    """The 64-token warning must be about the text that actually overflowed.

    `padding=True` pads every row of a batch out to the longest one, so a row's id
    count is the batch's length rather than its own. Reading truncation off that made
    one over-long sentence report every sentence batched beside it as truncated too,
    each against its own (much shorter) char count -- a warning pointing at texts that
    were simplified in full.
    """
    long_text = 'x' * 400
    short_text = 'A short sentence.'
    encoded = {
        # what a real tokenizer hands back for these two: both rows padded out to the
        # ceiling because the first hit it, and the mask saying which tokens are real
        'input_ids': [list(range(64)), list(range(64))],
        'attention_mask': [[1] * 64, [1] * 5 + [0] * 59],
    }

    simplifier = main.Seq2SeqSimplifier.__new__(main.Seq2SeqSimplifier)
    simplifier.model_id = 'yunvs/bart-base-wikilarge-simplification'
    simplifier.max_length = 64

    with caplog.at_level(logging.WARNING):
        simplifier._warn_if_truncated([long_text, short_text], encoded)

    warnings = [
        r.getMessage() for r in caplog.records if 'token limit' in r.getMessage()
    ]
    assert len(warnings) == 1
    assert f'({len(long_text)} chars)' in warnings[0]
    assert f'({len(short_text)} chars)' not in warnings[0]


def test_truncation_warning_still_fires_without_an_attention_mask(caplog):
    """A stand-in tokenizer that returns ids alone still gets the safety net."""
    simplifier = main.Seq2SeqSimplifier.__new__(main.Seq2SeqSimplifier)
    simplifier.model_id = 'fake-model'
    simplifier.max_length = 8

    with caplog.at_level(logging.WARNING):
        simplifier._warn_if_truncated(
            ['over the ceiling'], {'input_ids': [list(range(8))]}
        )

    assert any('token limit' in r.getMessage() for r in caplog.records)


async def test_seq2seq_generation_is_serialised_onto_one_thread():
    """Two loaded checkpoints must not submit to the device from two threads at once.

    This is the MPS deadlock guard (see _SEQ2SEQ_EXECUTOR): torch's MPS backend takes
    Apple's dispatch queue and the GIL in opposite orders on the generating thread and on
    the "metal gpu stream" thread, so two checkpoints generating concurrently wedge the
    whole interpreter -- the process stays up holding its socket at ~0% CPU and answers
    nothing again. Reproduced here on this machine: one checkpoint on one thread ran 746
    batches clean, two on two threads wedged before the 25th.

    So the invariant is not "each model is serialised with itself" (which _register's
    per-key batch worker already gave us) but "all of them are serialised with each
    other". Asserted by thread identity rather than by naming the executor, since what
    breaks the fix is a call reaching *any* second thread -- `asyncio.to_thread`, which
    this used to be, is the obvious way back to two.
    """
    seen_threads = []

    class RecordingTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                'input_ids': [[1]] * len(texts),
                'attention_mask': [[1]] * len(texts),
            }

        def decode(self, token_id, skip_special_tokens=True):
            return 'a decoded sentence that is long enough to pass the guards'

    class SlowModel:
        """Holds its thread long enough that a concurrent call would have to use another."""

        def eval(self):
            return self

        def generate(self, input_ids=None, max_length=512, **kwargs):
            seen_threads.append(threading.current_thread().name)
            time.sleep(0.05)
            return [0]

    first = main.Seq2SeqSimplifier('fake-a', RecordingTokenizer(), SlowModel())
    second = main.Seq2SeqSimplifier('fake-b', RecordingTokenizer(), SlowModel())

    texts = ['Residents are requested to position their receptacles at the kerbside.']
    await asyncio.gather(
        *(
            s.generate(texts, main.DEFAULT_AUDIENCE)
            for s in (first, second)
            for _ in range(3)
        )
    )

    assert len(seen_threads) == 6, 'every generation should have reached the model'
    assert (
        len(set(seen_threads)) == 1
    ), f'generations ran on {len(set(seen_threads))} threads: {set(seen_threads)}'
    # ...and off the event loop, which is the property the threading is there for
    assert seen_threads[0] != threading.current_thread().name


def test_auto_device_does_not_pick_mps(monkeypatch):
    """`auto` must not reach for Metal, even where Metal is available.

    Measured 2026-08-26 through four fresh uvicorn processes, CPU is faster on every phase -- most
    of all at the batch width the extension actually issues (80 against 136 ms/sentence
    with eight in flight), because a batch of eight fills the cores while a model this
    small never fills the GPU. It is also the only device bit-identical to
    research/results/, and MPS carries the GIL deadlock _SEQ2SEQ_EXECUTOR exists for.
    """
    monkeypatch.delenv('SIMPLIFIER_DEVICE', raising=False)

    class FakeBackends:
        class mps:
            @staticmethod
            def is_available():
                return True

    fake_torch = type(
        'torch',
        (),
        {
            'backends': FakeBackends,
            'cuda': type('cuda', (), {'is_available': staticmethod(lambda: False)}),
        },
    )
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    assert main._resolve_serving_device() == 'cpu'


def test_auto_device_still_picks_cuda(monkeypatch):
    """CUDA is untouched: nothing here was measured on it, and the deadlock is MPS's."""
    monkeypatch.delenv('SIMPLIFIER_DEVICE', raising=False)
    fake_torch = type(
        'torch',
        (),
        {
            'backends': type('backends', (), {}),
            'cuda': type('cuda', (), {'is_available': staticmethod(lambda: True)}),
        },
    )
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    assert main._resolve_serving_device() == 'cuda'


def test_mps_is_still_selectable_by_asking(monkeypatch):
    """Turning off the *default* must not take the option away -- profiling wants it, and
    so does hardware where the balance comes out differently."""
    monkeypatch.setenv('SIMPLIFIER_DEVICE', 'mps')
    assert main._resolve_serving_device() == 'mps'


@pytest.mark.parametrize('value', ['', '   '])
def test_blank_device_is_treated_as_unset(monkeypatch, value):
    """A blank SIMPLIFIER_DEVICE must mean "decide for me", not "device ''".

    `SIMPLIFIER_DEVICE="$SOMETHING"` with SOMETHING unset is one shell expansion away, and
    it used to reach torch as an empty device string: every checkpoint failed to load with
    "Device string must not be empty", and the backend came up answering /health with
    models_loaded == [] -- a server that is up and serves nothing.
    """
    monkeypatch.setenv('SIMPLIFIER_DEVICE', value)
    fake_torch = type(
        'torch',
        (),
        {
            'backends': type('backends', (), {}),
            'cuda': type('cuda', (), {'is_available': staticmethod(lambda: False)}),
        },
    )
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    assert main._resolve_serving_device() == 'cpu'
