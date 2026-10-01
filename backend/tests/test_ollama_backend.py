"""Integration tests for the Ollama-backed prompted-LLM simplifier, at both
granularities -- `llm_7b` (sentence) and `llm_doc_7b` (whole sections).

No real sidecar is contacted: `main._make_ollama_client` is swapped for a fake that
records every prompt it was sent, so these run in the same offline, sub-second suite
as everything else. The `disable_all_models_by_default` autouse fixture in conftest.py
turns every model off but the baseline; each test here enables the key it needs.
"""

import httpx
import main
import prompting
import pytest
import vocabulary
from fastapi.testclient import TestClient

TAG = 'qwen2.5:7b-instruct-q4_K_M'


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeOllamaClient:
    """Stand-in for `httpx.AsyncClient` pointed at a sidecar.

    `reply` maps a prompt-substring to the raw string the model should "generate";
    anything unmatched falls back to `default_reply`. Every request payload is kept in
    `posts` so tests can assert on the prompt that was actually built.
    """

    def __init__(
        self,
        tags=(TAG,),
        default_reply='The cat sat down.',
        replies=None,
        raise_on_post=None,
    ):
        self.tags = list(tags)
        self.default_reply = default_reply
        self.replies = replies or {}
        self.raise_on_post = raise_on_post
        self.posts = []
        self.closed = False

    async def get(self, url):
        assert url == '/api/tags'
        return FakeResponse({'models': [{'name': t} for t in self.tags]})

    async def post(self, url, json=None):
        assert url == '/api/generate'
        self.posts.append(json)
        if self.raise_on_post is not None:
            raise self.raise_on_post
        prompt = (json or {})['prompt']
        for needle, reply in self.replies.items():
            if needle in prompt:
                return FakeResponse({'response': reply})
        return FakeResponse({'response': self.default_reply})

    async def aclose(self):
        self.closed = True


@pytest.fixture
def fake_ollama(monkeypatch):
    """Enable the 'llm_7b' model and back it with a FakeOllamaClient.

    Returns a factory so a test can configure the fake's behaviour before the app's
    lifespan runs. The seq2seq models are forced off so `models_loaded` contains
    exactly 'llm_7b' and nothing needs the transformers fakes.
    """
    monkeypatch.setenv('SIMPLIFIER_MODEL_LLM_7B', TAG)
    monkeypatch.setenv('SIMPLIFIER_MODEL_ONLINE', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_FINETUNED', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_DOCUMENT', '')

    def make(**kwargs):
        client = FakeOllamaClient(**kwargs)
        monkeypatch.setattr(main, '_make_ollama_client', lambda: client)
        return client

    return make


def test_health_reports_llm_as_audience_capable(fake_ollama):
    fake_ollama()
    with TestClient(main.app) as client:
        body = client.get('/health').json()

    assert body['models_loaded'] == ['llm_7b']
    assert body['model_names'] == {'llm_7b': TAG}
    assert body['audience_models'] == ['llm_7b']
    assert body['default_audience'] == prompting.DEFAULT_AUDIENCE.value
    # every enum member is offered, each with a display label for the dropdown
    assert [a['value'] for a in body['audiences']] == [
        a.value for a in prompting.Audience
    ]
    assert all(a['label'] for a in body['audiences'])


def test_simplify_sends_bless_prompt_2_and_returns_sanitised_output(fake_ollama):
    ollama = fake_ollama(
        default_reply="\n  Sure! Here's a simpler version: The cat sat down."
    )
    text = 'The felid subsequently assumed a seated posture upon the floor covering.'

    with TestClient(main.app) as client:
        body = client.post('/simplify', json={'text': text, 'model': 'llm_7b'}).json()

    assert body['simplified'] == 'The cat sat down.'
    assert body['fallback_reason'] is None
    assert body['audience'] == prompting.DEFAULT_AUDIENCE.value

    sent = ollama.posts[0]
    assert sent['model'] == TAG
    assert sent['stream'] is False
    assert sent['options'] == prompting.SERVING_DECODING
    assert prompting.render_instruction() in sent['prompt']
    assert sent['prompt'].endswith(f'Complex: {text}\nSimple:')


def test_requested_audience_reaches_the_prompt(fake_ollama):
    ollama = fake_ollama()

    with TestClient(main.app) as client:
        body = client.post(
            '/simplify',
            json={
                'text': 'A complex sentence here.',
                'model': 'llm_7b',
                'audience': 'children',
            },
        ).json()

    assert body['audience'] == 'children'
    prompt = ollama.posts[0]['prompt']
    assert prompting.audience_description(prompting.Audience.CHILDREN) in prompt
    assert 'non-native speakers' not in prompt
    # the demonstrations must come from the matching pool, or they'd pull the model
    # back toward the default audience
    assert prompting.FEWSHOT_EXAMPLES[prompting.Audience.CHILDREN][0].simple in prompt


def test_cache_is_keyed_by_audience(fake_ollama):
    """Two audiences for the same sentence are two different generations; a cache
    that ignored the audience would serve the first one for both."""
    ollama = fake_ollama(
        replies={
            prompting.audience_description(prompting.Audience.CHILDREN): 'Kid version.',
            prompting.audience_description(
                prompting.Audience.GENERAL_ADULT
            ): 'Adult version.',
        }
    )
    text = 'A sufficiently complex sentence to simplify.'

    with TestClient(main.app) as client:
        first = client.post(
            '/simplify', json={'text': text, 'model': 'llm_7b', 'audience': 'children'}
        ).json()
        second = client.post(
            '/simplify',
            json={'text': text, 'model': 'llm_7b', 'audience': 'general_adult'},
        ).json()
        repeat = client.post(
            '/simplify', json={'text': text, 'model': 'llm_7b', 'audience': 'children'}
        ).json()

    assert first['simplified'] == 'Kid version.'
    assert first['cached'] is False
    assert second['simplified'] == 'Adult version.'
    assert second['cached'] is False
    # same audience again -> served from cache, no third generation
    assert repeat['simplified'] == 'Kid version.'
    assert repeat['cached'] is True
    assert len(ollama.posts) == 2


@pytest.mark.parametrize(
    'reply,expected_reason',
    [
        ("I'm sorry, I can't rewrite that.", 'refusal'),
        ('', 'empty'),
        ('Complex: something else entirely.', 'prompt_echo'),
    ],
)
def test_bad_generations_fall_back_to_the_original(fake_ollama, reply, expected_reason):
    fake_ollama(default_reply=reply)
    text = 'The felid subsequently assumed a seated posture upon the floor covering.'

    with TestClient(main.app) as client:
        body = client.post('/simplify', json={'text': text, 'model': 'llm_7b'}).json()

    assert body['simplified'] == text
    assert body['fallback_reason'] == expected_reason


def test_timeout_falls_back_to_the_original_rather_than_hanging(fake_ollama):
    fake_ollama(raise_on_post=httpx.ReadTimeout('too slow'))
    text = 'The felid subsequently assumed a seated posture upon the floor covering.'

    with TestClient(main.app) as client:
        body = client.post('/simplify', json={'text': text, 'model': 'llm_7b'}).json()

    assert body['simplified'] == text
    assert body['fallback_reason'] == 'timeout'


def test_transport_error_falls_back_to_the_original(fake_ollama):
    fake_ollama(raise_on_post=httpx.ConnectError('sidecar went away'))

    with TestClient(main.app) as client:
        body = client.post(
            '/simplify', json={'text': 'Some complex sentence.', 'model': 'llm_7b'}
        ).json()

    assert body['simplified'] == 'Some complex sentence.'
    assert body['fallback_reason'] == 'error'


def test_llm_is_not_registered_when_the_tag_is_not_pulled(fake_ollama):
    """Ollama is up but the model was never pulled -- the same clean
    "not available" state as Ollama being down, not a 500 on first use."""
    fake_ollama(tags=('some-other-model',))

    with TestClient(main.app) as client:
        assert client.get('/health').json()['models_loaded'] == []
        response = client.post(
            '/simplify', json={'text': 'hello there', 'model': 'llm_7b'}
        )

    # no model at all came up in this configuration
    assert response.status_code == 503


def test_llm_is_not_registered_when_the_sidecar_is_down(monkeypatch):
    """With Ollama unreachable, 'llm_7b' is simply absent while the seq2seq model
    still works -- so requesting it gets the existing, actionable 400 rather than
    taking the whole backend down."""
    monkeypatch.setenv('SIMPLIFIER_MODEL_LLM_7B', TAG)
    monkeypatch.setenv('SIMPLIFIER_MODEL_FINETUNED', '')
    monkeypatch.setenv('SIMPLIFIER_MODEL_DOCUMENT', '')

    class DeadClient(FakeOllamaClient):
        async def get(self, url):
            raise httpx.ConnectError('connection refused')

    monkeypatch.setattr(main, '_make_ollama_client', lambda: DeadClient())

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            self._last = texts
            return {'texts': texts}

        def decode(self, token_id, skip_special_tokens=True):
            return f'{self._last[token_id]} [SIMPLIFIED]'

    class FakeModel:
        def eval(self):
            return self

        def generate(self, texts, max_length=512):
            return list(range(len(texts)))

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name):
            return FakeModel()

    monkeypatch.setattr('transformers.AutoTokenizer', FakeAutoTokenizer)
    monkeypatch.setattr('transformers.AutoModelForSeq2SeqLM', FakeAutoModel)

    with TestClient(main.app) as client:
        assert client.get('/health').json()['models_loaded'] == ['online']
        response = client.post(
            '/simplify', json={'text': 'hello there', 'model': 'llm_7b'}
        )

    assert response.status_code == 400
    assert 'llm_7b' in response.json()['detail']


def test_sidecar_client_is_closed_on_shutdown(fake_ollama):
    ollama = fake_ollama()
    with TestClient(main.app):
        pass
    assert ollama.closed is True


async def test_register_rejects_audience_model_with_a_multi_item_batch(monkeypatch):
    """The worker renders one prompt per batch, so an audience-sensitive simplifier
    with room for more than one item could silently apply the first item's audience
    to the rest. Caught at registration rather than in production."""
    monkeypatch.setattr(main, 'simplifiers', {})
    monkeypatch.setattr(main, 'batch_queues', {})
    monkeypatch.setattr(main, 'worker_tasks', {})

    class Broken:
        model_id = 'broken'
        supports_audience = True
        batch_size = 8

        async def generate(self, texts, audience):
            return [(t, None) for t in texts]

        async def aclose(self):
            return None

    with pytest.raises(ValueError, match='batch_size=1'):
        main._register('broken', Broken())  # type: ignore[arg-type]

    assert 'broken' not in main.simplifiers


async def test_llm_simplifier_uses_a_batch_of_one(fake_ollama):
    """Guards the invariant the test above enforces: the prompted-LLM path must not
    accumulate a multi-audience batch."""
    assert main.PromptedLLMSimplifier.batch_size == 1
    assert main.PromptedLLMSimplifier.supports_audience is True


@pytest.mark.parametrize(
    'raw,expected',
    [
        ('-1', -1),  # "never unload" -- the default
        ('0', 0),  # unload immediately after the request
        ('300', 300),  # seconds
        ('2.5', 2.5),
        ('10m', '10m'),  # Go duration strings pass through untouched
        ('1h30m', '1h30m'),
    ],
)
def test_keep_alive_env_is_coerced_to_the_type_ollama_expects(raw, expected):
    """Regression, Phase 1 spike 2026-08-14: OLLAMA_KEEP_ALIVE was passed through as a
    string, and Ollama parses a *string* keep_alive as a Go duration -- so the default
    "-1" came back as `time: missing unit in duration "-1"`, HTTP 400, on every single
    generation. Numeric values have to be sent as JSON numbers.

    No fake transport can catch this class of bug (a fake accepts whatever payload it
    is handed), which is exactly why it survived to the spike.
    """
    assert main._parse_keep_alive(raw) == expected
    assert type(main._parse_keep_alive(raw)) is type(expected)


def test_keep_alive_default_is_numeric_not_a_string():
    """The module-level default is what almost every deployment actually uses, so pin
    its resolved type rather than only the parser's behaviour."""
    assert main.OLLAMA_KEEP_ALIVE == -1
    assert isinstance(main.OLLAMA_KEEP_ALIVE, int)


def test_http_error_body_is_logged(fake_ollama, caplog):
    """Ollama reports *why* it rejected a request in the response body, which
    raise_for_status() throws away -- and that body is the only thing separating a
    malformed payload from a missing model."""
    request = httpx.Request('POST', 'http://127.0.0.1:11434/api/generate')
    response = httpx.Response(
        400, json={'error': 'time: missing unit in duration "-1"'}, request=request
    )
    fake_ollama(
        raise_on_post=httpx.HTTPStatusError('bad', request=request, response=response)
    )

    with caplog.at_level('ERROR'):
        with TestClient(main.app) as client:
            body = client.post(
                '/simplify', json={'text': 'Some complex sentence.', 'model': 'llm_7b'}
            ).json()

    assert body['fallback_reason'] == 'error'
    assert 'missing unit in duration' in caplog.text


# --- document granularity (prompted LLM) ---

# Same served tag as the sentence key; Ollama keeps one resident copy per tag (see
# MODEL_ENV_CONFIG).
DOC_TAG = TAG

LONG_SOURCE = (
    'The organisation, which was established in 1994 as a developer of interactive '
    'entertainment software, subsequently relocated its headquarters to Berlin '
    'following a period of substantial expansion, and presently employs approximately '
    'two hundred members of staff across three continents.'
)


@pytest.fixture
def fake_ollama_doc(monkeypatch):
    """Enable only the document-granularity prompted model."""
    monkeypatch.setenv('SIMPLIFIER_MODEL_LLM_DOC_7B', DOC_TAG)
    monkeypatch.setenv('SIMPLIFIER_MODEL_ONLINE', '')

    def make(**kwargs):
        client = FakeOllamaClient(tags=(DOC_TAG,), **kwargs)
        monkeypatch.setattr(main, '_make_ollama_client', lambda: client)
        return client

    return make


def test_document_llm_uses_the_document_prompt_and_budget(fake_ollama_doc):
    # long enough to pass the too_short guard for a 37-word passage
    ollama = fake_ollama_doc(
        default_reply=(
            'The group started in 1994. It made video games. Later it moved its main '
            'office to Berlin after growing quickly. It now has about two hundred staff '
            'in three parts of the world.'
        )
    )

    with TestClient(main.app) as client:
        body = client.post(
            '/simplify', json={'text': LONG_SOURCE, 'model': 'llm_doc_7b'}
        ).json()

    assert body['fallback_reason'] is None
    sent = ollama.posts[0]
    # the document instruction, not BLESS Prompt 2
    assert prompting.render_document_instruction() in sent['prompt']
    # not the sentence template: checked via its opening clause, since the document
    # template also contains "complex sentence(s)"
    assert 'the following complex sentence' not in sent['prompt']
    assert prompting.INSTRUCTION_TEMPLATE.split('{')[0] not in sent['prompt']
    assert sent['prompt'].endswith(f'Text: {LONG_SOURCE}\nSimplified text:')
    # document decoding: room for a whole section, explicit context window
    assert sent['options']['num_predict'] == 1024
    assert sent['options']['num_ctx'] == 8192
    # "\n\n" must NOT be a stop sequence here, or every paragraph break truncates
    assert '\n\n' not in sent['options']['stop']


def test_document_llm_receives_whole_text_unsplit(fake_ollama_doc):
    """Sentence models get one request per sentence; a document model must get the lot
    in one, or it cannot merge or reorder across sentence boundaries at all."""
    ollama = fake_ollama_doc(
        default_reply='A short but adequate rewrite of the passage above.'
    )
    text = 'First sentence here. Second sentence here. Third sentence here.'

    with TestClient(main.app) as client:
        client.post('/simplify', json={'text': text, 'model': 'llm_doc_7b'})

    assert len(ollama.posts) == 1
    assert (
        'First sentence here. Second sentence here. Third sentence here.'
        in ollama.posts[0]['prompt']
    )


def test_document_llm_preserves_paragraph_breaks(fake_ollama_doc):
    """Regression guard for the sentence sanitiser's first-line-only rule, which would
    silently reduce a whole section to its opening sentence."""
    reply = (
        'The group began in 1994.\n\nIt later moved to Berlin and grew to 200 staff.'
    )
    fake_ollama_doc(default_reply=reply)

    with TestClient(main.app) as client:
        body = client.post(
            '/simplify', json={'text': LONG_SOURCE, 'model': 'llm_doc_7b'}
        ).json()

    assert '\n\n' in body['simplified']
    assert 'Berlin' in body['simplified']


def test_document_llm_rejects_a_rewrite_that_deleted_the_passage(fake_ollama_doc):
    """The document-only failure mode: handed 40 words and returning 4 is deletion, not
    simplification. The sentence path has no equivalent guard and needs none."""
    fake_ollama_doc(default_reply='A software company.')

    with TestClient(main.app) as client:
        body = client.post(
            '/simplify', json={'text': LONG_SOURCE, 'model': 'llm_doc_7b'}
        ).json()

    assert body['simplified'] == LONG_SOURCE
    assert body['fallback_reason'] == 'too_short'


def test_health_reports_the_llm_context_budget_not_the_seq2seq_ceiling(fake_ollama_doc):
    """The point of the per-model map: an LLM document model must not be chunked to the
    seq2seq checkpoint's 512-token trained limit."""
    fake_ollama_doc()

    with TestClient(main.app) as client:
        body = client.get('/health').json()

    assert body['granularities'] == {
        'llm_doc_7b': vocabulary.GRANULARITY_WHOLE_SECTIONS
    }
    assert body['methods'] == {'llm_doc_7b': vocabulary.METHOD_PROMPTED_LLM}
    budget = body['document_max_tokens']['llm_doc_7b']
    assert budget > 512, budget
    # window minus reserved output minus instruction headroom
    assert budget == 8192 - 1024 - 200


def test_document_llm_does_not_apply_the_corpus_normalisation(fake_ollama_doc):
    """document_text.py exists to match D-Wikipedia's lowercased, pre-tokenized
    convention for the seq2seq checkpoint. An LLM wants ordinary prose, and lowercasing
    its input would be actively harmful -- so the prompt must carry the text verbatim.
    """
    ollama = fake_ollama_doc(
        default_reply='The group began in Berlin in 1994 and grew steadily.'
    )

    with TestClient(main.app) as client:
        client.post('/simplify', json={'text': LONG_SOURCE, 'model': 'llm_doc_7b'})

    prompt = ollama.posts[0]['prompt']
    assert 'Berlin' in prompt  # capitalisation intact
    assert 'berlin' not in prompt  # not lowercased
    assert ' ,' not in prompt  # not PTB pre-tokenized
