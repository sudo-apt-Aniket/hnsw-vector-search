"""
test_ai_providers.py — tests the Ollama -> sentence-transformers ->
Groq cascade logic in isolation, using mocks throughout. This
deliberately never performs a real network call or model load: in CI
and in this sandbox there's no Ollama daemon, no cached
sentence-transformers model, and no Groq key, so anything that hit the
real backends would either fail loudly or (worse) hang trying to
download a model. The cascade *logic* is what's under test here, not
the third-party services themselves.
"""

from unittest.mock import patch, MagicMock

from ai_providers import AIProvider


def test_embed_falls_back_to_sentence_transformers_when_ollama_down():
    ai = AIProvider()
    with patch.object(ai.ollama, "is_available", return_value=False), \
         patch.object(ai.st, "embed", return_value=[0.1] * 384), \
         patch.object(ai.st, "is_available", return_value=True):
        vec, backend = ai.embed("hello world")
    assert backend == "sentence-transformers"
    assert len(vec) == 384


def test_embed_prefers_ollama_when_available():
    ai = AIProvider()
    with patch.object(ai.ollama, "is_available", return_value=True), \
         patch.object(ai.ollama, "embed", return_value=[0.2] * 768):
        vec, backend = ai.embed("hello world")
    assert backend == "ollama"
    assert len(vec) == 768


def test_embed_backend_pins_after_first_success():
    """Once sentence-transformers has been used, later calls must not
    silently switch to Ollama even if it becomes available — mixing
    embedding dimensions in the same index would corrupt search."""
    ai = AIProvider()
    with patch.object(ai.ollama, "is_available", return_value=False), \
         patch.object(ai.st, "embed", return_value=[0.1] * 384):
        ai.embed("first call")

    assert ai._embed_backend == "sentence-transformers"

    with patch.object(ai.ollama, "is_available", return_value=True), \
         patch.object(ai.ollama, "embed", return_value=[0.9] * 768), \
         patch.object(ai.st, "embed", return_value=[0.1] * 384):
        vec2, backend2 = ai.embed("second call")

    assert backend2 == "sentence-transformers"
    assert len(vec2) == 384  # NOT 768 — pinned dimension preserved


def test_embed_returns_empty_when_nothing_available():
    ai = AIProvider()
    with patch.object(ai.ollama, "is_available", return_value=False), \
         patch.object(ai.st, "embed", return_value=[]):
        vec, backend = ai.embed("no backend works")
    assert vec == []
    assert backend == "none"


def test_generate_prefers_ollama():
    ai = AIProvider()
    with patch.object(ai.ollama, "is_available", return_value=True), \
         patch.object(ai.ollama, "generate", return_value="ollama's answer"):
        answer, backend = ai.generate("a question")
    assert answer == "ollama's answer"
    assert backend == "ollama"


def test_generate_falls_back_to_groq_with_correct_api_contract():
    ai = AIProvider()
    ai.groq.api_key = "fake-test-key"

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "groq's answer"}}]
    }

    with patch.object(ai.ollama, "is_available", return_value=False), \
         patch("ai_providers.requests.post", return_value=fake_response) as mock_post:
        answer, backend = ai.generate("a question")

    assert answer == "groq's answer"
    assert backend == "groq"

    call_args = mock_post.call_args
    assert call_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert "Authorization" in call_args.kwargs["headers"]
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer fake-test-key"
    assert call_args.kwargs["json"]["messages"][0]["content"] == "a question"


def test_generate_returns_empty_when_groq_not_configured():
    ai = AIProvider()
    ai.groq.api_key = ""  # not configured
    with patch.object(ai.ollama, "is_available", return_value=False):
        answer, backend = ai.generate("a question")
    assert answer == ""
    assert backend == "none"


def test_generate_handles_groq_error_response_gracefully():
    ai = AIProvider()
    ai.groq.api_key = "fake-test-key"
    fake_response = MagicMock()
    fake_response.status_code = 401  # bad key
    with patch.object(ai.ollama, "is_available", return_value=False), \
         patch("ai_providers.requests.post", return_value=fake_response):
        answer, backend = ai.generate("a question")
    assert answer == ""
    assert backend == "none"


def test_status_reports_all_three_backends():
    ai = AIProvider()
    ai.groq.api_key = "fake-key"
    with patch.object(ai.ollama, "is_available", return_value=False):
        status = ai.status()
    assert status["ollamaAvailable"] is False
    assert status["groqConfigured"] is True
    assert status["embedBackend"] == "not yet used"
