"""Testa o classificador de intenção (regex + chamada ao Ollama mockada)."""
from unittest.mock import patch

import httpx

from services.intent_classifier import (
    Prior,
    _parse_intent,
    classify,
    classify_with_llm,
)


PRIOR = Prior(question="Qual a duração?", answer="4 anos")


# --- _parse_intent --------------------------------------------------------

def test_parse_intent_from_clean_json():
    assert _parse_intent('{"intent": "yes"}') == "yes"
    assert _parse_intent('{"intent": "new_topic"}') == "new_topic"


def test_parse_intent_tolerates_surrounding_noise():
    raw = 'Claro! Aqui está: {"intent": "rephrase"} — é isso.'
    assert _parse_intent(raw) == "rephrase"


def test_parse_intent_is_case_insensitive():
    assert _parse_intent('{"intent": "YES"}') == "yes"


def test_parse_intent_unknown_value_falls_through_to_unclear():
    assert _parse_intent('{"intent": "maybe"}') == "unclear"


def test_parse_intent_falls_back_to_keyword_search():
    # Sem JSON — busca por palavra-chave no texto cru.
    assert _parse_intent("I think the student said no, very clearly.") == "no"


def test_parse_intent_picks_first_keyword():
    # Texto com várias keywords — deve pegar a primeira ocorrência.
    raw = "rephrase or new_topic? probably rephrase."
    assert _parse_intent(raw) == "rephrase"


def test_parse_intent_empty_returns_unclear():
    assert _parse_intent("") == "unclear"


# --- classify_with_llm ----------------------------------------------------

def test_classify_with_llm_posts_to_ollama_and_returns_intent():
    request = httpx.Request("POST", "http://localhost:11434/api/generate")
    fake_resp = httpx.Response(200, json={"response": '{"intent": "no"}'}, request=request)

    class FakeClient:
        is_closed = False

        def post(self, url, json):
            assert url == "/api/generate"
            assert json["stream"] is False
            assert json["options"]["temperature"] == 0.0
            return fake_resp

    with patch("services.intent_classifier._get_client", return_value=FakeClient()):
        assert classify_with_llm(PRIOR, "não entendi") == "no"


def test_classify_with_llm_returns_unclear_on_network_error():
    class Bad:
        is_closed = False

        def post(self, *a, **kw):
            raise httpx.ConnectError("down")

    with patch("services.intent_classifier._get_client", return_value=Bad()):
        assert classify_with_llm(PRIOR, "qualquer coisa") == "unclear"


# --- classify (fachada) ---------------------------------------------------

def test_classify_regex_shortcircuits_llm():
    with patch("services.intent_classifier.classify_with_llm") as llm:
        assert classify("obrigado!", prior=PRIOR) == "yes"
    llm.assert_not_called()


def test_classify_without_prior_skips_llm():
    with patch("services.intent_classifier.classify_with_llm") as llm:
        assert classify("qual a duração?", prior=None) == "unclear"
    llm.assert_not_called()


def test_classify_with_prior_delegates_to_llm_when_regex_unclear():
    with patch(
        "services.intent_classifier.classify_with_llm", return_value="new_topic",
    ) as llm:
        assert classify("e quanto ao estágio?", prior=PRIOR) == "new_topic"
    llm.assert_called_once()
