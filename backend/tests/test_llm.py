"""Testa o wrapper de chamada ao Ollama LLM."""
from unittest.mock import MagicMock, patch

import pipeline.llm as llm_module
from pipeline.llm import generate


def _fake_client(response_text):
    response = MagicMock()
    response.json.return_value = {"response": response_text}
    response.raise_for_status.return_value = None
    response.status_code = 200

    client = MagicMock()
    client.post.return_value = response
    client.is_closed = False
    return client


def test_generate_returns_stripped_response():
    fake = _fake_client("\n  resposta do modelo  \n")

    with patch.object(llm_module, "_get_client", return_value=fake):
        out = generate("qual a duração?")

    assert out == "resposta do modelo"


def test_generate_sends_correct_payload():
    fake = _fake_client("ok")

    with patch.object(llm_module, "_get_client", return_value=fake):
        generate("pergunta")

    payload = fake.post.call_args.kwargs["json"]
    assert payload["prompt"] == "pergunta"
    assert payload["stream"] is False
    assert payload["model"]  # modelo vindo da config de teste


def test_generate_strips_meta_commentary():
    """Regressão: o modelo às vezes vaza '(Ambiguidade resolvida: ...)'
    e suposições no fim — devem ser podados antes de virar mensagem
    do WhatsApp pro aluno."""
    cases = [
        (
            "A ADA será de 15 a 19 de junho. (Ambiguidade resolvida: supus que se referia à avaliação de 2026.)",
            "A ADA será de 15 a 19 de junho.",
        ),
        (
            "Não é permitido. (Supus que a pergunta era sobre calculadoras na ADA.)",
            "Não é permitido.",
        ),
        (
            "30 questões. (Considerando que se referia à prova final.)",
            "30 questões.",
        ),
    ]
    for raw, expected in cases:
        fake = _fake_client(raw)
        with patch.object(llm_module, "_get_client", return_value=fake):
            out = generate("x")
        assert out == expected, f"raw={raw!r} → got {out!r}"


def test_generate_strips_non_latin_tokens():
    """qwen2.5:14b às vezes vaza chinês/CJK no meio de respostas PT.
    Pós-processamento remove esses caracteres pra não chegar no aluno."""
    raw = "Walisson Ferreira de Carvalho 协调评估表现。 (Resolução ADA)"
    fake = _fake_client(raw)
    with patch.object(llm_module, "_get_client", return_value=fake):
        out = generate("x")
    assert "协" not in out and "评" not in out
    assert "Walisson Ferreira de Carvalho" in out
    assert "(Resolução ADA)" in out


def test_generate_preserves_legit_source_citation():
    """Citações curtas como '(Resolução ADA, §3.1)' devem ser mantidas."""
    fake = _fake_client("Você tem 1h40 para a prova. (Resolução ADA, §3.5)")
    with patch.object(llm_module, "_get_client", return_value=fake):
        out = generate("x")
    assert "(Resolução ADA, §3.5)" in out


def test_generate_raises_on_http_error(monkeypatch):
    """Erro 4xx (não 5xx) propaga sem retry."""
    import httpx
    response = MagicMock()
    response.status_code = 400
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=response,
    )
    client = MagicMock()
    client.post.return_value = response
    client.is_closed = False

    with patch.object(llm_module, "_get_client", return_value=client):
        try:
            generate("x")
        except httpx.HTTPStatusError as e:
            assert "boom" in str(e)
        else:
            raise AssertionError("deveria ter levantado")


def test_generate_retries_on_500(monkeypatch):
    """Erro 5xx do Ollama (OOM transitório) faz retry com backoff."""
    import httpx
    err_resp = MagicMock(); err_resp.status_code = 500; err_resp.text = "out of memory"
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=err_resp,
    )
    ok_resp = MagicMock(); ok_resp.status_code = 200
    ok_resp.json.return_value = {"response": "resposta certa"}
    ok_resp.raise_for_status.return_value = None

    client = MagicMock(); client.is_closed = False
    client.post.side_effect = [err_resp, ok_resp]

    # Sem backoff durante o teste
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch.object(llm_module, "_get_client", return_value=client):
        out = generate("texto")
    assert out == "resposta certa"
    assert client.post.call_count == 2
