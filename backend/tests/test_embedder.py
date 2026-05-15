"""Testa o embedder com httpx mockado."""
from unittest.mock import MagicMock, patch

import pipeline.embedder as embedder_module
from pipeline.embedder import embed_batch, embed_text


def _fake_client_returning(embedding):
    response = MagicMock()
    response.json.return_value = {"embedding": embedding}
    response.raise_for_status.return_value = None
    response.status_code = 200

    client = MagicMock()
    client.post.return_value = response
    client.is_closed = False
    return client


def test_embed_text_returns_vector():
    fake = _fake_client_returning([0.1, 0.2, 0.3])

    with patch.object(embedder_module, "_get_client", return_value=fake):
        vec = embed_text("regulamento do curso")

    assert vec == [0.1, 0.2, 0.3]
    payload = fake.post.call_args.kwargs["json"]
    assert payload["prompt"] == "regulamento do curso"
    assert payload["model"]  # usa o modelo configurado


def test_embed_text_raises_on_http_error():
    import httpx
    response = MagicMock()
    response.status_code = 400
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=response,
    )
    client = MagicMock()
    client.post.return_value = response
    client.is_closed = False

    with patch.object(embedder_module, "_get_client", return_value=client):
        try:
            embed_text("texto não vazio")
        except httpx.HTTPStatusError as e:
            assert "boom" in str(e)
        else:
            raise AssertionError("deveria ter levantado")


def test_embed_text_rejects_empty_input():
    """Texto vazio nem deve chegar no Ollama (evita 500)."""
    fake = _fake_client_returning([0.1])
    with patch.object(embedder_module, "_get_client", return_value=fake):
        try:
            embed_text("   \n\t  ")
        except ValueError as e:
            assert "vazio" in str(e).lower()
        else:
            raise AssertionError("deveria ter levantado ValueError")
    fake.post.assert_not_called()


def test_embed_text_strips_control_chars(monkeypatch):
    """Controle como NUL é removido antes do request."""
    fake = _fake_client_returning([0.1])
    # Sem retry/backoff durante o teste.
    monkeypatch.setattr(embedder_module, "_RETRIES", 0)
    with patch.object(embedder_module, "_get_client", return_value=fake):
        embed_text("ola\x00mundo")
    payload = fake.post.call_args.kwargs["json"]
    assert payload["prompt"] == "olamundo"


def test_embed_text_retries_on_500(monkeypatch):
    import httpx
    err_resp = MagicMock(); err_resp.status_code = 500; err_resp.text = "oops"
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=err_resp,
    )
    ok_resp = MagicMock(); ok_resp.status_code = 200
    ok_resp.json.return_value = {"embedding": [0.9]}
    ok_resp.raise_for_status.return_value = None

    client = MagicMock(); client.is_closed = False
    client.post.side_effect = [err_resp, ok_resp]
    monkeypatch.setattr(embedder_module, "_BACKOFF_SECONDS", 0.0)
    with patch.object(embedder_module, "_get_client", return_value=client):
        vec = embed_text("texto válido")
    assert vec == [0.9]
    assert client.post.call_count == 2


def test_embed_batch_wraps_failure_with_chunk_index(monkeypatch):
    import httpx
    err_resp = MagicMock(); err_resp.status_code = 500; err_resp.text = ""
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=err_resp,
    )
    ok_resp = MagicMock(); ok_resp.status_code = 200
    ok_resp.json.return_value = {"embedding": [0.1]}
    ok_resp.raise_for_status.return_value = None

    client = MagicMock(); client.is_closed = False
    # 1º chunk ok, 2º chunk falha em todas as tentativas.
    client.post.side_effect = [ok_resp, err_resp, err_resp, err_resp]
    monkeypatch.setattr(embedder_module, "_BACKOFF_SECONDS", 0.0)
    with patch.object(embedder_module, "_get_client", return_value=client):
        try:
            embed_batch(["a", "b", "c"])
        except RuntimeError as e:
            assert "chunk 2/3" in str(e)
        else:
            raise AssertionError("deveria ter levantado")


def test_embed_batch_calls_embed_text_for_each():
    fake = _fake_client_returning([0.5])

    with patch.object(embedder_module, "_get_client", return_value=fake):
        vectors = embed_batch(["a", "b", "c"])

    assert vectors == [[0.5], [0.5], [0.5]]
    assert fake.post.call_count == 3


def test_client_is_recreated_when_closed():
    embedder_module._client = None
    c1 = embedder_module._get_client()
    c1_id = id(c1)
    c1.close()

    c2 = embedder_module._get_client()
    # Quando is_closed=True, um novo cliente é criado.
    assert id(c2) != c1_id
    c2.close()
