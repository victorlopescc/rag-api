"""Testa services.evolution_client com httpx mockado."""
import httpx
import pytest

from services.evolution_client import EvolutionClient


def _make_response(status_code: int, json_body: dict | None = None):
    request = httpx.Request("POST", "http://localhost:8080/message/sendText/x")
    if json_body is None:
        return httpx.Response(status_code, request=request)
    return httpx.Response(status_code, json=json_body, request=request)


class _FakeAsyncClient:
    def __init__(self, response):
        self._response = response
        self.captured_url = None
        self.captured_payload = None
        self.captured_headers = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        self.captured_url = url
        self.captured_payload = json
        self.captured_headers = headers
        return self._response


@pytest.mark.asyncio
async def test_send_text_returns_message_id_on_success(monkeypatch):
    fake = _FakeAsyncClient(_make_response(201, {"key": {"id": "MSG-42"}}))
    monkeypatch.setattr(
        "services.evolution_client.httpx.AsyncClient", lambda timeout=None: fake
    )

    client = EvolutionClient(
        base_url="http://localhost:8080",
        api_key="k",
        instance="inst",
    )
    msg_id = await client.send_text("5511999999999", "hello")

    assert msg_id == "MSG-42"
    assert fake.captured_url == "http://localhost:8080/message/sendText/inst"
    assert fake.captured_payload == {
        "number": "5511999999999",
        "textMessage": {"text": "hello"},
    }
    assert fake.captured_headers["apikey"] == "k"


@pytest.mark.asyncio
async def test_send_text_returns_none_on_http_error(monkeypatch):
    fake = _FakeAsyncClient(_make_response(400, {"detail": "bad"}))
    monkeypatch.setattr(
        "services.evolution_client.httpx.AsyncClient", lambda timeout=None: fake
    )

    client = EvolutionClient(base_url="http://x", api_key="k", instance="i")
    assert await client.send_text("x", "y") is None


@pytest.mark.asyncio
async def test_send_text_returns_none_when_key_missing(monkeypatch):
    fake = _FakeAsyncClient(_make_response(200, {"nokey": True}))
    monkeypatch.setattr(
        "services.evolution_client.httpx.AsyncClient", lambda timeout=None: fake
    )

    client = EvolutionClient(base_url="http://x", api_key="k", instance="i")
    assert await client.send_text("x", "y") is None


def test_constructor_falls_back_to_settings():
    # Sem argumentos, usa settings do .env de teste (ver conftest).
    client = EvolutionClient()
    assert client.base_url == "http://localhost:8080"
    assert client.api_key == "test-key"
    assert client.instance == "test-instance"


def test_base_url_trailing_slash_is_stripped():
    client = EvolutionClient(base_url="http://x/", api_key="k", instance="i")
    assert client.base_url == "http://x"


# --- send_poll -----------------------------------------------------------

@pytest.mark.asyncio
async def test_send_poll_posts_to_correct_endpoint(monkeypatch):
    fake = _FakeAsyncClient(_make_response(201, {"key": {"id": "POLL-7"}}))
    monkeypatch.setattr(
        "services.evolution_client.httpx.AsyncClient", lambda timeout=None: fake
    )
    client = EvolutionClient(base_url="http://x", api_key="k", instance="inst")

    pid = await client.send_poll(
        number="5511999999999",
        name="O bot ajudou?",
        options=["Sim", "Não"],
        selectable_count=1,
    )

    assert pid == "POLL-7"
    assert fake.captured_url == "http://x/message/sendPoll/inst"
    assert fake.captured_payload == {
        "number": "5511999999999",
        "pollMessage": {
            "name": "O bot ajudou?",
            "selectableCount": 1,
            "values": ["Sim", "Não"],
        },
    }


@pytest.mark.asyncio
async def test_send_poll_returns_none_on_error(monkeypatch):
    fake = _FakeAsyncClient(_make_response(500))
    monkeypatch.setattr(
        "services.evolution_client.httpx.AsyncClient", lambda timeout=None: fake
    )
    client = EvolutionClient(base_url="http://x", api_key="k", instance="i")
    assert await client.send_poll("n", "name", ["a"]) is None
