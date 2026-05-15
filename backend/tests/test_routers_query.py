"""Testa o router /query com rag_engine mockado."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from database import get_db
from main import app
from rag_engine import RAGResponse


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_query_requires_api_key(client):
    resp = client.post("/query", json={"question": "x"})
    assert resp.status_code == 401


def test_empty_question_is_rejected(client, api_headers):
    resp = client.post("/query", json={"question": "   "}, headers=api_headers)
    assert resp.status_code == 400


def test_query_returns_answer_and_logs(client, api_headers, mock_db):
    fake = RAGResponse(
        answer="O curso dura 4 anos.",
        was_fallback=False,
        chunks_used=[{"id": "c1", "document_id": "d1", "score": 0.9}],
        latency_ms=123,
    )
    with patch("routers.query.ask", return_value=fake):
        resp = client.post(
            "/query",
            headers=api_headers,
            json={"question": "Qual a duração?", "phone_number": "5511"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "O curso dura 4 anos."
    assert body["was_fallback"] is False
    assert body["latency_ms"] == 123

    # Persistiu QueryLog.
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_query_passes_category_to_rag_engine(client, api_headers):
    fake = RAGResponse(answer="ok", was_fallback=False, chunks_used=[], latency_ms=1)
    with patch("routers.query.ask", return_value=fake) as ask_mock:
        client.post(
            "/query",
            headers=api_headers,
            json={"question": "x", "category": "regulamento"},
        )

    assert ask_mock.call_args.kwargs["category"] == "regulamento"
