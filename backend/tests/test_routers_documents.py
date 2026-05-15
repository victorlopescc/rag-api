"""Testa o router /documents via FastAPI TestClient."""
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from database import Document, get_db
from main import app


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _doc(status="indexed"):
    d = Document(
        id=uuid.uuid4(),
        filename="reg.pdf",
        file_type="pdf",
        category="regulamento",
        description="d",
        file_size=100,
        total_chunks=3,
        status=status,
        error_msg=None,
    )
    d.created_at = datetime.now(timezone.utc)
    return d


def test_upload_requires_api_key(client):
    resp = client.post("/documents", files={"file": ("a.txt", b"data")})
    assert resp.status_code == 401


def test_upload_rejects_unsupported_extension(client, api_headers):
    resp = client.post(
        "/documents",
        headers=api_headers,
        files={"file": ("a.exe", b"data")},
    )
    assert resp.status_code == 400
    assert "não suportado" in resp.text.lower()


def test_upload_calls_ingestor_and_returns_document(client, api_headers):
    fake_doc = _doc()
    with patch("routers.documents.ingest_document", return_value=fake_doc) as ingest:
        resp = client.post(
            "/documents",
            headers=api_headers,
            files={"file": ("a.txt", b"conteudo")},
            data={"category": "regulamento", "description": "desc"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "reg.pdf"
    assert body["status"] == "indexed"
    ingest.assert_called_once()


def test_list_documents_returns_all(client, api_headers, mock_db):
    mock_db.query.return_value.order_by.return_value.all.return_value = [_doc(), _doc()]
    resp = client.get("/documents", headers=api_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_delete_document_not_found(client, api_headers):
    with patch("routers.documents.delete_document", return_value=False):
        resp = client.delete("/documents/missing", headers=api_headers)
    assert resp.status_code == 404


def test_delete_document_success(client, api_headers):
    with patch("routers.documents.delete_document", return_value=True):
        resp = client.delete("/documents/abc", headers=api_headers)
    assert resp.status_code == 200
    assert "removido" in resp.json()["detail"].lower()


# --- suggest-category ------------------------------------------------------

def test_suggest_category_requires_api_key(client):
    resp = client.post(
        "/documents/suggest-category", files={"file": ("a.pdf", b"data")}
    )
    assert resp.status_code == 401


def test_suggest_category_rejects_unsupported_extension(client, api_headers):
    resp = client.post(
        "/documents/suggest-category",
        headers=api_headers,
        files={"file": ("a.exe", b"data")},
    )
    assert resp.status_code == 400


def test_suggest_category_uses_filename_signal(client, api_headers):
    """Sigla no nome do arquivo deve casar mesmo sem texto."""
    with patch("routers.documents.extract_text", return_value=""):
        resp = client.post(
            "/documents/suggest-category",
            headers=api_headers,
            files={"file": ("Resolucao_ADA_2026.pdf", b"%PDF-")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["suggested_category"] == "ADA"
    assert "ADA" in body["available_categories"]


# --- /documents/{id}/chunks (preview) ---------------------------------------

def _chunk_row(idx, content="conteudo do chunk", tokens=10):
    from database import Chunk
    c = Chunk(
        chunk_index=idx,
        content=content,
        chroma_id=f"id_{idx}",
        token_count=tokens,
    )
    return c


def test_chunks_endpoint_requires_api_key(client):
    resp = client.get("/documents/abc/chunks")
    assert resp.status_code == 401


def test_chunks_endpoint_404_when_doc_not_found(client, api_headers, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    resp = client.get("/documents/missing/chunks", headers=api_headers)
    assert resp.status_code == 404


def test_chunks_endpoint_returns_paginated_chunks(client, api_headers, mock_db):
    doc = _doc()
    chain = mock_db.query.return_value.filter.return_value
    # Primeira chamada: .first() → doc. Segunda chamada (count): retorna 5.
    chain.first.return_value = doc
    chain.count.return_value = 5
    chain.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
        _chunk_row(0, content="primeiro chunk"),
        _chunk_row(1, content="segundo chunk"),
    ]
    resp = client.get(
        "/documents/abc/chunks?offset=0&limit=2", headers=api_headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_chunks"] == 5
    assert body["offset"] == 0
    assert body["limit"] == 2
    assert len(body["chunks"]) == 2
    assert body["chunks"][0]["chunk_index"] == 0
    assert body["chunks"][0]["content"] == "primeiro chunk"
    assert body["chunks"][0]["char_count"] == len("primeiro chunk")


def test_chunks_endpoint_validates_query_params(client, api_headers):
    # limit > 200 não é permitido (proteção contra abuso).
    resp = client.get("/documents/abc/chunks?limit=500", headers=api_headers)
    assert resp.status_code == 422


def test_suggest_category_returns_null_when_no_signal(client, api_headers):
    with patch("routers.documents.extract_text", return_value="texto sem siglas"):
        resp = client.post(
            "/documents/suggest-category",
            headers=api_headers,
            files={"file": ("documento_qualquer.pdf", b"%PDF-")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["suggested_category"] is None
    # Lista de categorias disponíveis sempre vem (não-vazia se há siglas registradas).
    assert len(body["available_categories"]) >= 1
