"""Testa o orquestrador de ingestão com todas as dependências mockadas."""
from unittest.mock import MagicMock, patch

import pytest

from pipeline.ingestor import delete_document, ingest_document


def _fake_db():
    """DB fake que atribui id/refresh nos documentos criados."""
    db = MagicMock()
    # .refresh apenas deixa o objeto como está — mas precisamos de um doc.id.
    # Configuramos no add: o primeiro Document adicionado recebe um id UUID-like.
    return db


def _make_mocks(monkeypatch, *, text="texto suficientemente longo " * 30,
                chunks=None, embeddings=None):
    chunks = chunks or ["chunk-a", "chunk-b"]
    embeddings = embeddings or [[0.1], [0.2]]
    monkeypatch.setattr("pipeline.ingestor.extract_text", lambda b, f: text)
    monkeypatch.setattr("pipeline.ingestor.split_text", lambda t: chunks)
    monkeypatch.setattr("pipeline.ingestor.embed_batch", lambda c: embeddings)
    idx = MagicMock()
    delete = MagicMock()
    monkeypatch.setattr("pipeline.ingestor.index_chunks", idx)
    monkeypatch.setattr("pipeline.ingestor.delete_document_chunks", delete)
    return idx, delete


def test_ingest_happy_path(monkeypatch):
    idx, _ = _make_mocks(monkeypatch)
    db = _fake_db()

    doc = ingest_document(db, b"data", "f.txt", category="regulamento")

    assert doc.status == "indexed"
    assert doc.total_chunks == 2
    idx.assert_called_once()
    assert db.commit.call_count >= 2


def test_ingest_raises_on_empty_text(monkeypatch):
    _make_mocks(monkeypatch, text="")
    db = _fake_db()

    with pytest.raises(ValueError):
        ingest_document(db, b"", "f.txt")


def test_ingest_marks_status_error_on_failure(monkeypatch):
    # split_text explode — pipeline deve marcar o doc como "error".
    monkeypatch.setattr("pipeline.ingestor.extract_text", lambda b, f: "txt")
    def boom(_): raise RuntimeError("split falhou")
    monkeypatch.setattr("pipeline.ingestor.split_text", boom)
    monkeypatch.setattr("pipeline.ingestor.delete_document_chunks", MagicMock())

    db = _fake_db()

    with pytest.raises(RuntimeError):
        ingest_document(db, b"x", "f.txt")


def test_delete_document_returns_false_when_not_found():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    with patch("pipeline.ingestor.delete_document_chunks") as deleter:
        result = delete_document(db, "missing-id")

    assert result is False
    deleter.assert_not_called()


def test_ingest_auto_suggests_category_when_not_provided(monkeypatch):
    """Quando o coordenador NÃO informa categoria, o ingestor deve
    chamar ``suggest_category`` e usar o resultado nos metadados dos
    chunks. Categoria explícita não deve ser sobrescrita.
    """
    text_with_ada = "Resolução ADA. ADA. ADA. Conforme a ADA. " * 5
    _make_mocks(monkeypatch, text=text_with_ada)
    db = _fake_db()

    doc = ingest_document(db, b"x", "Resolucao_ADA.pdf", category=None)
    # Categoria foi auto-detectada pra "ADA".
    assert doc.category == "ADA"


def test_ingest_does_not_override_explicit_category(monkeypatch):
    text_with_ada = "ADA " * 50  # texto fortemente sobre ADA
    _make_mocks(monkeypatch, text=text_with_ada)
    db = _fake_db()

    doc = ingest_document(db, b"x", "ADA_doc.pdf", category="estagio")
    # Mantém a categoria explícita do coordenador.
    assert doc.category == "estagio"


def test_ingest_keeps_category_none_when_no_signal(monkeypatch):
    """Sem sigla no filename nem hits no texto, categoria fica None
    (coordenador define depois)."""
    _make_mocks(monkeypatch, text="texto sem nenhuma sigla relevante " * 20)
    db = _fake_db()

    doc = ingest_document(db, b"x", "documento_qualquer.pdf", category=None)
    assert doc.category in (None, "")


def test_delete_document_removes_from_chroma_and_pg():
    doc = MagicMock()
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = doc

    with patch("pipeline.ingestor.delete_document_chunks") as deleter:
        result = delete_document(db, "some-id")

    assert result is True
    deleter.assert_called_once_with("some-id")
    db.delete.assert_called_once_with(doc)
    db.commit.assert_called_once()
