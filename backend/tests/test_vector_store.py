"""Testa o wrapper do ChromaDB com a collection mockada."""
from unittest.mock import MagicMock, patch

from pipeline.vector_store import (
    COLLECTION_NAME,
    delete_document_chunks,
    index_chunks,
    search,
)


def _collection_with_results(ids, docs, metas, distances):
    col = MagicMock()
    col.query.return_value = {
        "ids":       [ids],
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }
    return col


def test_index_chunks_calls_upsert_with_payload():
    col = MagicMock()
    with patch("pipeline.vector_store.get_collection", return_value=col):
        index_chunks(
            chroma_ids=["a", "b"],
            embeddings=[[0.1], [0.2]],
            documents=["x", "y"],
            metadatas=[{"k": 1}, {"k": 2}],
        )

    col.upsert.assert_called_once_with(
        ids=["a", "b"],
        embeddings=[[0.1], [0.2]],
        documents=["x", "y"],
        metadatas=[{"k": 1}, {"k": 2}],
    )


def test_search_converts_distance_to_score():
    col = _collection_with_results(
        ids=["c1"], docs=["trecho"], metas=[{"m": 1}], distances=[0.2]
    )
    with patch("pipeline.vector_store.get_collection", return_value=col):
        results = search([0.1, 0.2])

    assert len(results) == 1
    assert results[0]["id"] == "c1"
    assert results[0]["content"] == "trecho"
    assert results[0]["distance"] == 0.2
    assert results[0]["score"] == round(1 - 0.2, 4)


def test_search_passes_filters():
    col = _collection_with_results([], [], [], [])
    with patch("pipeline.vector_store.get_collection", return_value=col):
        search([0.0], n_results=3, where={"category": "regulamento"})

    kwargs = col.query.call_args.kwargs
    assert kwargs["n_results"] == 3
    assert kwargs["where"] == {"category": "regulamento"}


def test_delete_document_chunks_uses_where_filter():
    col = MagicMock()
    with patch("pipeline.vector_store.get_collection", return_value=col):
        delete_document_chunks("doc-id-42")

    col.delete.assert_called_once_with(where={"document_id": "doc-id-42"})


def test_collection_name_constant():
    assert COLLECTION_NAME == "rag_documents"
