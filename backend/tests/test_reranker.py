"""Testa o módulo de reranker.

Não baixamos / carregamos o cross-encoder de verdade nos testes — o modelo
pesa ~120MB e o load demora 2-5s, o que estouraria o tempo de CI.
Ao invés, mockamos ``CrossEncoder.predict`` e validamos a lógica de
sigmoid, ordenação, top-K, min_score e o flow do ``retrieve()`` que
chama o reranker.
"""
from unittest.mock import MagicMock, patch

import pytest

from pipeline import reranker


def _chunk(id_, embed_score, content="x"):
    return {"id": id_, "content": content, "metadata": {}, "score": embed_score}


# --- _sigmoid --------------------------------------------------------------

def test_sigmoid_zero_returns_half():
    assert abs(reranker._sigmoid(0.0) - 0.5) < 1e-9


def test_sigmoid_large_positive_saturates_near_one():
    assert reranker._sigmoid(50.0) > 0.99


def test_sigmoid_large_negative_saturates_near_zero():
    assert reranker._sigmoid(-50.0) < 0.01


# --- rerank ----------------------------------------------------------------

def _mock_cross_encoder(scores_by_content: dict[str, float]):
    """Cria um mock de CrossEncoder cujo predict() devolve scores
    derivados do conteúdo dos chunks (pra teste reproduzível).

    ``scores_by_content`` mapeia substring → logit. O score do par
    (query, chunk) é o do primeiro match; default 0.0.
    """
    def predict_fn(pairs):
        out = []
        for _q, content in pairs:
            score = 0.0
            for substr, s in scores_by_content.items():
                if substr in content:
                    score = s
                    break
            out.append(score)
        return out
    m = MagicMock()
    m.predict.side_effect = predict_fn
    return m


def test_rerank_empty_input_returns_empty():
    out = reranker.rerank("q", [], top_k=10)
    assert out == []


def test_rerank_reorders_by_logit_desc():
    chunks = [
        _chunk("a", 0.6, content="cabeçalho irrelevante"),
        _chunk("b", 0.5, content="resposta direta — 5 pontos"),
        _chunk("c", 0.4, content="meio relevante: prova"),
    ]
    fake_model = _mock_cross_encoder({
        "5 pontos": 5.0,
        "prova": 1.0,
        "irrelevante": -3.0,
    })
    with patch("pipeline.reranker._load_model", return_value=fake_model):
        out = reranker.rerank("Quanto vale a prova?", chunks, top_k=3)
    assert [c["id"] for c in out] == ["b", "c", "a"]


def test_rerank_replaces_score_with_probability():
    chunks = [_chunk("a", 0.99, content="match")]
    fake_model = _mock_cross_encoder({"match": 0.0})  # logit 0 → prob 0.5
    with patch("pipeline.reranker._load_model", return_value=fake_model):
        out = reranker.rerank("q", chunks, top_k=1)
    assert abs(out[0]["score"] - 0.5) < 1e-9
    assert out[0]["embed_score"] == 0.99
    assert "rerank_score" in out[0]


def test_rerank_caps_at_top_k():
    chunks = [_chunk(f"c{i}", 0.5, content=str(i)) for i in range(10)]
    fake_model = _mock_cross_encoder({})  # all 0
    with patch("pipeline.reranker._load_model", return_value=fake_model):
        out = reranker.rerank("q", chunks, top_k=3)
    assert len(out) == 3


def test_rerank_filters_by_min_score():
    chunks = [
        _chunk("a", 0.5, content="strong"),
        _chunk("b", 0.5, content="weak"),
    ]
    # logit 5 → prob ~0.99, logit -3 → prob ~0.047
    fake_model = _mock_cross_encoder({"strong": 5.0, "weak": -3.0})
    with patch("pipeline.reranker._load_model", return_value=fake_model):
        out = reranker.rerank("q", chunks, top_k=10, min_score=0.1)
    assert [c["id"] for c in out] == ["a"]


# --- retrieve() integration ------------------------------------------------

def test_retrieve_calls_reranker_when_enabled():
    """Quando ``use_reranker=True``, retrieve() chama o cross-encoder
    e devolve chunks com ``rerank_score``."""
    from pipeline.retrieval_strategies import retrieve

    base_chunks = [
        _chunk("a", 0.7, content="header genérico"),
        _chunk("b", 0.5, content="resposta literal: 5 pontos"),
    ]
    fake_model = _mock_cross_encoder({"5 pontos": 4.0, "header": -2.0})
    with patch("pipeline.retrieval_strategies.retrieve_default",
               return_value=base_chunks), \
         patch("pipeline.reranker._load_model", return_value=fake_model):
        out = retrieve("q", strategy="default", use_reranker=True)

    assert out[0]["id"] == "b"
    assert "rerank_score" in out[0]


def test_retrieve_skips_reranker_when_disabled():
    from pipeline.retrieval_strategies import retrieve

    base_chunks = [_chunk("a", 0.7), _chunk("b", 0.5)]
    with patch("pipeline.retrieval_strategies.retrieve_default",
               return_value=base_chunks), \
         patch("pipeline.reranker._load_model") as load:
        out = retrieve("q", strategy="default", use_reranker=False)
    load.assert_not_called()
    assert [c["id"] for c in out] == ["a", "b"]


def test_retrieve_falls_back_when_reranker_raises():
    """Se o reranker lança, retrieve devolve a ordem do retrieval base."""
    from pipeline.retrieval_strategies import retrieve

    base_chunks = [_chunk("a", 0.7), _chunk("b", 0.5)]
    with patch("pipeline.retrieval_strategies.retrieve_default",
               return_value=base_chunks), \
         patch("pipeline.reranker.rerank", side_effect=RuntimeError("boom")):
        out = retrieve("q", strategy="default", use_reranker=True)
    assert [c["id"] for c in out] == ["a", "b"]
