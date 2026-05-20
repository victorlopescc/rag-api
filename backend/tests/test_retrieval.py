"""Testa o retrieval híbrido (denso + BM25 + RRF).

Cobre:
- Fusão por RRF (Reciprocal Rank Fusion)
- Híbrido denso + BM25 com filtro por categoria
- Variantes de acento e sigla
- Integração com reranker
"""
from unittest.mock import patch

from pipeline.retrieval import _rrf_fuse, retrieve


def _chunk(id_, score, content="x", metadata=None):
    return {
        "id": id_,
        "content": content,
        "metadata": metadata or {},
        "distance": 1 - score,
        "score": score,
    }


# --- RRF fusion ------------------------------------------------------------

def test_rrf_fuse_empty_returns_empty():
    assert _rrf_fuse([]) == []
    assert _rrf_fuse([[]]) == []


def test_rrf_fuse_single_ranking_preserves_order():
    a = _chunk("a", 0.9)
    b = _chunk("b", 0.5)
    out = _rrf_fuse([[a, b]])
    assert [c["id"] for c in out] == ["a", "b"]


def test_rrf_fuse_combines_two_rankings_giving_higher_to_consensus():
    """Chunk que aparece em ambos rankings deve ranquear acima de
    chunks que aparecem em só um, mesmo se em posição mediana."""
    dense = [_chunk("a", 0.9), _chunk("b", 0.7), _chunk("c", 0.5)]
    bm25 = [_chunk("c", 5.0), _chunk("d", 4.0), _chunk("a", 1.0)]
    out = _rrf_fuse([dense, bm25])
    ids = [c["id"] for c in out]
    # 'a' rank 0 + rank 2 = 1/61 + 1/63 ≈ 0.0322
    # 'c' rank 2 + rank 0 = 1/63 + 1/61 ≈ 0.0322 (idem!)
    # 'b' rank 1 = 1/62 ≈ 0.0161
    # 'd' rank 1 = 1/62 ≈ 0.0161
    # 'a' e 'c' empatam (consenso), ambos vêm antes de 'b' e 'd'.
    assert set(ids[:2]) == {"a", "c"}


def test_rrf_fuse_substitutes_score_with_rrf_score():
    a = _chunk("a", 999.0)  # score original gigante
    out = _rrf_fuse([[a]])
    # RRF score pra rank 0 é 1/(60+1) = 0.0164
    assert abs(out[0]["score"] - (1 / 61)) < 1e-9


# --- retrieve híbrido ------------------------------------------------------

def test_retrieve_passes_category_filter_to_search():
    """Filtro de categoria é repassado pro Chroma; BM25 desliga durante
    o teste pra a asserção valer só da busca densa."""
    with patch("pipeline.retrieval.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval.search", return_value=[]) as s, \
         patch("pipeline.retrieval.settings") as cfg:
        cfg.enable_bm25 = False
        cfg.bm25_top_k = 50
        cfg.enable_reranker = False
        retrieve("q", category="ADA")
    for call in s.call_args_list:
        assert call.kwargs["where"] == {"category": "ADA"}


def test_retrieve_dedups_original_and_plain_accent_variants():
    with patch("pipeline.retrieval.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval.search", side_effect=[
             [_chunk("c1", 0.5)],
             [_chunk("c1", 0.9)],
         ]), \
         patch("pipeline.retrieval.settings") as cfg:
        cfg.enable_bm25 = False
        cfg.bm25_top_k = 50
        cfg.enable_reranker = False
        out = retrieve("qual a duração?", category=None)
    assert len(out) == 1
    assert out[0]["id"] == "c1"


def test_retrieve_includes_bm25_results_when_enabled():
    """Chunks que só aparecem no BM25 (não no denso) entram no resultado."""
    dense_chunks = [_chunk("dense_only", 0.6)]
    bm25_chunks = [_chunk("bm25_only", 5.0)]

    with patch("pipeline.retrieval.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval.search", return_value=dense_chunks), \
         patch("pipeline.retrieval._bm25_search", return_value=bm25_chunks), \
         patch("pipeline.retrieval.settings") as cfg:
        cfg.enable_reranker = False
        out = retrieve("alguma pergunta", category=None)

    ids = {c["id"] for c in out}
    assert "dense_only" in ids
    assert "bm25_only" in ids


# --- reranker hook ---------------------------------------------------------

def test_retrieve_skips_reranker_when_disabled():
    with patch("pipeline.retrieval.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval.search", return_value=[_chunk("c1", 0.6)]), \
         patch("pipeline.retrieval._bm25_search", return_value=[]), \
         patch("pipeline.reranker.rerank") as rer:
        retrieve("q", use_reranker=False)
    rer.assert_not_called()


def test_retrieve_calls_reranker_when_enabled_and_has_chunks():
    base = [_chunk("c1", 0.6)]
    with patch("pipeline.retrieval.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval.search", return_value=base), \
         patch("pipeline.retrieval._bm25_search", return_value=[]), \
         patch("pipeline.reranker.rerank", return_value=base) as rer:
        retrieve("q", use_reranker=True)
    rer.assert_called_once()
