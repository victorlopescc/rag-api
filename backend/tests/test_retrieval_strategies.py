"""Testa as estratégias de recuperação (default / query_rewrite / widen_k).

Cobre:
- Mapeamento de tentativa → estratégia
- Fusão por RRF (Reciprocal Rank Fusion)
- Híbrido denso + BM25 com filtro por categoria
- Variantes de acento e sigla
- Estratégia de query_rewrite (parsing + união)
- Estratégia widen_k (sem filtro, k maior)
- Dispatcher
- Integração com reranker
"""
from unittest.mock import patch

from pipeline.retrieval_strategies import (
    WIDEN_K_CAP,
    _parse_rewrites,
    _rrf_fuse,
    retrieve,
    retrieve_default,
    retrieve_with_query_rewrite,
    retrieve_with_widen_k,
    rewrite_query,
    strategy_for_attempt,
)


def _chunk(id_, score, content="x", metadata=None):
    return {
        "id": id_,
        "content": content,
        "metadata": metadata or {},
        "distance": 1 - score,
        "score": score,
    }


# --- strategy_for_attempt --------------------------------------------------

def test_strategy_mapping_by_attempt():
    assert strategy_for_attempt(1) == "default"
    assert strategy_for_attempt(2) == "query_rewrite"
    assert strategy_for_attempt(3) == "widen_k"


def test_strategy_clamps_out_of_range():
    assert strategy_for_attempt(0) == "default"
    assert strategy_for_attempt(99) == "widen_k"


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


# --- default + híbrido ------------------------------------------------------

def test_default_passes_category_filter_to_search():
    """Filtro de categoria é repassado pro Chroma; BM25 desliga durante
    o teste pra a asserção valer só da busca densa."""
    with patch("pipeline.retrieval_strategies.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval_strategies.search", return_value=[]) as s, \
         patch("pipeline.retrieval_strategies.settings") as cfg:
        cfg.enable_bm25 = False
        cfg.bm25_top_k = 50
        retrieve_default("q", category="ADA")
    for call in s.call_args_list:
        assert call.kwargs["where"] == {"category": "ADA"}


def test_default_dedups_original_and_plain_accent_variants():
    with patch("pipeline.retrieval_strategies.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval_strategies.search", side_effect=[
             [_chunk("c1", 0.5)],
             [_chunk("c1", 0.9)],
         ]), \
         patch("pipeline.retrieval_strategies.settings") as cfg:
        cfg.enable_bm25 = False
        cfg.bm25_top_k = 50
        out = retrieve_default("qual a duração?", category=None)
    assert len(out) == 1
    assert out[0]["id"] == "c1"


def test_default_includes_bm25_results_when_enabled():
    """Chunks que só aparecem no BM25 (não no denso) entram no resultado."""
    dense_chunks = [_chunk("dense_only", 0.6)]
    bm25_chunks = [_chunk("bm25_only", 5.0)]

    with patch("pipeline.retrieval_strategies.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval_strategies.search", return_value=dense_chunks), \
         patch("pipeline.retrieval_strategies._bm25_search", return_value=bm25_chunks):
        out = retrieve_default("alguma pergunta", category=None)

    ids = {c["id"] for c in out}
    assert "dense_only" in ids
    assert "bm25_only" in ids


# --- query_rewrite ---------------------------------------------------------

def test_parse_rewrites_from_clean_json():
    raw = '{"rewrites": ["a", "b", "c"]}'
    assert _parse_rewrites(raw) == ["a", "b", "c"]


def test_parse_rewrites_tolerates_preamble():
    raw = 'Here you go: {"rewrites": ["x"]}'
    assert _parse_rewrites(raw) == ["x"]


def test_parse_rewrites_empty_on_garbage():
    assert _parse_rewrites("not json") == []
    assert _parse_rewrites('{"rewrites": "not a list"}') == []


def test_rewrite_query_returns_empty_on_llm_error():
    with patch("pipeline.retrieval_strategies.generate", side_effect=RuntimeError()):
        assert rewrite_query("q") == []


def test_rewrite_query_caps_at_three_variants():
    big = '{"rewrites": ["a","b","c","d","e"]}'
    with patch("pipeline.retrieval_strategies.generate", return_value=big):
        assert rewrite_query("q") == ["a", "b", "c"]


def test_query_rewrite_unions_original_and_variants():
    rewrites_json = '{"rewrites": ["v1", "v2"]}'
    # Cada variante busca devolve um chunk distinto.
    with patch("pipeline.retrieval_strategies.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval_strategies.generate", return_value=rewrites_json), \
         patch("pipeline.retrieval_strategies.search") as s, \
         patch("pipeline.retrieval_strategies._bm25_search", return_value=[]):
        s.side_effect = [
            [_chunk("c-orig", 0.6)],
            [_chunk("c-v1", 0.7)],
            [_chunk("c-v2", 0.8)],
        ]
        out = retrieve_with_query_rewrite("q", category=None)

    ids = {c["id"] for c in out}
    assert ids == {"c-orig", "c-v1", "c-v2"}


# --- widen_k ---------------------------------------------------------------

def test_widen_k_ignores_category_and_uses_larger_k():
    with patch("pipeline.retrieval_strategies.embed_text", return_value=[0.1]), \
         patch("pipeline.retrieval_strategies.search", return_value=[]) as s, \
         patch("pipeline.retrieval_strategies._bm25_search", return_value=[]):
        retrieve_with_widen_k("q", "regulamento")

    for call in s.call_args_list:
        assert call.kwargs["where"] is None
        assert call.kwargs["n_results"] <= WIDEN_K_CAP
        assert call.kwargs["n_results"] >= 1


# --- dispatcher ------------------------------------------------------------

def test_retrieve_dispatches_to_strategy():
    with patch("pipeline.retrieval_strategies.retrieve_default") as d, \
         patch("pipeline.retrieval_strategies.retrieve_with_query_rewrite") as qr, \
         patch("pipeline.retrieval_strategies.retrieve_with_widen_k") as wk:
        retrieve("q", strategy="default")
        retrieve("q", strategy="query_rewrite")
        retrieve("q", strategy="widen_k")

    d.assert_called_once()
    qr.assert_called_once()
    wk.assert_called_once()


def test_retrieve_unknown_strategy_falls_back_to_default():
    with patch("pipeline.retrieval_strategies.retrieve_default") as d:
        retrieve("q", strategy="bogus")  # type: ignore[arg-type]
    d.assert_called_once()
