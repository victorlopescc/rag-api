"""Retrieval híbrido (BM25 + denso) com fusão por RRF.

Devolve uma lista de chunks ``{id, score, ...}`` no mesmo formato de
``pipeline.vector_store.search``.

Sinais combinados:
  - Denso (nomic-embed-text): paráfrase e semântica geral. Roda em
    variantes de acento e expansão de sigla (``query_variants``).
  - BM25: match literal — nomes próprios, siglas raras, tokens-chave
    que o embedding dilui. Cobre o caso de aluno usar vocabulário
    diferente do documento sem precisar de regex hardcoded.
  - Overlap cru de tokens: complementa BM25 em corpora pequenos onde
    IDF satura. Doc-agnóstico — pura contagem de tokens em comum.
  - Boost por categoria detectada: quando a query menciona uma sigla
    registrada (ADA, TCC, PPC, ...), uma busca EXTRA restrita aos chunks
    daquela categoria entra no RRF. Compensa o problema de docs pequenos
    sumirem na busca global quando há um doc grande tipo PPC (~970
    chunks). Pra novo doc ser priorizado, basta registrar sua sigla em
    ``ACRONYM_TO_CATEGORY``.

Histórico
---------
Antes existiam três estratégias intercambiáveis (``default`` /
``query_rewrite`` / ``widen_k``), escolhidas pela tentativa atual da
sessão (1/2/3). A ideia era "tentar diferente" quando a anterior falhou.

O eval mostrou que:
- ``query_rewrite`` (LLM reformula a pergunta em 3 variantes) custava
  uma chamada extra de LLM por tentativa e dava ganho marginal já que
  BM25 cobria a maior parte do benefício (sinônimos).
- ``widen_k`` (drop do filtro de categoria + k maior) raramente entrava
  em ação porque a tentativa 3 quase sempre escala antes.

Ambas removidas. Restou esta função única, equivalente ao antigo
``default``. A lógica de "3 tentativas → escala" permanece em
``services.session_manager.plan_interaction`` — o que muda entre as
tentativas é a pergunta do aluno (ele reformulou), não a estratégia.

Versões mais antigas também tinham ``_CONCEPT_EXPANSIONS`` (regex pattern
→ tokens hardcoded por domínio) e ``_lexical_boost``. Substituídos por
BM25 + RRF, doc-agnósticos.
"""
from __future__ import annotations

import logging
import unicodedata

from config import settings
from pipeline.acronyms import detect_categories, query_variants
from pipeline.embedder import embed_text
from pipeline.vector_store import search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")


# Reciprocal Rank Fusion (Cormack et al. 2009). Combina múltiplos
# rankings via 1/(k+rank). Vantagem sobre fusão por score: robusto a
# escalas diferentes (BM25 ∈ [0,∞], cosseno ∈ [0,1]) sem precisar
# calibrar pesos. k=60 é o default do paper, funciona bem out-of-box.
_RRF_K = 60


def _rrf_fuse(rankings: list[list[dict]]) -> list[dict]:
    """Funde múltiplos rankings via RRF. Mantém o primeiro chunk
    encontrado pra cada id (metadados + content), substitui o score
    pela soma RRF e ordena desc."""
    fused_chunks: dict[str, dict] = {}
    rrf_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if cid not in fused_chunks:
                fused_chunks[cid] = chunk
    out = [{**fused_chunks[cid], "score": s} for cid, s in rrf_scores.items()]
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def _bm25_search(question: str, *, where: dict | None) -> list[dict]:
    """Roda BM25 e (opcionalmente) filtra resultados por metadado."""
    if not settings.enable_bm25:
        return []
    from pipeline import bm25_index
    if not bm25_index.is_built():
        return []
    if where is None:
        return bm25_index.search(question, n_results=settings.bm25_top_k)
    cat = where.get("category")
    raw = bm25_index.search(question, n_results=settings.bm25_top_k * 2)
    if not cat:
        return raw[: settings.bm25_top_k]
    return _filter_by_category(raw, cat)[: settings.bm25_top_k]


def _lexical_overlap_search(question: str, *, where: dict | None) -> list[dict]:
    """Ranking de overlap cru de tokens (complementa o BM25 em casos
    de corpora pequenos onde IDF satura). Doc-agnóstico — pura
    contagem de tokens em comum.
    """
    if not settings.enable_bm25:
        return []
    from pipeline import bm25_index
    if not bm25_index.is_built():
        return []
    if where is None:
        return bm25_index.lexical_overlap_search(question, n_results=settings.bm25_top_k)
    cat = where.get("category")
    raw = bm25_index.lexical_overlap_search(question, n_results=settings.bm25_top_k * 2)
    if not cat:
        return raw[: settings.bm25_top_k]
    return _filter_by_category(raw, cat)[: settings.bm25_top_k]


def _filter_by_category(chunks: list[dict], category: str) -> list[dict]:
    return [
        c for c in chunks
        if (c.get("metadata") or {}).get("category") == category
    ]


def _dense_with_variants(
    question: str, *, n_results: int | None, where: dict | None
) -> list[list[dict]]:
    """Roda a busca densa pra cada variante distinta da query
    (acento removido + sigla expandida). Devolve lista de rankings
    pra alimentar o RRF.
    """
    rankings: list[list[dict]] = []
    seen_queries: set[str] = set()
    for v in query_variants(question):
        if v in seen_queries:
            continue
        seen_queries.add(v)
        rankings.append(search(embed_text(v), n_results=n_results, where=where))
        plain = _strip_accents(v)
        if plain != v and plain not in seen_queries:
            seen_queries.add(plain)
            rankings.append(search(embed_text(plain), n_results=n_results, where=where))
    return rankings


def _hybrid_search(
    question: str, *, n_results: int | None, where: dict | None
) -> list[dict]:
    """Busca híbrida: denso (com variantes de acento e sigla) + BM25 +
    overlap lexical, com boost via busca extra restrita à categoria
    detectada na query (se houver sigla). Tudo fundido por RRF."""
    rankings: list[list[dict]] = []

    # Denso com variantes
    rankings.extend(_dense_with_variants(question, n_results=n_results, where=where))

    # BM25 (TF-IDF normalizado)
    bm25_chunks = _bm25_search(question, where=where)
    if bm25_chunks:
        rankings.append(bm25_chunks)

    # Overlap cru de tokens (complementa BM25 em corpora pequenos).
    overlap_chunks = _lexical_overlap_search(question, where=where)
    if overlap_chunks:
        rankings.append(overlap_chunks)

    # Boost por categoria detectada — só quando o caller não passou
    # ``where`` explícito (caso contrário ele já está priorizando 1 cat).
    if where is None:
        for cat in detect_categories(question):
            cat_filter = {"category": cat}
            rankings.extend(
                _dense_with_variants(question, n_results=n_results, where=cat_filter)
            )
            cat_bm25 = _bm25_search(question, where=cat_filter)
            if cat_bm25:
                rankings.append(cat_bm25)
            cat_overlap = _lexical_overlap_search(question, where=cat_filter)
            if cat_overlap:
                rankings.append(cat_overlap)

    return _rrf_fuse(rankings)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    *,
    category: str | None = None,
    use_reranker: bool | None = None,
) -> list[dict]:
    """Recupera chunks relevantes pra ``question`` e (opcionalmente)
    reranqueia com cross-encoder.

    Parâmetros:
        question: pergunta do aluno (já pode vir com prior_question
                  concatenado pelo caller — ver ``rag_engine.ask``).
        category: filtra a busca por categoria de documento. Se None,
                  busca em tudo + boost por categoria detectada na query.
        use_reranker: None → segue ``settings.enable_reranker``. True/False
                      força o comportamento (útil pro eval harness).
    """
    where = {"category": category} if category else None
    chunks = _hybrid_search(question, n_results=None, where=where)

    if use_reranker is None:
        use_reranker = settings.enable_reranker
    if not use_reranker or not chunks:
        return chunks

    # Reranker recebe top-K do retrieval; entrega os max_chunks_retrieved
    # melhores. min_score=0 aqui — quem aplica threshold é o ``rag_engine``,
    # que sabe se o reranker está on/off e usa o limiar correto.
    from pipeline.reranker import rerank
    try:
        top_for_rerank = chunks[: settings.reranker_input_k]
        return rerank(
            question,
            top_for_rerank,
            top_k=settings.max_chunks_retrieved,
            min_score=0.0,
        )
    except Exception as e:  # pragma: no cover - log path
        logger.warning(f"rerank falhou, caindo pro ranking sem rerank: {e}")
        return chunks
