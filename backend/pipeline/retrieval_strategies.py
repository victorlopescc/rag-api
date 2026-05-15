"""Estratégias de recuperação usadas nas tentativas 1–3 da sessão.

A tentativa N escolhe uma estratégia diferente para aumentar a chance
de achar um chunk relevante mesmo quando o RAG "default" falhou.

- ``default``       — retrieval híbrido (BM25 + denso) com variantes de
                       acento/sigla, fundido por RRF, opcionalmente
                       filtrado por categoria do documento.
- ``query_rewrite`` — LLM reformula a pergunta em 3 variantes; mesma
                       busca híbrida em todas, união pelo RRF.
- ``widen_k``       — descarta filtro de categoria, mantém híbrido.

Cada função devolve uma lista de chunks ``{id, score, ...}`` no mesmo
formato de ``pipeline.vector_store.search``.

Histórico
---------
Versões anteriores tinham:
  - ``_CONCEPT_EXPANSIONS``: regex pattern → tokens, hardcoded por
    domínio ("quanto vale" → "pontos", etc.).
  - ``_lexical_boost``: bônus manual por overlap de keywords.
  - ``CATEGORY_BOOST``: bônus por chunk vir da busca filtrada por
    categoria detectada via siglas/keywords.
Todas essas peças resolviam o sintoma específico de queries onde
o aluno usa vocabulário diferente do documento, mas eram
band-aids: cada novo documento exigia editar regex e listas de
keywords.

Foram REMOVIDAS e substituídas por BM25 (lexical) + RRF (fusão).
BM25 pega match literal de qualquer token (incluindo siglas, nomes
próprios e números de artigo) sem nenhum hardcoding por documento.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import unicodedata
from typing import Literal

from config import settings
from pipeline.acronyms import detect_categories, query_variants
from pipeline.embedder import embed_text
from pipeline.llm import generate
from pipeline.vector_store import search

logger = logging.getLogger(__name__)

Strategy = Literal["default", "query_rewrite", "widen_k"]
STRATEGIES: tuple[Strategy, ...] = ("default", "query_rewrite", "widen_k")


def strategy_for_attempt(attempt_number: int) -> Strategy:
    """Mapeamento 1→default, 2→query_rewrite, 3→widen_k (clamp)."""
    idx = max(1, min(attempt_number, 3)) - 1
    return STRATEGIES[idx]


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
    """Busca híbrida: denso (com variantes de acento e sigla) + BM25,
    com boost via busca extra restrita à categoria detectada na query
    (se houver sigla). Tudo fundido por RRF.

    Sinais combinados:
      - Denso (nomic-embed-text): paráfrase e semântica geral.
      - BM25: match literal — nomes próprios, siglas raras, tokens-chave
              que o embedding dilui.
      - Categoria detectada: quando a query menciona uma sigla registrada
              (ADA, TCC, PPC, ...), uma busca EXTRA restrita aos chunks
              daquela categoria entra no RRF. Compensa o problema de
              docs pequenos (poucos chunks) sumirem na busca global pra
              docs grandes (PPC com ~970 chunks). Doc-agnóstico: pra
              novo doc ser priorizado, basta registrar sua sigla em
              ``ACRONYM_TO_CATEGORY``.
    """
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
# default
# ---------------------------------------------------------------------------

def retrieve_default(question: str, category: str | None) -> list[dict]:
    where = {"category": category} if category else None
    return _hybrid_search(question, n_results=None, where=where)


# ---------------------------------------------------------------------------
# query_rewrite
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = (
    "Você ajuda um sistema de busca semântica. Dada a pergunta original "
    "de um aluno, gere 3 reformulações diferentes que preservem o "
    "significado mas variem vocabulário e estrutura. Inclua sinônimos "
    "acadêmicos quando fizer sentido.\n\n"
    "Responda APENAS um JSON no formato: "
    '{"rewrites": ["...", "...", "..."]}\n\n'
    "Pergunta original: {question}\n\n"
    "JSON:"
)

_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_rewrites(raw: str) -> list[str]:
    m = _JSON_BLOCK_RE.search(raw or "")
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = obj.get("rewrites") or []
    if not isinstance(out, list):
        return []
    return [str(x).strip() for x in out if isinstance(x, (str, int, float)) and str(x).strip()]


def rewrite_query(question: str) -> list[str]:
    """Chama o LLM para gerar variações. Falha → lista vazia (sem quebrar)."""
    try:
        raw = generate(_REWRITE_PROMPT.replace("{question}", question))
    except Exception as e:  # pragma: no cover - log path
        logger.warning(f"query_rewrite falhou: {e}")
        return []
    rewrites = _parse_rewrites(raw)
    logger.info(f"query_rewrite gerou {len(rewrites)} variações")
    return rewrites[:3]


def retrieve_with_query_rewrite(question: str, category: str | None) -> list[dict]:
    """Busca com a pergunta original + até 3 variações. Fusão RRF."""
    where = {"category": category} if category else None
    rankings: list[list[dict]] = [_hybrid_search(question, n_results=None, where=where)]
    for variant in rewrite_query(question):
        try:
            rankings.append(_hybrid_search(variant, n_results=None, where=where))
        except Exception as e:  # pragma: no cover
            logger.warning(f"busca da variante falhou: {e}")
    return _rrf_fuse(rankings)


# ---------------------------------------------------------------------------
# widen_k
# ---------------------------------------------------------------------------

WIDEN_K_CAP = 10


def retrieve_with_widen_k(question: str, _category: str | None) -> list[dict]:
    """Aumenta k e remove filtro de categoria (busca em tudo)."""
    k = min(WIDEN_K_CAP, max(settings.max_chunks_retrieved * 2, settings.max_chunks_retrieved))
    return _hybrid_search(question, n_results=k, where=None)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, str] = {
    "default": "retrieve_default",
    "query_rewrite": "retrieve_with_query_rewrite",
    "widen_k": "retrieve_with_widen_k",
}


def retrieve(
    question: str,
    *,
    category: str | None = None,
    strategy: Strategy = "default",
    use_reranker: bool | None = None,
) -> list[dict]:
    """Recupera chunks usando a estratégia escolhida e (opcionalmente)
    reranqueia com cross-encoder.

    ``use_reranker``: None → segue ``settings.enable_reranker``. True/False
    força o comportamento (útil pro eval harness comparar A/B).
    """
    fn_name = _DISPATCH.get(strategy, "retrieve_default")
    fn = getattr(sys.modules[__name__], fn_name)
    chunks = fn(question, category)

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
