"""Índice BM25 em memória sobre todos os chunks indexados no ChromaDB.

Por que existe
--------------
Embeddings densos (nomic-embed-text) ranqueiam por similaridade
semântica. Bom em queries onde o aluno parafraseia ("Quanto vale
a prova?" matchando "5 pontos"). Ruim em queries cujo sinal forte é
um TOKEN específico — nome próprio ("Capanema"), sigla rara ("AED"),
número de artigo, palavra-chave do documento ("estágio") — onde o
embedding pode dispersar o sinal entre vários chunks.

BM25 é o complemento natural: pesa palavras por TF-IDF + comprimento
do documento. Pega match literal sem cair no falso amigo do embedding.

Combinamos os dois via RRF (Reciprocal Rank Fusion) — uma pontuação
por POSIÇÃO em cada ranking, não por score, que é robusta às escalas
diferentes do BM25 (0..∞) e do cosseno (0..1).

Privacidade
-----------
Tudo local. ``rank-bm25`` é puro Python+numpy, sem chamada externa.

Custo
-----
Memória: ~2-5MB por mil chunks (lista de tokens + tabela de
frequências). Pra ~1000 chunks da PUC, irrelevante.
Latência: ~10-30ms por query mesmo com 10k chunks.
Build inicial: <1s pra ~1k chunks; recomendado fazer no startup.
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# Lock pro build (idempotente e thread-safe).
_build_lock = threading.Lock()

# Estado do índice. None até o primeiro build.
_bm25: Any = None
_chunk_ids: list[str] = []
_chunk_docs: list[str] = []
_chunk_metas: list[dict] = []
# Tokenização cacheada por chunk (set por chunk pra lookup O(1)).
# Usada pelo ``lexical_overlap_search``. Mantida fora do BM25Okapi
# pra evitar refactor do upstream.
_chunk_tokens: list[set[str]] = []


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")


# Tokenização simples e doc-agnóstica:
#   1. lowercase
#   2. strip de acentos (PT-BR: "questão" e "questao" tokenizam igual)
#   3. quebra em sequências alfanuméricas
#   4. drop de tokens muito curtos (1-2 chars) que viram ruído
#
# DELIBERADAMENTE não removemos stopwords aqui — BM25 já desvaloriza
# tokens de alta frequência via IDF, e PT-BR não tem uma stoplist
# canônica curta que valha a pena codar à mão. Manter simples.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    norm = _strip_accents(text.lower())
    return [t for t in _TOKEN_RE.findall(norm) if len(t) >= 3]


def build() -> None:
    """(Re)constrói o índice a partir do ChromaDB.

    Idempotente. Use após reindexar documentos pra refletir mudanças.
    Thread-safe — chamadas concorrentes durante o build esperam.
    """
    global _bm25, _chunk_ids, _chunk_docs, _chunk_metas, _chunk_tokens
    with _build_lock:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "rank-bm25 não está instalado. Rode `pip install rank-bm25` "
                "ou desligue o BM25 via ENABLE_BM25=false no .env."
            ) from e
        from pipeline.vector_store import get_collection

        col = get_collection()
        all_chunks = col.get(include=["documents", "metadatas"])
        ids = list(all_chunks.get("ids") or [])
        docs = list(all_chunks.get("documents") or [])
        metas = list(all_chunks.get("metadatas") or [])
        if not ids:
            logger.warning("BM25 build: ChromaDB vazio. Índice marcado como pronto mas vazio.")
            _bm25 = None
            _chunk_ids = []
            _chunk_docs = []
            _chunk_metas = []
            _chunk_tokens = []
            return
        tokenized = [tokenize(d) for d in docs]
        _bm25 = BM25Okapi(tokenized)
        _chunk_ids = ids
        _chunk_docs = docs
        _chunk_metas = metas
        _chunk_tokens = [set(t) for t in tokenized]
        logger.info(f"BM25 indexado: {len(ids)} chunks.")


def is_built() -> bool:
    return _bm25 is not None and len(_chunk_ids) > 0


def search(query: str, *, n_results: int) -> list[dict]:
    """Retorna os top-N chunks por score BM25 (descendente).

    Saída no MESMO formato da busca densa em ``vector_store.search``:
    ``[{"id", "content", "metadata", "distance", "score"}, ...]``,
    pra que o caller possa fundir os dois rankings sem branching.
    O ``distance`` é definido como ``-score`` (BM25 não tem noção
    de distância; manter o campo mantém o shape consistente).

    Se o índice não foi construído, retorna lista vazia em vez de
    levantar — chamada concorrente ao build não bloqueia o caminho
    quente. ``warmup()`` no startup garante o caso comum.
    """
    if _bm25 is None or not _chunk_ids:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = _bm25.get_scores(tokens)
    # Top-N por score
    if len(scores) <= n_results:
        idxs = list(range(len(scores)))
    else:
        # argsort decrescente, top n_results
        # (sem numpy: enumera + sort. ~0.5ms pra 1k chunks.)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    out: list[dict] = []
    for i in idxs:
        s = float(scores[i])
        if s <= 0:
            continue
        out.append({
            "id": _chunk_ids[i],
            "content": _chunk_docs[i],
            "metadata": _chunk_metas[i],
            "distance": -s,
            "score": s,
        })
    return out


def lexical_overlap_search(query: str, *, n_results: int) -> list[dict]:
    """Ranking complementar ao BM25: ordena chunks por OVERLAP CRU
    de tokens distintos da query (sem TF-IDF).

    Por que existe além do BM25
    ---------------------------
    BM25 normaliza por TF-IDF — bom em corpora grandes e diversos, mas
    em corpus pequeno onde um termo aparece em muitos chunks (ex.: "ADA"
    em 100% dos chunks da resolução ADA), o IDF colapsa pra ~zero e o
    sinal some. Já o overlap cru não tem esse colapso: chunks com mais
    tokens distintos da query rankeiam acima, ponto.

    Por outro lado, overlap cru é ingênuo em corpora grandes (favorece
    chunks longos). Daí a combinação dos dois via RRF: BM25 puxa em
    docs grandes, overlap puxa em docs pequenos.

    Saída no MESMO formato de ``search()`` pra fundir nos rankings.
    """
    if not _chunk_tokens:
        return []
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []
    scored: list[tuple[int, int]] = []
    for i, ct in enumerate(_chunk_tokens):
        overlap = len(query_tokens & ct)
        if overlap > 0:
            scored.append((overlap, i))
    if not scored:
        return []
    # Ordena por overlap desc; sort do Python é estável → ties mantêm
    # a ordem do índice (chunks anteriores no doc primeiro, decisão
    # arbitrária mas determinística).
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for overlap, i in scored[:n_results]:
        out.append({
            "id": _chunk_ids[i],
            "content": _chunk_docs[i],
            "metadata": _chunk_metas[i],
            "distance": -float(overlap),
            "score": float(overlap),
        })
    return out


def warmup() -> None:
    """Constrói o índice no startup pra evitar custo na primeira request.

    No-op se BM25 está desabilitado na config.
    """
    from config import settings
    if not settings.enable_bm25:
        return
    try:
        build()
    except Exception as e:  # pragma: no cover
        logger.warning(f"BM25 warmup falhou (retrieval cai pra denso-only): {e}")
