"""
Gerencia a coleção vetorial no ChromaDB.
Responsável por indexar chunks e fazer busca semântica.

Suporta dois modos:
- **HTTP** (produção): conecta a um container ``chromadb/chroma`` via
  ``HttpClient`` em ``settings.chroma_host:chroma_port``. É o caminho
  default e o que usamos no docker-compose — persistência fica num
  volume Docker, fácil de backup e desacoplado do filesystem do app.
- **Embedded** (fallback): usa ``PersistentClient`` num diretório local
  (``settings.chroma_persist_path``). Ativado quando ``chroma_host``
  está vazio. Útil pra rodar testes ou dev sem subir o container.
"""

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings

logger = logging.getLogger(__name__)

# `chromadb.PersistentClient` é uma função-fábrica em versões recentes,
# não uma classe — então não dá pra usá-la em type hint com `| None`.
_chroma_client = None
COLLECTION_NAME = "rag_documents"


def _build_client():
    """Decide entre HttpClient (modo Docker) e PersistentClient (embedded).

    Se ``settings.chroma_host`` está preenchido, modo HTTP.
    Caso contrário, embedded com ``chroma_persist_path``.
    """
    host = (settings.chroma_host or "").strip()
    if host:
        logger.info(
            f"ChromaDB: HttpClient → {host}:{settings.chroma_port}"
        )
        return chromadb.HttpClient(
            host=host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    logger.info(
        f"ChromaDB: PersistentClient (embedded) → {settings.chroma_persist_path}"
    )
    return chromadb.PersistentClient(
        path=settings.chroma_persist_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection() -> chromadb.Collection:
    """Retorna a coleção ChromaDB (cria se não existir)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = _build_client()

    collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # distância cosseno para embeddings de texto
    )
    return collection


def index_chunks(
    chroma_ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """
    Insere chunks com seus embeddings e metadados na coleção.
    Se um ID já existir, ele é atualizado (upsert).
    """
    collection = get_collection()
    collection.upsert(
        ids=chroma_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def search(
    query_embedding: list[float],
    n_results: int | None = None,
    where: dict | None = None,
) -> list[dict]:
    """
    Busca os chunks mais similares ao embedding da query.
    Retorna lista de dicts com: content, metadata, distance, id.
    """
    collection = get_collection()
    n = n_results or settings.max_chunks_retrieved

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id":       results["ids"][0][i],
            "content":  results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            # Converte distância cosseno em score de similaridade (0-1)
            "score":    round(1 - results["distances"][0][i], 4),
        })

    return chunks


def delete_document_chunks(document_id: str) -> None:
    """Remove todos os chunks de um documento do ChromaDB."""
    collection = get_collection()
    collection.delete(where={"document_id": document_id})