"""Re-indexa todos os documentos com o chunker atual.

Útil quando você muda o chunker (estrutural vs recursivo, tamanhos,
overlap) e precisa que os chunks no Chroma reflitam isso. Sem isso,
os chunks antigos ficam no índice indefinidamente.

Como funciona
-------------
Pra cada documento ``indexed`` no Postgres:
  1. Lê os chunks existentes ordenados por ``chunk_index``
  2. Reconstrói o texto original juntando os chunks e DEDUPLICANDO
     o overlap (o overlap velho era ``last_N(chunk[i])`` prepended
     em ``chunk[i+1]`` — busca a maior sobreposição suffix↔prefix).
  3. Re-chunkifica com o chunker atual (``split_text``)
  4. Gera embeddings novos
  5. APAGA os chunks antigos (Postgres cascade + Chroma)
  6. INSERE os chunks novos
  7. Rebuilda o índice BM25

Uso
---
  python -m scripts.reindex            # reindexa todos
  python -m scripts.reindex --dry-run  # mostra o que faria
  python -m scripts.reindex --doc <ID> # apenas um doc
  python -m scripts.reindex --no-bm25  # pula rebuild BM25 (faz no fim)

Idempotente: rodar duas vezes seguidas dá o mesmo resultado.
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

# Garante que ``backend/`` está no sys.path (script roda como módulo).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy.orm import Session  # noqa: E402

from database import Chunk, Document, SessionLocal  # noqa: E402
from pipeline.chunker import split_text  # noqa: E402
from pipeline.embedder import embed_batch  # noqa: E402
from pipeline.vector_store import (  # noqa: E402
    delete_document_chunks,
    index_chunks,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ---------------------------------------------------------------------------
# Reconstrução de texto a partir dos chunks
# ---------------------------------------------------------------------------

def _longest_overlap(prev: str, curr: str, max_overlap: int = 200) -> int:
    """Maior N tal que ``prev[-N:] == curr[:N]``. 0 se nenhum.

    Limitado a ``max_overlap`` pra evitar falso positivo em chunks
    muito parecidos no meio (improvável mas defensivo).
    """
    n = min(len(prev), len(curr), max_overlap)
    while n > 0:
        if prev[-n:] == curr[:n]:
            return n
        n -= 1
    return 0


def _reconstruct_text(chunks: list[Chunk]) -> str:
    """Junta os chunks ordenados por chunk_index, deduplicando overlap.

    Não recupera EXATAMENTE o texto original (perdemos alguns
    quebras-de-linha quando o split passou pelo separador "\\n"),
    mas mantém a ordem e o conteúdo. O novo chunker aceita perfeitamente
    — vai detectar marcadores estruturais nos lugares certos.
    """
    if not chunks:
        return ""
    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)
    parts: list[str] = [sorted_chunks[0].content]
    for prev, curr in zip(sorted_chunks, sorted_chunks[1:]):
        n = _longest_overlap(prev.content, curr.content)
        parts.append(curr.content[n:])
    # Junção simples — a perda de alguns "\n" entre parágrafos não
    # afeta o chunker estrutural (ele detecta marcadores em qualquer
    # posição de linha, e os marcadores ficam nas mesmas linhas).
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Re-ingestão de um documento
# ---------------------------------------------------------------------------

def _reindex_one(db: Session, doc: Document, *, dry_run: bool) -> tuple[int, int]:
    """Re-indexa um documento. Devolve (chunks_antes, chunks_depois)."""
    old_chunks = list(doc.chunks)
    text = _reconstruct_text(old_chunks)
    if not text.strip():
        logger.warning(f"[{doc.id}] sem texto reconstruído — pulando.")
        return (len(old_chunks), len(old_chunks))

    new_chunks_text = split_text(text)
    new_chunks_text = [c for c in new_chunks_text if c and c.strip()]
    if not new_chunks_text:
        logger.warning(f"[{doc.id}] novo chunker produziu zero chunks — pulando.")
        return (len(old_chunks), len(old_chunks))

    n_before = len(old_chunks)
    n_after = len(new_chunks_text)

    logger.info(
        f"[{doc.id}] {doc.filename}: {n_before} chunks -> {n_after} chunks "
        f"(delta={n_after - n_before:+d})"
    )

    if dry_run:
        return (n_before, n_after)

    # Embeddings
    logger.info(f"[{doc.id}] gerando {n_after} embeddings...")
    embeddings = embed_batch(new_chunks_text)

    # Limpa antigos (Postgres cascade + Chroma)
    doc_id_str = str(doc.id)
    delete_document_chunks(doc_id_str)
    for old in old_chunks:
        db.delete(old)
    db.flush()

    # Insere novos
    chroma_ids: list[str] = []
    metadatas: list[dict] = []
    db_chunks: list[Chunk] = []
    for i, (text_i, emb_i) in enumerate(zip(new_chunks_text, embeddings)):
        cid = f"{doc_id_str}_chunk_{i}"
        chroma_ids.append(cid)
        metadatas.append({
            "document_id": doc_id_str,
            "filename":    doc.filename,
            "category":    doc.category or "",
            "chunk_index": i,
        })
        db_chunks.append(Chunk(
            document_id=doc.id,
            chunk_index=i,
            content=text_i,
            chroma_id=cid,
            token_count=len(text_i.split()),
        ))

    index_chunks(chroma_ids, embeddings, new_chunks_text, metadatas)
    db.add_all(db_chunks)
    doc.total_chunks = n_after
    db.commit()
    logger.info(f"[{doc.id}] re-indexado.")
    return (n_before, n_after)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_uuid(s: str) -> uuid.UUID:
    try:
        return uuid.UUID(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"UUID inválido: {s}") from e


def main() -> int:
    p = argparse.ArgumentParser(description="Re-indexa documentos com o chunker atual.")
    p.add_argument("--dry-run", action="store_true",
                   help="Não modifica nada — só mostra o delta de chunks.")
    p.add_argument("--doc", type=_parse_uuid, default=None,
                   help="UUID de um documento específico. Default: todos.")
    p.add_argument("--no-bm25", action="store_true",
                   help="Pula rebuild do BM25 ao final (dispare manualmente depois).")
    args = p.parse_args()

    db = SessionLocal()
    try:
        q = db.query(Document).filter(Document.status == "indexed")
        if args.doc:
            q = q.filter(Document.id == args.doc)
        docs = q.all()
        if not docs:
            print("Nenhum documento indexado encontrado.")
            return 0

        print(f"{'DRY RUN: ' if args.dry_run else ''}reindexando {len(docs)} doc(s)...")
        total_before = 0
        total_after = 0
        for doc in docs:
            try:
                before, after = _reindex_one(db, doc, dry_run=args.dry_run)
                total_before += before
                total_after += after
            except Exception as e:  # pragma: no cover
                logger.error(f"[{doc.id}] falhou: {e}")
                db.rollback()

        print(
            f"\nResumo: {total_before} chunks antes -> {total_after} chunks depois "
            f"(delta={total_after - total_before:+d})"
        )

        if not args.dry_run and not args.no_bm25:
            print("Rebuilding BM25...")
            from pipeline import bm25_index
            bm25_index.build()
            print("BM25 rebuild OK.")

        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
