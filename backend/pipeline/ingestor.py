"""
Orquestra o pipeline completo de ingestão de um documento:
  upload → extração → chunking → embedding → ChromaDB + PostgreSQL
"""

import uuid
import logging
from sqlalchemy.orm import Session

from database import Document, Chunk
from pipeline.extractor import extract_text
from pipeline.chunker import split_text
from pipeline.embedder import embed_batch
from pipeline.vector_store import index_chunks, delete_document_chunks

logger = logging.getLogger(__name__)


def ingest_document(
    db: Session,
    file_bytes: bytes,
    filename: str,
    category: str | None = None,
    description: str | None = None,
) -> Document:
    """
    Pipeline completo de ingestão.
    Cria o registro no banco, processa o arquivo e indexa no ChromaDB.
    Atualiza o status do documento em cada etapa.

    Quando ``category`` é None ou vazia, tenta auto-detectar via
    ``suggest_category`` (sigla registrada em ``ACRONYM_TO_CATEGORY``).
    O coordenador pode override depois pelo painel admin.
    """

    # 1. Cria registro no PostgreSQL com status "processing"
    doc = Document(
        filename=filename,
        file_type=filename.rsplit(".", 1)[-1].lower(),
        category=category,
        description=description,
        file_size=len(file_bytes),
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = str(doc.id)
    logger.info(f"[{doc_id}] Documento criado: {filename}")

    try:
        # 2. Extrai texto do arquivo
        logger.info(f"[{doc_id}] Extraindo texto...")
        text = extract_text(file_bytes, filename)
        if not text.strip():
            raise ValueError("Nenhum texto encontrado no documento.")

        # 2b. Se o coordenador não informou categoria, tenta auto-detectar.
        # Nunca sobrescreve uma categoria explicitamente fornecida.
        if not (category or "").strip():
            from pipeline.acronyms import suggest_category
            suggestion = suggest_category(filename, text)
            if suggestion:
                logger.info(
                    f"[{doc_id}] Categoria auto-detectada: '{suggestion}' "
                    f"(filename + scan de siglas no texto)."
                )
                doc.category = suggestion
                db.commit()
                db.refresh(doc)
                # Atualiza a variável local pra que os metadatas dos
                # chunks (gravados mais abaixo) carreguem a categoria
                # correta — sem isso o filtro por categoria no retrieval
                # não enxergaria o doc.
                category = suggestion

        # 3. Divide em chunks
        logger.info(f"[{doc_id}] Gerando chunks...")
        chunks_text = split_text(text)
        # Filtra chunks vazios / whitespace-only — Ollama responde 500 neles
        # e o lote inteiro vai pro lixo. Acontece em PDFs com cabeçalhos
        # repetidos / paginas só com imagem.
        before = len(chunks_text)
        chunks_text = [c for c in chunks_text if c and c.strip()]
        skipped = before - len(chunks_text)
        if skipped:
            logger.warning(
                f"[{doc_id}] {skipped} chunks vazios descartados antes do embedding."
            )
        if not chunks_text:
            raise ValueError("Todos os chunks ficaram vazios após filtragem.")
        logger.info(f"[{doc_id}] {len(chunks_text)} chunks gerados.")

        # 4. Gera embeddings (pode demorar para docs grandes)
        logger.info(f"[{doc_id}] Gerando embeddings via Ollama...")
        embeddings = embed_batch(chunks_text)

        # 5. Prepara dados para ChromaDB e PostgreSQL
        chroma_ids = []
        metadatas  = []
        db_chunks  = []

        for i, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings)):
            chroma_id = f"{doc_id}_chunk_{i}"
            chroma_ids.append(chroma_id)
            metadatas.append({
                "document_id": doc_id,
                "filename":    filename,
                "category":    category or "",
                "chunk_index": i,
            })
            db_chunks.append(Chunk(
                document_id=doc.id,
                chunk_index=i,
                content=chunk_text,
                chroma_id=chroma_id,
                token_count=len(chunk_text.split()),
            ))

        # 6. Indexa no ChromaDB
        logger.info(f"[{doc_id}] Indexando no ChromaDB...")
        index_chunks(chroma_ids, embeddings, chunks_text, metadatas)

        # 7. Salva chunks no PostgreSQL
        db.add_all(db_chunks)
        doc.total_chunks = len(db_chunks)
        doc.status = "indexed"
        db.commit()
        db.refresh(doc)

        # 8. Rebuilda o índice BM25 (in-memory) pra refletir os novos
        # chunks. Best-effort — se falhar, retrieval ainda funciona via
        # busca densa, só perde o sinal lexical até o próximo restart.
        try:
            from pipeline import bm25_index
            bm25_index.build()
        except Exception as e:  # pragma: no cover
            logger.warning(f"[{doc_id}] BM25 rebuild falhou: {e}")

        logger.info(f"[{doc_id}] Ingestão concluída — {len(db_chunks)} chunks indexados.")
        return doc

    except Exception as e:
        # Em caso de erro, marca o documento e remove do ChromaDB (se parcialmente indexado)
        logger.error(f"[{doc_id}] Erro na ingestão: {e}")
        doc.status = "error"
        doc.error_msg = str(e)
        db.commit()
        try:
            delete_document_chunks(doc_id)
        except Exception:
            pass
        raise


def delete_document(db: Session, doc_id: str) -> bool:
    """
    Remove documento do PostgreSQL (cascade deleta os chunks)
    e do ChromaDB.
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return False

    delete_document_chunks(doc_id)
    db.delete(doc)
    db.commit()
    # Rebuilda BM25 pra refletir a remoção. Best-effort.
    try:
        from pipeline import bm25_index
        bm25_index.build()
    except Exception as e:  # pragma: no cover
        logger.warning(f"BM25 rebuild após delete falhou: {e}")
    logger.info(f"Documento {doc_id} removido.")
    return True