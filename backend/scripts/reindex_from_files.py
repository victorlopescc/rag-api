"""Re-indexa documentos a partir dos arquivos originais (PDF/DOCX/TXT).

Use isto em vez de ``scripts.reindex`` quando você TEM acesso aos
arquivos originais. Resultado mais limpo: re-extrai do arquivo, evita
o degrade que vem da reconstrução do texto a partir dos chunks
(overlap antigo deixa duplicações sutis que se acumulam a cada round).

Casos de uso típicos
--------------------
- Você mudou o chunker, quer reindexar do zero.
- Você suspeita que o índice está corrupto / degradado.
- Você está testando uma nova configuração e quer ground truth limpo.

Como funciona
-------------
Pra cada arquivo passado:
  1. Calcula o filename basename (sem o caminho)
  2. Procura no Postgres por um Document com esse filename
  3. Se achou: deleta os chunks dele (mantém Document + metadata)
  4. Se não achou: cria um Document novo (precisa de --category)
  5. Re-extrai texto do arquivo, re-chunka, re-embeda, re-indexa
  6. No fim, rebuilda o BM25 de uma vez só

Uso
---
  python -m scripts.reindex_from_files \\
      ~/Downloads/PPC.pdf \\
      ~/Downloads/Regulamento_TCC.pdf \\
      ~/Downloads/Resolucao_ADA.pdf

  # Forçar categoria (override pra docs novos ou recategorizar):
  python -m scripts.reindex_from_files \\
      ~/Downloads/PPC.pdf --category PPC

  # Dry-run pra ver o que faria:
  python -m scripts.reindex_from_files ~/Downloads/PPC.pdf --dry-run

Idempotente: rodar duas vezes seguidas com os mesmos arquivos dá o
mesmo resultado.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Garante backend/ no sys.path quando rodar como ``python -m scripts....``
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy.orm import Session  # noqa: E402

from database import Chunk, Document, SessionLocal  # noqa: E402
from pipeline.chunker import split_text  # noqa: E402
from pipeline.embedder import embed_batch  # noqa: E402
from pipeline.extractor import extract_text  # noqa: E402
from pipeline.vector_store import (  # noqa: E402
    delete_document_chunks,
    index_chunks,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def _find_existing(db: Session, filename: str) -> Document | None:
    return db.query(Document).filter(Document.filename == filename).first()


def _replace_chunks(
    db: Session,
    doc: Document,
    chunks_text: list[str],
    embeddings: list[list[float]],
) -> None:
    """Apaga chunks antigos e insere novos. NÃO remove o Document
    em si — só o conteúdo associado."""
    doc_id = str(doc.id)

    # Limpa antigos
    delete_document_chunks(doc_id)
    for old in list(doc.chunks):
        db.delete(old)
    db.flush()

    # Insere novos
    chroma_ids: list[str] = []
    metadatas: list[dict] = []
    db_chunks: list[Chunk] = []
    for i, (text_i, emb_i) in enumerate(zip(chunks_text, embeddings)):
        cid = f"{doc_id}_chunk_{i}"
        chroma_ids.append(cid)
        metadatas.append({
            "document_id": doc_id,
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

    index_chunks(chroma_ids, embeddings, chunks_text, metadatas)
    db.add_all(db_chunks)
    doc.total_chunks = len(db_chunks)
    doc.status = "indexed"


def _ingest_one(
    db: Session,
    file_path: Path,
    *,
    category_override: str | None,
    dry_run: bool,
) -> tuple[str, int, int]:
    """Devolve (label, chunks_antes, chunks_depois)."""
    filename = file_path.name
    file_bytes = file_path.read_bytes()
    text = extract_text(file_bytes, filename)
    if not text.strip():
        raise ValueError(f"{filename}: arquivo sem texto extraído")

    chunks_text = [c for c in split_text(text) if c and c.strip()]
    if not chunks_text:
        raise ValueError(f"{filename}: chunker produziu zero chunks")

    existing = _find_existing(db, filename)
    if existing is None:
        if not category_override:
            raise ValueError(
                f"{filename}: documento não existe no banco — passe "
                f"--category pra criar um novo registro"
            )
        n_before = 0
        action = "NEW"
    else:
        n_before = len(list(existing.chunks))
        action = "UPDATE"

    n_after = len(chunks_text)
    label = f"{action} {filename}"
    print(f"  {label}: {n_before} chunks -> {n_after} chunks")

    if dry_run:
        return (label, n_before, n_after)

    if existing is None:
        # Cria registro novo
        existing = Document(
            filename=filename,
            file_type=filename.rsplit(".", 1)[-1].lower(),
            category=category_override,
            file_size=len(file_bytes),
            status="processing",
        )
        db.add(existing)
        db.flush()  # pra ter doc.id
    else:
        if category_override and category_override != existing.category:
            print(f"    (override de categoria: {existing.category} -> {category_override})")
            existing.category = category_override
        existing.file_size = len(file_bytes)
        existing.status = "processing"
        db.flush()

    # Embeddings (etapa cara — feita só fora do dry-run)
    print(f"    gerando {n_after} embeddings...")
    embeddings = embed_batch(chunks_text)

    _replace_chunks(db, existing, chunks_text, embeddings)
    db.commit()
    return (label, n_before, n_after)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-indexa documentos a partir dos arquivos originais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", type=Path,
                   help="Caminho(s) de arquivo PDF/DOCX/TXT.")
    p.add_argument("--category", type=str, default=None,
                   help=("Categoria a aplicar a todos os arquivos. "
                         "Obrigatório quando o arquivo é novo (não existe no banco). "
                         "Quando ausente, mantém a categoria existente."))
    p.add_argument("--dry-run", action="store_true",
                   help="Não modifica nada — só mostra o que faria.")
    args = p.parse_args()

    # Validação de arquivos
    missing = [f for f in args.files if not f.is_file()]
    if missing:
        for f in missing:
            print(f"ERRO: arquivo não encontrado: {f}", file=sys.stderr)
        return 2

    db = SessionLocal()
    try:
        print(
            f"{'DRY RUN: ' if args.dry_run else ''}"
            f"reindexando {len(args.files)} arquivo(s)..."
        )
        total_before = 0
        total_after = 0
        for path in args.files:
            try:
                _, before, after = _ingest_one(
                    db, path,
                    category_override=args.category,
                    dry_run=args.dry_run,
                )
                total_before += before
                total_after += after
            except Exception as e:
                logger.error(f"{path.name}: falhou - {e}")
                db.rollback()

        print(
            f"\nResumo: {total_before} chunks antes -> {total_after} chunks depois "
            f"(delta={total_after - total_before:+d})"
        )

        if not args.dry_run:
            print("Rebuilding BM25...")
            from pipeline import bm25_index
            bm25_index.build()
            print("BM25 rebuild OK.")

        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
