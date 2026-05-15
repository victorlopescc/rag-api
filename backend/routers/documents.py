from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import require_api_key
from database import Chunk, Document, get_db
from pipeline.acronyms import ACRONYM_TO_CATEGORY, suggest_category
from pipeline.extractor import extract_text
from pipeline.ingestor import delete_document, ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    id:           str
    filename:     str
    file_type:    str
    category:     str | None
    description:  str | None
    file_size:    int | None
    total_chunks: int
    status:       str
    error_msg:    str | None
    created_at:   datetime

    class Config:
        from_attributes = True


class CategorySuggestion(BaseModel):
    suggested_category: str | None
    available_categories: list[str]


class ChunkPreview(BaseModel):
    chunk_index: int
    content: str
    char_count: int
    token_count: int | None


class ChunksPage(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    offset: int
    limit: int
    chunks: list[ChunkPreview]


@router.post("", response_model=DocumentOut, dependencies=[Depends(require_api_key)])
async def upload_document(
    file:        UploadFile = File(...),
    category:    str | None = Form(None),
    description: str | None = Form(None),
    db:          Session    = Depends(get_db),
):
    """Recebe um arquivo, aciona o pipeline de ingestão e retorna o documento criado."""
    allowed = {"pdf", "docx", "txt"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Tipo não suportado. Use: {allowed}")

    file_bytes = await file.read()
    doc = ingest_document(
        db=db,
        file_bytes=file_bytes,
        filename=file.filename,
        category=category,
        description=description,
    )
    return _to_out(doc)


@router.get("", response_model=list[DocumentOut], dependencies=[Depends(require_api_key)])
def list_documents(db: Session = Depends(get_db)):
    """Lista todos os documentos indexados."""
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [_to_out(d) for d in docs]


@router.post(
    "/suggest-category",
    response_model=CategorySuggestion,
    dependencies=[Depends(require_api_key)],
)
async def suggest_category_for_file(file: UploadFile = File(...)):
    """Recebe um arquivo e sugere uma categoria baseada no nome e
    no scan de siglas registradas no texto. NÃO indexa nada.

    Útil pro frontend mostrar uma sugestão pré-preenchida no formulário
    de upload — o coordenador pode aceitar ou trocar antes de subir
    de fato. Se a sugestão for None, é porque nenhuma sigla registrada
    aparece com força suficiente no doc; o coordenador escolhe na mão.
    """
    allowed = {"pdf", "docx", "txt"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Tipo não suportado. Use: {allowed}")

    file_bytes = await file.read()
    try:
        text = extract_text(file_bytes, file.filename)
    except Exception:
        text = ""

    return CategorySuggestion(
        suggested_category=suggest_category(file.filename, text),
        available_categories=sorted(set(ACRONYM_TO_CATEGORY.values())),
    )


@router.get(
    "/{doc_id}/chunks",
    response_model=ChunksPage,
    dependencies=[Depends(require_api_key)],
)
def list_document_chunks(
    doc_id: str,
    db: Session = Depends(get_db),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
):
    """Devolve uma página de chunks de um documento, ordenada por
    ``chunk_index``. Útil pro coordenador verificar como o chunker
    fragmentou o doc — ajuda a debugar problemas de retrieval (chunk
    muito grande / muito pequeno / cortado no meio de uma frase).
    """
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    total = (
        db.query(Chunk)
        .filter(Chunk.document_id == doc.id)
        .count()
    )
    rows = (
        db.query(Chunk)
        .filter(Chunk.document_id == doc.id)
        .order_by(Chunk.chunk_index.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return ChunksPage(
        document_id=str(doc.id),
        filename=doc.filename,
        total_chunks=total,
        offset=offset,
        limit=limit,
        chunks=[
            ChunkPreview(
                chunk_index=c.chunk_index,
                content=c.content,
                char_count=len(c.content or ""),
                token_count=c.token_count,
            )
            for c in rows
        ],
    )


@router.delete("/{doc_id}", dependencies=[Depends(require_api_key)])
def remove_document(doc_id: str, db: Session = Depends(get_db)):
    """Remove documento do PostgreSQL e do ChromaDB."""
    deleted = delete_document(db, doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Documento não encontrado.")
    return {"detail": "Documento removido com sucesso."}


def _to_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=str(doc.id),
        filename=doc.filename,
        file_type=doc.file_type,
        category=doc.category,
        description=doc.description,
        file_size=doc.file_size,
        total_chunks=doc.total_chunks or 0,
        status=doc.status,
        error_msg=doc.error_msg,
        created_at=doc.created_at,
    )