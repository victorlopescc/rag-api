from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from auth import require_api_key
from config import settings
from database import QueryLog, get_db
from rag_engine import ask

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    category: str | None = None
    phone_number: str | None = None  # para logar quando vier do WhatsApp


class QueryResponse(BaseModel):
    answer:       str
    was_fallback: bool
    latency_ms:   int


@router.post("", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query_rag(req: QueryRequest, db: Session = Depends(get_db)):
    """Recebe uma pergunta, consulta o RAG e retorna a resposta."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    result = ask(question=req.question, category=req.category)

    # Salva log no banco (QueryLog.chunks_used é ARRAY(Text), então
    # extraímos só os IDs dos dicts — aceita shape antigo (list[str]) também).
    chunk_ids = [
        c["id"] if isinstance(c, dict) else c
        for c in (result.chunks_used or [])
    ]
    log = QueryLog(
        phone_number=req.phone_number,
        question=req.question,
        answer=result.answer,
        chunks_used=chunk_ids,
        model_used=settings.llm_model,
        latency_ms=result.latency_ms,
        was_fallback=result.was_fallback,
    )
    db.add(log)
    db.commit()

    return QueryResponse(
        answer=result.answer,
        was_fallback=result.was_fallback,
        latency_ms=result.latency_ms,
    )