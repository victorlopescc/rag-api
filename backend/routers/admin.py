"""Endpoints do painel do coordenador (milestone 3).

Todas as rotas são protegidas por ``X-API-Key`` (mesma chave usada para
deletar alunos). O frontend Mantine consome essas rotas para:

- listar escalações pendentes / histórico;
- ver o detalhe (histórico completo da sessão + tentativas do bot);
- atualizar rótulos / notas do coordenador (dados da tese);
- rodar manutenção (fechar sessões ociosas / fim-de-dia) manualmente.

O envio da resposta do coordenador ao aluno é milestone 4.
"""
from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import require_api_key
from database import Escalation, QAAttempt, QASession, Student, ThreadMessage, get_db, utcnow
from services import maintenance, thread_service
from services.evolution_client import evolution_client
from services.thread_service import ThreadConflictError, ThreadStateError
from services.whatsapp import (
    COORDINATOR_PREFIX,
    THREAD_CLOSED_BY_COORDINATOR_NOTICE,
    THREAD_OPENED_NOTICE,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_api_key)],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

EscalationStatus = Literal[
    "pending", "coordinator_replied", "resolved_by_bot_later",
    "live", "resolved", "abandoned",
]
CoordinatorLabel = Literal[
    "bot_was_wrong", "missing_document", "student_misunderstood", "other"
]


class StudentLite(BaseModel):
    id: str
    full_name: str
    matricula: str
    phone_number: str


class EscalationListItem(BaseModel):
    id: str
    status: EscalationStatus
    summary: str
    student: StudentLite
    session_id: str
    created_at: str
    replied_at: str | None
    coordinator_label: CoordinatorLabel | None


class AttemptOut(BaseModel):
    attempt_number: int
    question: str
    answer: str
    retrieval_strategy: str
    was_fallback: bool
    feedback_signal: str | None
    created_at: str


class EscalationDetail(EscalationListItem):
    coordinator_reply: str | None
    coordinator_notes: str | None
    closing_feedback: str | None
    attempts: list[AttemptOut]
    # Campos da live thread (todos None enquanto a thread nunca foi aberta).
    live_opened_at: str | None
    live_closed_at: str | None
    last_activity_at: str | None


class ThreadMessageOut(BaseModel):
    id: str
    direction: Literal["student", "coordinator"]
    text: str
    sent_at: str


class ThreadView(BaseModel):
    escalation_id: str
    status: EscalationStatus
    live_opened_at: str | None
    live_closed_at: str | None
    last_activity_at: str | None
    messages: list[ThreadMessageOut]


class ThreadSendPayload(BaseModel):
    text: str


class EscalationPatch(BaseModel):
    status: EscalationStatus | None = None
    coordinator_label: CoordinatorLabel | None = None
    coordinator_notes: str | None = None
    coordinator_reply: str | None = None


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------

def _student_lite(s: Student) -> StudentLite:
    return StudentLite(
        id=str(s.id),
        full_name=s.full_name,
        matricula=s.matricula,
        phone_number=s.phone_number,
    )


def _attempt_out(a: QAAttempt) -> AttemptOut:
    return AttemptOut(
        attempt_number=a.attempt_number,
        question=a.question,
        answer=a.answer,
        retrieval_strategy=a.retrieval_strategy,
        was_fallback=bool(a.was_fallback),
        feedback_signal=a.feedback_signal,
        created_at=a.created_at.isoformat() if a.created_at else "",
    )


def _list_item(e: Escalation, student: Student) -> EscalationListItem:
    return EscalationListItem(
        id=str(e.id),
        status=e.status,
        summary=e.summary,
        student=_student_lite(student),
        session_id=str(e.session_id),
        created_at=e.created_at.isoformat() if e.created_at else "",
        replied_at=e.replied_at.isoformat() if e.replied_at else None,
        coordinator_label=e.coordinator_label,
    )


def _detail(
    e: Escalation, student: Student, session: QASession
) -> EscalationDetail:
    attempts = sorted(session.attempts or [], key=lambda x: x.attempt_number)
    base = _list_item(e, student).model_dump()
    return EscalationDetail(
        **base,
        coordinator_reply=e.coordinator_reply,
        coordinator_notes=e.coordinator_notes,
        closing_feedback=session.closing_feedback,
        attempts=[_attempt_out(a) for a in attempts],
        live_opened_at=e.live_opened_at.isoformat() if e.live_opened_at else None,
        live_closed_at=e.live_closed_at.isoformat() if e.live_closed_at else None,
        last_activity_at=e.last_activity_at.isoformat() if e.last_activity_at else None,
    )


def _thread_msg_out(m: ThreadMessage) -> ThreadMessageOut:
    return ThreadMessageOut(
        id=str(m.id),
        direction=m.direction,
        text=m.text,
        sent_at=m.sent_at.isoformat() if m.sent_at else "",
    )


def _thread_view(e: Escalation, msgs: list[ThreadMessage]) -> ThreadView:
    return ThreadView(
        escalation_id=str(e.id),
        status=e.status,
        live_opened_at=e.live_opened_at.isoformat() if e.live_opened_at else None,
        live_closed_at=e.live_closed_at.isoformat() if e.live_closed_at else None,
        last_activity_at=e.last_activity_at.isoformat() if e.last_activity_at else None,
        messages=[_thread_msg_out(m) for m in msgs],
    )


def _get_escalation_or_404(db: Session, escalation_id: uuid.UUID) -> Escalation:
    e = db.query(Escalation).filter(Escalation.id == escalation_id).first()
    if e is None:
        raise HTTPException(status_code=404, detail="Escalação não encontrada.")
    return e


# ---------------------------------------------------------------------------
# Escalations — list / detail / patch
# ---------------------------------------------------------------------------

@router.get("/escalations", response_model=list[EscalationListItem])
def list_escalations(
    status: EscalationStatus | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Lista escalações. Use ``?status=pending`` para a caixa de entrada."""
    limit = max(1, min(limit, 200))
    q = db.query(Escalation, Student).join(Student, Escalation.student_id == Student.id)
    if status is not None:
        q = q.filter(Escalation.status == status)
    rows = q.order_by(Escalation.created_at.desc()).limit(limit).all()
    return [_list_item(e, s) for e, s in rows]


@router.get("/escalations/{escalation_id}", response_model=EscalationDetail)
def get_escalation(escalation_id: uuid.UUID, db: Session = Depends(get_db)):
    e = db.query(Escalation).filter(Escalation.id == escalation_id).first()
    if e is None:
        raise HTTPException(status_code=404, detail="Escalação não encontrada.")
    student = db.query(Student).filter(Student.id == e.student_id).first()
    session = db.query(QASession).filter(QASession.id == e.session_id).first()
    if student is None or session is None:
        raise HTTPException(status_code=500, detail="Dados inconsistentes.")
    return _detail(e, student, session)


@router.patch("/escalations/{escalation_id}", response_model=EscalationDetail)
def patch_escalation(
    escalation_id: uuid.UUID, patch: EscalationPatch, db: Session = Depends(get_db)
):
    """Atualiza rótulos / notas / status da escalação.

    Se ``coordinator_reply`` vier preenchido, marca ``replied_at``.
    (O envio efetivo da mensagem ao aluno é milestone 4.)
    """
    e = db.query(Escalation).filter(Escalation.id == escalation_id).first()
    if e is None:
        raise HTTPException(status_code=404, detail="Escalação não encontrada.")

    if patch.status is not None:
        e.status = patch.status
    if patch.coordinator_label is not None:
        e.coordinator_label = patch.coordinator_label
    if patch.coordinator_notes is not None:
        e.coordinator_notes = patch.coordinator_notes
    if patch.coordinator_reply is not None:
        e.coordinator_reply = patch.coordinator_reply
        if e.replied_at is None:
            e.replied_at = utcnow()

    db.commit()
    db.refresh(e)

    student = db.query(Student).filter(Student.id == e.student_id).first()
    session = db.query(QASession).filter(QASession.id == e.session_id).first()
    return _detail(e, student, session)


# ---------------------------------------------------------------------------
# Reply — envia a resposta do coordenador ao aluno via WhatsApp (milestone 4)
# ---------------------------------------------------------------------------

class EscalationReply(BaseModel):
    message: str
    coordinator_label: CoordinatorLabel | None = None
    coordinator_notes: str | None = None


@router.post("/escalations/{escalation_id}/reply", response_model=EscalationDetail)
async def reply_escalation(
    escalation_id: uuid.UUID,
    payload: EscalationReply,
    db: Session = Depends(get_db),
):
    """Envia a resposta do coordenador ao aluno via WhatsApp e persiste.

    - Se a escalação já estiver com status ``coordinator_replied`` responde 409
      para evitar duplo envio (o admin pode atualizar notas via PATCH).
    - Se o envio via Evolution falhar, a transação é revertida e responde 502.
    """
    text = (payload.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    e = db.query(Escalation).filter(Escalation.id == escalation_id).first()
    if e is None:
        raise HTTPException(status_code=404, detail="Escalação não encontrada.")

    if e.status == "coordinator_replied":
        raise HTTPException(
            status_code=409,
            detail="Esta escalação já foi respondida pelo coordenador.",
        )

    student = db.query(Student).filter(Student.id == e.student_id).first()
    session = db.query(QASession).filter(QASession.id == e.session_id).first()
    if student is None or session is None:
        raise HTTPException(status_code=500, detail="Dados inconsistentes.")

    prefixed = f"👨‍🏫 *Coordenação:*\n\n{text}"
    try:
        msg_id = await evolution_client.send_text(student.phone_number, prefixed)
    except Exception as ex:
        logger.exception(f"Erro enviando resposta ao aluno: {ex}")
        raise HTTPException(
            status_code=502,
            detail="Falha ao enviar via WhatsApp. Tente novamente.",
        )
    if msg_id is None:
        raise HTTPException(
            status_code=502,
            detail="Falha ao enviar via WhatsApp. Tente novamente.",
        )

    e.coordinator_reply = text
    e.status = "coordinator_replied"
    e.replied_at = utcnow()
    if payload.coordinator_label is not None:
        e.coordinator_label = payload.coordinator_label
    if payload.coordinator_notes is not None:
        e.coordinator_notes = payload.coordinator_notes

    db.commit()
    db.refresh(e)

    return _detail(e, student, session)


# ---------------------------------------------------------------------------
# Live thread (conversa ao vivo aluno ↔ coordenador)
# ---------------------------------------------------------------------------

@router.get("/escalations/{escalation_id}/thread", response_model=ThreadView)
def get_thread(escalation_id: uuid.UUID, db: Session = Depends(get_db)):
    """Devolve o estado da thread e as mensagens em ordem cronológica.

    Funciona em qualquer status — durante ``live``, o painel pode chamar
    em polling pra ver mensagens novas do aluno. Após ``resolved`` ou
    ``abandoned``, devolve o histórico read-only.
    """
    e = _get_escalation_or_404(db, escalation_id)
    msgs = thread_service.list_messages(db, e)
    return _thread_view(e, msgs)


@router.post("/escalations/{escalation_id}/thread/start", response_model=EscalationDetail)
async def start_thread(escalation_id: uuid.UUID, db: Session = Depends(get_db)):
    """Inicia a live thread. Notifica o aluno via WhatsApp que o
    coordenador entrou na conversa.

    Erros:
      - 409 se a escalação não estiver em estado iniciável
        (pending / coordinator_replied).
      - 409 se o aluno já tem outra escalação ``live``.
      - 502 se o envio do aviso ao aluno falhar (mas o estado é
        revertido).
    """
    e = _get_escalation_or_404(db, escalation_id)
    student = db.query(Student).filter(Student.id == e.student_id).first()
    session = db.query(QASession).filter(QASession.id == e.session_id).first()
    if student is None or session is None:
        raise HTTPException(status_code=500, detail="Dados inconsistentes.")

    try:
        thread_service.start_live(db, e)
    except ThreadConflictError as ex:
        raise HTTPException(status_code=409, detail=str(ex))
    except ThreadStateError as ex:
        raise HTTPException(status_code=409, detail=str(ex))

    # Notifica o aluno. Se falhar, reverte pra não deixar a thread
    # aberta sem o aluno saber.
    try:
        await evolution_client.send_text(student.phone_number, THREAD_OPENED_NOTICE)
    except Exception as ex:
        logger.exception(f"Erro avisando aluno da abertura da thread: {ex}")
        db.rollback()
        raise HTTPException(
            status_code=502,
            detail="Falha ao avisar o aluno via WhatsApp. Thread não iniciada.",
        )

    db.commit()
    db.refresh(e)
    return _detail(e, student, session)


@router.post(
    "/escalations/{escalation_id}/thread/messages",
    response_model=ThreadMessageOut,
)
async def post_thread_message(
    escalation_id: uuid.UUID,
    payload: ThreadSendPayload,
    db: Session = Depends(get_db),
):
    """Coordenador envia uma mensagem pro aluno via WhatsApp.

    Pré-condição: thread em status ``live``. Mensagens são prefixadas
    com COORDINATOR_PREFIX pra deixar claro pro aluno quem está
    falando.
    """
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    e = _get_escalation_or_404(db, escalation_id)
    if e.status != "live":
        raise HTTPException(
            status_code=409,
            detail=f"Thread não está ativa (status='{e.status}').",
        )

    student = db.query(Student).filter(Student.id == e.student_id).first()
    if student is None:
        raise HTTPException(status_code=500, detail="Aluno não encontrado.")

    # Envia primeiro pro WhatsApp — se falhar, NÃO grava no histórico
    # (não queremos UI dizendo "enviado" pra algo que o aluno não recebeu).
    prefixed = f"{COORDINATOR_PREFIX}{text}"
    try:
        msg_id = await evolution_client.send_text(student.phone_number, prefixed)
    except Exception as ex:
        logger.exception(f"Erro enviando msg da thread ao aluno: {ex}")
        raise HTTPException(
            status_code=502,
            detail="Falha ao enviar via WhatsApp. Tente novamente.",
        )
    if msg_id is None:
        raise HTTPException(
            status_code=502,
            detail="Falha ao enviar via WhatsApp. Tente novamente.",
        )

    msg = thread_service.append_message(
        db, e,
        direction="coordinator",
        text=text,
        evolution_msg_id=msg_id,
    )
    db.commit()
    db.refresh(msg)
    return _thread_msg_out(msg)


@router.post("/escalations/{escalation_id}/thread/close", response_model=EscalationDetail)
async def close_thread(escalation_id: uuid.UUID, db: Session = Depends(get_db)):
    """Coordenador encerra a thread. Avisa o aluno via WhatsApp e
    retorna o estado final.

    Erros:
      - 409 se a thread não está ``live``.
      - 502 se o aviso ao aluno falhar (estado já foi alterado,
        coordenador pode reabrir manualmente se necessário).
    """
    e = _get_escalation_or_404(db, escalation_id)
    student = db.query(Student).filter(Student.id == e.student_id).first()
    session = db.query(QASession).filter(QASession.id == e.session_id).first()
    if student is None or session is None:
        raise HTTPException(status_code=500, detail="Dados inconsistentes.")

    try:
        thread_service.close_live(db, e, reason="coordinator")
    except ThreadStateError as ex:
        raise HTTPException(status_code=409, detail=str(ex))

    db.commit()
    db.refresh(e)

    # Notifica o aluno. Falha aqui é loggada mas NÃO reverte — o
    # coordenador já decidiu encerrar, melhor honrar mesmo que o
    # aviso falhe (o status no banco já é a verdade).
    try:
        await evolution_client.send_text(
            student.phone_number, THREAD_CLOSED_BY_COORDINATOR_NOTICE
        )
    except Exception as ex:
        logger.warning(f"Thread fechada mas aviso falhou: {ex}")

    return _detail(e, student, session)


# ---------------------------------------------------------------------------
# Maintenance — manual triggers (cron externo chama esses endpoints)
# ---------------------------------------------------------------------------

class MaintenanceResult(BaseModel):
    closed: int


@router.post("/maintenance/close-stale", response_model=MaintenanceResult)
def trigger_close_stale(db: Session = Depends(get_db)):
    return MaintenanceResult(closed=maintenance.close_stale_sessions(db))


@router.post("/maintenance/end-of-day", response_model=MaintenanceResult)
def trigger_end_of_day(db: Session = Depends(get_db)):
    return MaintenanceResult(closed=maintenance.close_sessions_end_of_day(db))


@router.post("/maintenance/close-stale-threads", response_model=MaintenanceResult)
async def trigger_close_stale_threads(db: Session = Depends(get_db)):
    """Fecha live threads aluno↔coordenador ociosas há mais de 24h."""
    return MaintenanceResult(closed=await maintenance.close_stale_live_threads(db))
