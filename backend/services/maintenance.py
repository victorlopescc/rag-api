"""Tarefas de manutenção invocadas por cron externo (systemd/Task Scheduler)
ou manualmente via endpoint admin.

Rotinas:

- :func:`close_stale_sessions` — fecha sessões ``open`` cuja última
  atividade passou de :data:`session_manager.SESSION_IDLE_TIMEOUT` (6h).
- :func:`close_sessions_end_of_day` — fecha *todas* as sessões abertas
  daquele dia, sem checar idle. Ideal para rodar à 00:00.
- :func:`close_stale_live_threads` — fecha live threads aluno↔coordenador
  cuja última mensagem do coordenador foi há mais de 24h, avisando o
  aluno via WhatsApp que a conversa foi encerrada por inatividade.

Todas devolvem o número de itens fechados, para o chamador logar.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone

from sqlalchemy.orm import Session

from database import QASession, utcnow
from services import session_manager

logger = logging.getLogger(__name__)


def _open_sessions(db: Session) -> list[QASession]:
    return db.query(QASession).filter(QASession.status == "open").all()


def close_stale_sessions(db: Session) -> int:
    """Fecha como ``abandoned`` toda sessão aberta ociosa há mais que o
    timeout configurado. Retorna o total fechado.
    """
    closed = 0
    for sess in _open_sessions(db):
        if session_manager._is_stale(sess):
            session_manager.close_as_abandoned(db, sess, signal="timeout")
            closed += 1
    if closed:
        db.commit()
        logger.info(f"close_stale_sessions: {closed} sessão(ões) fechada(s).")
    return closed


async def close_stale_live_threads(db: Session) -> int:
    """Fecha live threads ociosas (sem atividade do coordenador por mais
    de 24h) e avisa o aluno via WhatsApp.

    Async porque o envio do WhatsApp é async — não bloqueia o cron.
    Falha no envio é loggada mas não impede o fechamento (estado no
    banco é a verdade; aluno descobre quando mandar outra mensagem).
    """
    from services import thread_service
    from services.evolution_client import evolution_client
    from services.whatsapp import THREAD_CLOSED_TIMEOUT_NOTICE
    from database import Student

    stale = thread_service.find_stale_live_threads(db)
    closed = 0
    for esc in stale:
        try:
            thread_service.close_live(db, esc, reason="timeout")
            closed += 1
        except Exception as e:
            logger.warning(f"Falha fechando thread {esc.id} por timeout: {e}")
            continue

    if closed:
        db.commit()
        logger.info(f"close_stale_live_threads: {closed} thread(s) fechada(s) por timeout.")

    # Notifica os alunos. Fora da transação principal pra não bloquear
    # o commit em caso de falha do WhatsApp.
    for esc in stale[:closed]:
        student = db.query(Student).filter(Student.id == esc.student_id).first()
        if student is None:
            continue
        try:
            await evolution_client.send_text(
                student.phone_number, THREAD_CLOSED_TIMEOUT_NOTICE
            )
        except Exception as e:
            logger.warning(f"Aviso de timeout falhou pra {student.phone_number}: {e}")

    return closed


def close_sessions_end_of_day(db: Session, *, now: datetime | None = None) -> int:
    """Fecha todas as sessões abertas cujo ``opened_at`` é anterior a hoje
    (UTC). Usado pelo cron de fim-de-dia.

    Recebe ``now`` para testabilidade.
    """
    now = now or utcnow()
    cutoff = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)

    closed = 0
    for sess in _open_sessions(db):
        opened = session_manager._as_aware(sess.opened_at)
        if opened < cutoff:
            session_manager.close_as_abandoned(db, sess, signal="timeout")
            closed += 1
    if closed:
        db.commit()
        logger.info(f"close_sessions_end_of_day: {closed} sessão(ões) fechada(s).")
    return closed
