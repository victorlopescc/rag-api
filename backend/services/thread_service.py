"""Gerencia o ciclo de vida da live thread entre aluno e coordenador.

Contrato
--------
Uma ``Escalation`` pode estar em vários estados; a vista "live thread"
opera sobre os 3 estados conversacionais:

  ``pending``   → o coordenador ainda não interveio.
  ``live``      → conversa ao vivo aberta. Mensagens do aluno via
                  WhatsApp pulam o bot; respostas do coordenador
                  vão direto pra ele.
  ``resolved``  → thread encerrada pelo coordenador. Bot retoma.
  ``abandoned`` → thread encerrada por timeout (24h sem atividade do
                  coordenador) ou pelo aluno (``/encerrar``).

Transições válidas:
  pending  → live      (start_live)
  live     → resolved  (close_live com motivo='coordinator')
  live     → abandoned (close_live com motivo='student' ou 'timeout')

Tudo o mais é erro 409.

Invariante operacional
----------------------
**Um aluno só pode ter UMA escalação live por vez.** ``find_live_for_student``
consulta o índice parcial ``idx_escalations_live_by_student``. Se já
tem uma, ``start_live`` numa segunda escalação devolve 409.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Literal

from sqlalchemy.orm import Session

from database import Escalation, Student, ThreadMessage, utcnow

logger = logging.getLogger(__name__)

# Timeout de inatividade: depois de 24h sem mensagem do coordenador,
# a thread auto-fecha pra não deixar o aluno preso fora do fluxo do bot.
# (Mensagens do aluno NÃO resetam o timer — o aluno fica respondendo
# sozinho não deve manter a thread aberta indefinidamente.)
LIVE_TIMEOUT = timedelta(hours=24)


CloseReason = Literal["coordinator", "student", "timeout"]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def find_live_for_student(db: Session, student: Student) -> Escalation | None:
    """Devolve a escalação ``live`` desse aluno, se houver. None caso contrário."""
    return (
        db.query(Escalation)
        .filter(
            Escalation.student_id == student.id,
            Escalation.status == "live",
        )
        .first()
    )


def find_stale_live_threads(db: Session, *, now=None) -> list[Escalation]:
    """Devolve escalações ``live`` cuja última atividade do coordenador
    foi há mais de ``LIVE_TIMEOUT``. Usada pelo cron de cleanup."""
    cutoff = (now or utcnow()) - LIVE_TIMEOUT
    return (
        db.query(Escalation)
        .filter(
            Escalation.status == "live",
            # last_activity_at NULL é tratado como "muito tempo" (defensivo,
            # mas normalmente preenchido no start_live).
            (Escalation.last_activity_at.is_(None))
            | (Escalation.last_activity_at < cutoff),
        )
        .all()
    )


# ---------------------------------------------------------------------------
# Transições
# ---------------------------------------------------------------------------

class ThreadStateError(Exception):
    """Tentativa de transição inválida (ex.: fechar uma thread que não está live)."""
    pass


class ThreadConflictError(Exception):
    """Conflito: aluno já tem outra escalação live aberta."""
    pass


def start_live(db: Session, escalation: Escalation) -> Escalation:
    """Abre a live thread. Pré-condição: status atual está em
    {pending, coordinator_replied} (qualquer estado terminal vira 409).

    Falha com :class:`ThreadConflictError` se o aluno já tem outra
    escalação live aberta.
    """
    if escalation.status not in ("pending", "coordinator_replied"):
        raise ThreadStateError(
            f"Não é possível iniciar conversa numa escalação com status "
            f"'{escalation.status}'."
        )

    # Garante UMA thread live por aluno.
    other = (
        db.query(Escalation)
        .filter(
            Escalation.student_id == escalation.student_id,
            Escalation.status == "live",
            Escalation.id != escalation.id,
        )
        .first()
    )
    if other is not None:
        raise ThreadConflictError(
            "Este aluno já tem uma conversa ao vivo em outra escalação. "
            "Encerre a outra antes de iniciar uma nova."
        )

    now = utcnow()
    escalation.status = "live"
    escalation.live_opened_at = now
    escalation.last_activity_at = now
    db.flush()
    return escalation


def close_live(
    db: Session, escalation: Escalation, *, reason: CloseReason
) -> Escalation:
    """Encerra a thread. Pré-condição: status atual é 'live'.

    O ``reason`` define o status final:
      - ``coordinator`` → 'resolved' (caminho feliz)
      - ``student``     → 'abandoned' (aluno usou /encerrar)
      - ``timeout``     → 'abandoned' (cron fechou por inatividade)
    """
    if escalation.status != "live":
        raise ThreadStateError(
            f"Só é possível encerrar threads em status 'live' "
            f"(atual: '{escalation.status}')."
        )
    escalation.status = "resolved" if reason == "coordinator" else "abandoned"
    escalation.live_closed_at = utcnow()
    db.flush()
    return escalation


# ---------------------------------------------------------------------------
# Append de mensagens
# ---------------------------------------------------------------------------

def append_message(
    db: Session,
    escalation: Escalation,
    *,
    direction: Literal["student", "coordinator"],
    text: str,
    evolution_msg_id: str | None = None,
) -> ThreadMessage:
    """Anexa uma mensagem na thread. Atualiza ``last_activity_at``
    APENAS quando ``direction == 'coordinator'`` — o timeout é baseado
    em inatividade do coordenador (atendimento), não do aluno (que
    pode ficar mandando mensagem esperando resposta).

    Não valida o status — o caller decide se pode escrever ou não
    (ex.: webhook só anexa se status='live', mas o teste pode anexar
    diretamente).
    """
    msg = ThreadMessage(
        escalation_id=escalation.id,
        direction=direction,
        text=text,
        evolution_msg_id=evolution_msg_id,
    )
    db.add(msg)
    if direction == "coordinator":
        escalation.last_activity_at = utcnow()
    db.flush()
    return msg


def list_messages(
    db: Session, escalation: Escalation
) -> list[ThreadMessage]:
    """Mensagens da thread em ordem cronológica."""
    return (
        db.query(ThreadMessage)
        .filter(ThreadMessage.escalation_id == escalation.id)
        .order_by(ThreadMessage.sent_at.asc())
        .all()
    )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def close_stale_live_threads(db: Session) -> list[Escalation]:
    """Fecha threads ``live`` ociosas por mais de ``LIVE_TIMEOUT``.

    Devolve a lista de escalações que foram fechadas — o caller usa
    pra notificar o aluno via WhatsApp ("Conversa encerrada por
    inatividade. Mande nova pergunta quando quiser.").
    """
    stale = find_stale_live_threads(db)
    closed: list[Escalation] = []
    for esc in stale:
        try:
            close_live(db, esc, reason="timeout")
            closed.append(esc)
        except ThreadStateError:
            # Race com outro caminho de fechamento — ok, ignora.
            continue
    if closed:
        db.commit()
        logger.info(f"close_stale_live_threads: fechou {len(closed)} thread(s) por timeout.")
    return closed
