"""Cria registros de ``Escalation`` para o coordenador.

Quando o bot esgota as 3 tentativas sem resolver a dúvida, o webhook
chama :func:`create_escalation`, que:

1. Resume as tentativas usando o LLM, destacando a pergunta
   original e o que o bot respondeu de errado — material cru para o
   coordenador agir e para a análise do TCC.
2. Cria a linha em ``escalations`` vinculada à ``QASession``.
3. Marca a sessão como ``escalated`` (se ainda não estiver).

Se o LLM falhar, geramos um *summary* de fallback concatenando as
tentativas, para nunca perder o registro.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from database import Escalation, QAAttempt, QASession, Student
from pipeline.llm import generate
from services import session_manager

logger = logging.getLogger(__name__)


_SUMMARY_PROMPT = (
    "Você é um assistente de triagem. Abaixo estão as tentativas que um "
    "chatbot fez de responder a dúvida de um aluno universitário. O bot "
    "falhou nas 3 tentativas. Escreva um resumo curto em português (3 a 6 "
    "linhas) destacando:\n"
    "- Qual é a dúvida real do aluno (reformule a pergunta);\n"
    "- Em que o bot se atrapalhou;\n"
    "- O que o coordenador precisa confirmar / esclarecer.\n\n"
    "Não invente fatos. Seja objetivo. Responda apenas o texto do resumo, "
    "sem cabeçalhos nem marcação.\n\n"
    "Tentativas:\n{attempts}\n"
)


def _render_attempts(attempts: list[QAAttempt]) -> str:
    lines: list[str] = []
    for a in attempts:
        lines.append(f"[Tentativa {a.attempt_number}]")
        lines.append(f"Pergunta do aluno: {a.question}")
        lines.append(f"Resposta do bot:   {a.answer}")
        if a.feedback_signal:
            lines.append(f"Feedback do aluno: {a.feedback_signal}")
        lines.append("")
    return "\n".join(lines).strip()


def summarize_attempts(attempts: list[QAAttempt]) -> str:
    """Gera o summary via LLM; em caso de erro, devolve fallback cru."""
    if not attempts:
        return "(sessão sem tentativas registradas)"
    rendered = _render_attempts(attempts)
    try:
        raw = generate(_SUMMARY_PROMPT.replace("{attempts}", rendered))
    except Exception as e:  # pragma: no cover - log path
        logger.warning(f"LLM falhou ao resumir escalação: {e}")
        return _fallback_summary(attempts, rendered)
    text = (raw or "").strip()
    if not text:
        return _fallback_summary(attempts, rendered)
    return text


def _fallback_summary(attempts: list[QAAttempt], rendered: str) -> str:
    first_q = attempts[0].question if attempts else "(?)"
    return (
        f"[Resumo automático indisponível — LLM falhou.]\n"
        f"Dúvida inicial do aluno: {first_q}\n\n"
        f"{rendered}"
    )


def create_escalation(
    db: Session, session: QASession, student: Student
) -> Escalation:
    """Cria (ou devolve a existente) a Escalation para esta sessão.

    Idempotente: se já existir escalação, não cria outra.
    Também garante que a sessão esteja marcada como ``escalated``.
    """
    existing = (
        db.query(Escalation).filter(Escalation.session_id == session.id).first()
    )
    if existing is not None:
        return existing

    # Ordena pela tentativa (attempt_number asc).
    attempts = sorted(session.attempts or [], key=lambda a: a.attempt_number)
    summary = summarize_attempts(attempts)

    escalation = Escalation(
        session_id=session.id,
        student_id=student.id,
        summary=summary,
        status="pending",
    )
    db.add(escalation)

    if session.status != "escalated":
        session_manager.close_as_escalated(db, session)

    db.flush()
    return escalation
