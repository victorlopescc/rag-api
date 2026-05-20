"""Gerenciamento de sessões de perguntas (Q&A) para avaliação do bot (TCC).

API pública é dividida em duas fases:

1. ``plan_interaction(db, student, message)`` — classifica a mensagem,
   decide se é ``ack_yes`` / ``answer`` / ``escalate`` e atualiza o estado
   da sessão (fecha antigas, abre novas). **Não** executa o RAG.
2. ``record_attempt(db, plan, question, answer, chunks_used, ...)``
   persiste a QAAttempt depois que o webhook rodou o RAG.

Também expõe utilitários de ciclo de vida:
- ``close_as_resolved(db, session, poll_id=None)``
- ``close_as_abandoned(db, session, poll_id=None)``
- ``close_as_escalated(db, session)``
- ``apply_poll_vote(db, poll_id, feedback)``

Histórico: o plano também escolhia uma "estratégia de retrieval"
(default/query_rewrite/widen_k) por tentativa. Removido — hoje todas
as tentativas usam o mesmo retrieval híbrido (``pipeline.retrieval``).
O que muda entre as tentativas é a pergunta do aluno (ele reformula);
a regra de escalar na 4ª tentativa permanece.
"""
from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from sqlalchemy.orm import Session

from database import QAAttempt, QASession, Student, utcnow

# ---------------------------------------------------------------------------
# Regex fast-path (somente "yes" — nem ``no`` nem ``new_topic`` passam aqui
# para evitar classificação errada; isso fica para o LLM).
# ---------------------------------------------------------------------------

_YES_TOKENS = {
    "sim", "s", "1",
    "ok", "okay", "blz", "beleza",
    "resolvido", "resolveu", "resolvi",
    "obrigado", "obrigada", "obg", "vlw", "valeu",
    "perfeito", "ajudou", "consegui", "funcionou",
    "show", "top", "massa",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(_strip_accents(text or "").lower())


def classify_fast(text: str) -> str:
    tokens = _tokenize(text)
    if not tokens or len(tokens) > 5:
        return "unclear"
    if all(tok in _YES_TOKENS for tok in tokens):
        return "yes"
    return "unclear"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

SESSION_IDLE_TIMEOUT = timedelta(hours=6)


def _as_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _is_stale(session: QASession) -> bool:
    last_activity = session.opened_at
    if session.attempts:
        last_activity = max(a.created_at for a in session.attempts)
    return (utcnow() - _as_aware(last_activity)) > SESSION_IDLE_TIMEOUT


# ---------------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------------

Action = Literal["ack_yes", "answer", "escalate", "thanks"]


@dataclass
class SessionPlan:
    action: Action
    # ``None`` quando ``action == "thanks"`` (não há sessão envolvida —
    # é só um ack educado fora do fluxo de sessões).
    session_id: uuid.UUID | None
    attempt_number: int          # 0 quando action ∈ {"ack_yes", "thanks"}
    prior_intent: str            # 'yes'|'no'|'rephrase'|'new_topic'|'unclear'|'none'|'thanks_no_session'|'sentinel_escalation'
    # Texto da pergunta anterior NA MESMA SESSÃO. Usado pelo RAG só para
    # retrieval (manter o tópico) — o prompt do LLM continua vendo só a
    # pergunta atual. ``None`` quando é a 1ª da sessão ou novo tópico.
    prior_question: str | None = None


# ---------------------------------------------------------------------------
# Helpers de ciclo de vida
# ---------------------------------------------------------------------------

def _get_open_session(db: Session, student: Student) -> QASession | None:
    return (
        db.query(QASession)
        .filter(QASession.student_id == student.id, QASession.status == "open")
        .order_by(QASession.opened_at.desc())
        .first()
    )


def _last_attempt(db: Session, session: QASession) -> QAAttempt | None:
    return (
        db.query(QAAttempt)
        .filter(QAAttempt.session_id == session.id)
        .order_by(QAAttempt.attempt_number.desc())
        .first()
    )


# Janela em que perguntas anteriores do mesmo aluno ainda contam como
# "tópico recente" mesmo se o intent classifier abrir uma nova sessão.
# Tradeoff: longa demais cola perguntas de testes antigos na conversa
# nova (já vi 30 min contaminar Q1); curta demais perde contexto entre
# perguntas factuais sequenciais sobre o mesmo doc. 5 min cobre uma
# conversa em ritmo de WhatsApp sem atravessar pausas.
_RECENT_TOPIC_WINDOW = timedelta(minutes=5)


def _recent_question(db: Session, student: Student) -> str | None:
    """Pergunta-âncora pro retrieval quando a sessão nova não tem prior.

    Retorna preferencialmente uma pergunta da janela recente que
    mencione uma sigla conhecida (ADA, PPC, TCC) — essas carregam o
    tópico do guarda-chuva. Sem isso, encadear "calculadora" → "horário"
    perde o anchor ADA. Cai pra última pergunta qualquer se nenhuma
    tiver sigla.
    """
    from pipeline.acronyms import detect_categories

    cutoff = utcnow() - _RECENT_TOPIC_WINDOW
    recent = (
        db.query(QAAttempt)
        .join(QASession, QAAttempt.session_id == QASession.id)
        .filter(
            QASession.student_id == student.id,
            QAAttempt.created_at >= cutoff,
        )
        .order_by(QAAttempt.created_at.desc())
        .limit(10)
        .all()
    )
    if not recent:
        return None
    # 1ª escolha: mais recente que mencione uma sigla (carrega tópico).
    for att in recent:
        if detect_categories(att.question):
            return att.question
    # Fallback: a última pergunta qualquer.
    return recent[0].question


def _choose_prior(
    candidate: str | None,
    current_message: str,
) -> str | None:
    """Decide se vale anexar prior_question ao retrieval.

    Quando a mensagem ATUAL já carrega o tópico (tem sigla detectável
    como ADA, TCC, PPC), o prior_question vira ruído — pior, arrasta
    o assunto velho pra recuperação. Cenário comum: aluno pergunta
    sobre ADA, depois muda pra TCC; sem essa checagem, o retrieval
    mistura ambos os tópicos e a resposta vira fallback.

    Regra:
    - Mensagem atual já tem sigla / categoria detectável → retorna None.
    - Mensagem atual é vaga (ex.: "quem orienta?", "qual o prazo?") e
      o candidate carrega contexto → mantém o candidate.
    """
    from pipeline.acronyms import detect_categories

    if candidate is None:
        return None
    # Se a NOVA mensagem já fala sobre uma categoria, ela vai sozinha.
    if detect_categories(current_message):
        return None
    # Se o prior é da MESMA categoria que a nova mensagem, faz sentido
    # manter — sticky topic real. Se forem categorias diferentes, descarta.
    cur_cats = set(detect_categories(current_message))
    prior_cats = set(detect_categories(candidate))
    if prior_cats and cur_cats and prior_cats != cur_cats:
        return None
    return candidate


def _attempt_count(db: Session, session: QASession) -> int:
    return db.query(QAAttempt).filter(QAAttempt.session_id == session.id).count()


def _mark_prev(db: Session, session: QASession, signal: str, resolved: bool | None) -> None:
    last = _last_attempt(db, session)
    if last is not None and last.feedback_signal is None:
        last.feedback_signal = signal
        last.resolved = resolved


def close_as_resolved(
    db: Session, session: QASession, *, poll_id: str | None = None
) -> None:
    _mark_prev(db, session, "explicit_yes", True)
    session.status = "resolved"
    session.closed_at = utcnow()
    if poll_id is not None:
        session.closing_poll_id = poll_id


def close_as_abandoned(
    db: Session, session: QASession, *, poll_id: str | None = None,
    signal: str = "timeout",
) -> None:
    _mark_prev(db, session, signal, None)
    session.status = "abandoned"
    session.closed_at = utcnow()
    if poll_id is not None:
        session.closing_poll_id = poll_id


def close_as_escalated(
    db: Session, session: QASession, *, poll_id: str | None = None,
) -> None:
    session.status = "escalated"
    session.closed_at = utcnow()
    if poll_id is not None:
        session.closing_poll_id = poll_id


def cancel_open_session(db: Session, student: Student) -> bool:
    """Encerra a sessão aberta atual do aluno (acionado por /cancelar).

    Marca como ``abandoned`` com sinal ``cancelled_by_user`` na última
    tentativa, pra distinguir de timeout natural na análise. Retorna
    True se uma sessão foi encerrada, False se não havia nenhuma aberta.
    """
    open_session = _get_open_session(db, student)
    if open_session is None:
        return False
    _mark_prev(db, open_session, "cancelled_by_user", None)
    open_session.status = "abandoned"
    open_session.closed_at = utcnow()
    db.flush()
    return True


# ---------------------------------------------------------------------------
# plan_interaction — decide o quê fazer com a próxima mensagem
# ---------------------------------------------------------------------------

def _classify(message: str, prior: QAAttempt | None) -> str:
    """Camada 1: regex. Camada 2: LLM (só se houver prior).

    Importa dentro para quebrar a circular dependency (intent_classifier
    importa deste módulo).
    """
    if classify_fast(message) == "yes":
        return "yes"
    if prior is None:
        return "unclear"
    from services.intent_classifier import Prior, classify_with_llm
    return classify_with_llm(Prior(question=prior.question, answer=prior.answer), message)


def plan_interaction(
    db: Session, student: Student, message: str
) -> SessionPlan:
    """Classifica a mensagem e atualiza o estado da sessão. **Não** chama RAG.

    Retorna um ``SessionPlan`` que diz ao webhook:
      - ``ack_yes``: o aluno confirmou — mande ack + poll e encerre.
      - ``answer``:  rode o RAG com ``plan.strategy`` e depois chame
        ``record_attempt``.
      - ``escalate``: 3 tentativas falharam — mande para o coordenador
        (milestone 3 implementa o que fazer aqui; milestone 2 apenas
        sinaliza).
    """
    open_session = _get_open_session(db, student)

    # Sessão vencida → fecha e começa do zero.
    if open_session is not None and _is_stale(open_session):
        close_as_abandoned(db, open_session)
        db.flush()
        open_session = None

    # ATALHO: aluno pediu coordenador explicitamente → escalação direta
    # antes de qualquer classificação. Funciona em qualquer momento da
    # conversa. Se não houver sessão aberta, abre uma nova só pra anexar
    # a escalação (o coordenador tem o contexto da pergunta inicial via
    # student.phone_number).
    if is_escalation_sentinel(message):
        if open_session is None:
            open_session = QASession(student_id=student.id, status="open")
            db.add(open_session)
            db.flush()
        attempt = _attempt_count(db, open_session)
        return SessionPlan(
            action="escalate",
            session_id=open_session.id,
            attempt_number=attempt,
            prior_intent="sentinel_escalation",
        )

    # Sem sessão aberta: "obrigado", "valeu", "blz" isolados são apenas
    # gentileza — não tem pergunta pra responder. Mandar pro RAG faz o
    # bot dar fallback ("não encontrei essa informação") o que parece
    # rude diante de um "obrigado". Devolvemos um ack educado.
    if open_session is None and classify_fast(message) == "yes":
        return SessionPlan(
            action="thanks",
            session_id=None,
            attempt_number=0,
            prior_intent="thanks_no_session",
        )

    # Sem sessão aberta: qualquer outra mensagem abre uma nova.
    if open_session is None:
        new = QASession(student_id=student.id, status="open")
        db.add(new)
        db.flush()
        return SessionPlan(
            action="answer",
            session_id=new.id,
            attempt_number=1,
            prior_intent="none",
            # Lookback de 5 min, mas só anexa se a mensagem atual NÃO
            # já carregar o tópico (sigla detectada). Evita arrastar
            # ADA pra pergunta sobre TCC.
            prior_question=_choose_prior(_recent_question(db, student), message),
        )

    prior = _last_attempt(db, open_session)
    intent = _classify(message, prior)

    # --- yes -> ack + close --------------------------------------------------
    if intent == "yes":
        close_as_resolved(db, open_session)
        db.flush()
        return SessionPlan(
            action="ack_yes",
            session_id=open_session.id,
            attempt_number=0,
            prior_intent="yes",
        )

    # --- new_topic -> fecha antiga como abandoned, abre nova -----------------
    if intent == "new_topic":
        _mark_prev(db, open_session, "implicit_new_topic", None)
        open_session.status = "abandoned"
        open_session.closed_at = utcnow()
        new = QASession(student_id=student.id, status="open")
        db.add(new)
        db.flush()
        # Mesmo "new_topic" pode ser falso positivo. Usamos o lookback,
        # mas filtramos via _choose_prior: se a mensagem atual já tem
        # sigla diferente, descarta o prior (não arrasta ADA pra TCC).
        return SessionPlan(
            action="answer",
            session_id=new.id,
            attempt_number=1,
            prior_intent="new_topic",
            prior_question=_choose_prior(_recent_question(db, student), message),
        )

    # --- no / rephrase / unclear -> continua a mesma sessão ------------------
    signal = {
        "no": "explicit_no",
        "rephrase": "implicit_rephrase",
        "unclear": "implicit_rephrase",
    }.get(intent, "implicit_rephrase")
    _mark_prev(db, open_session, signal, False)

    next_attempt = _attempt_count(db, open_session) + 1

    # Estourou o orçamento de 3 tentativas → escalação.
    if next_attempt > 3:
        return SessionPlan(
            action="escalate",
            session_id=open_session.id,
            attempt_number=_attempt_count(db, open_session),
            prior_intent=intent,
        )

    return SessionPlan(
        action="answer",
        session_id=open_session.id,
        attempt_number=next_attempt,
        prior_intent=intent,
        # Mesma sessão: prior é a pergunta anterior, mas só se a nova
        # ainda for do mesmo tópico (ou se a nova não tem sigla, caso
        # em que mantemos o anchor).
        prior_question=_choose_prior(
            prior.question if prior else None, message,
        ),
    )


# ---------------------------------------------------------------------------
# record_attempt — persiste a tentativa depois que o RAG rodou
# ---------------------------------------------------------------------------

def record_attempt(
    db: Session,
    plan: SessionPlan,
    question: str,
    answer: str,
    *,
    chunks_used: list[dict] | list[str] | None = None,
    was_fallback: bool = False,
    latency_ms: int | None = None,
) -> QAAttempt:
    """Grava a QAAttempt e devolve o objeto.

    Exige ``plan.action == 'answer'`` — para ``ack_yes`` não há o que gravar,
    e para ``escalate`` a responsabilidade é do caller (milestone 3).
    """
    if plan.action != "answer":
        raise ValueError(f"record_attempt não faz sentido para action={plan.action}")
    attempt = QAAttempt(
        session_id=plan.session_id,
        attempt_number=plan.attempt_number,
        question=question,
        answer=answer,
        retrieved_chunks=chunks_used or [],
        was_fallback=was_fallback,
        latency_ms=latency_ms,
    )
    db.add(attempt)
    db.flush()
    return attempt


# ---------------------------------------------------------------------------
# Enquete (poll) — aplicar o voto do aluno sobre uma sessão fechada
# ---------------------------------------------------------------------------

# Textos das opções da enquete, na ordem em que são enviados. O webhook
# mapeia o índice selecionado (ou o texto) para um desses rótulos.
#
# A 3ª opção ("Não resolveu, falar com coordenador") tem comportamento
# especial: se a sessão ainda estiver ABERTA quando o voto chegar,
# o webhook escala (cria Escalation). Se já estiver fechada (ex.: poll
# enviada após "obrigado"), só registra o feedback. Mesmo conjunto de
# opções é usado em 3 momentos:
#   1. Após "obrigado"/yes (sessão já fechada como resolved).
#   2. Após a 3ª tentativa — sessão ainda aberta, decisão final.
#   3. Após escalação criada (sessão já fechada como escalated).
POLL_QUESTION = "O bot conseguiu te ajudar?"
POLL_OPTIONS = [
    ("Sim, tudo resolvido", "resolved_fully"),
    ("Resolvi parcialmente", "resolved_partially"),
    ("Não, falar com coordenador", "not_resolved"),
]


def feedback_from_option(selected: str) -> str | None:
    """Dado o texto da opção escolhida pelo aluno, devolve o rótulo canônico."""
    selected_norm = (selected or "").strip().lower()
    for label, feedback in POLL_OPTIONS:
        if label.lower() == selected_norm:
            return feedback
    return None


def apply_poll_vote(
    db: Session, poll_id: str, feedback: str
) -> QASession | None:
    """Registra o voto na sessão correspondente. Retorna a sessão ou None.

    Não fecha a sessão aqui — o webhook decide o que fazer com base no
    status atual da sessão (open vs já fechada) e no feedback escolhido.
    """
    session = (
        db.query(QASession).filter(QASession.closing_poll_id == poll_id).first()
    )
    if session is None:
        return None
    session.closing_feedback = feedback
    return session


# ---------------------------------------------------------------------------
# Sentinel para escalação direta — atalho do aluno
# ---------------------------------------------------------------------------

# Frases que disparam escalação imediata (match exato, sem ambiguidade).
# Mantenha o conjunto ENXUTO — fuzzy matching aqui vira pegadinha:
# "Quem é o coordenador?" não pode escalar sem querer.
_ESCALATION_SENTINELS: frozenset[str] = frozenset({
    "coordenador",
    "/coordenador",
    "/coord",
    "humano",
    "atendimento humano",
    "falar com coordenador",
    "falar com o coordenador",
    "quero coordenador",
    "quero falar com coordenador",
    "quero falar com o coordenador",
})


def is_escalation_sentinel(text: str) -> bool:
    """True se a mensagem é um pedido EXATO pra falar com o coordenador.

    Match estrito: depois de lowercase + tirar pontuação das pontas, o
    texto inteiro precisa ser uma das frases-sentinela. Isso evita falso
    positivo em perguntas tipo "Quem é o coordenador da ADA?".
    """
    if not text:
        return False
    norm = text.strip().lower()
    # Tira pontuação simples nas pontas
    while norm and norm[0] in "?!.,;:":
        norm = norm[1:]
    while norm and norm[-1] in "?!.,;:":
        norm = norm[:-1]
    norm = norm.strip()
    return norm in _ESCALATION_SENTINELS


# Respostas em texto da poll de feedback que aparece em cada resposta do
# RAG. O webhook intercepta esses códigos antes da triagem normal:
#
#   "1" → resolveu — fecha sessão como ``resolved``
#   "2" → quer reformular — NÃO fecha sessão; bot manda ack pedindo a
#         nova pergunta; quando ela chega, ``plan_interaction`` trata
#         como rephrase (incrementa attempt — na 3ª, escala automático)
#   "3" → não resolveu — escala pro coordenador
#
# A semântica do "2" mudou: antes era "resolveu parcialmente" (fechava
# como ``resolved_partially``). Hoje significa "vou tentar de novo",
# alinhado ao fluxo de feedback contínuo desenhado pela coordenação.
_TEXT_VOTE_MAP: dict[str, str] = {
    "1": "resolved_fully",
    "2": "wants_rephrase",
    "3": "not_resolved",
}


def text_vote_for(message: str) -> str | None:
    """Mapeia texto-resposta da poll fallback ('1'/'2'/'3') ao feedback.
    Retorna None se a mensagem não é uma resposta numérica válida."""
    norm = (message or "").strip()
    return _TEXT_VOTE_MAP.get(norm)
