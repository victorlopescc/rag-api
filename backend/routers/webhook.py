import logging

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from config import settings
from database import QueryLog, QASession, Student, get_db
from rag_engine import ask
from services.dedup import message_dedup
from services.evolution_client import evolution_client
from services.lid_resolver import resolve_student_by_lid
from services import escalation_service, message_triage, session_manager, thread_service
from services.whatsapp import (
    CANCEL_REPLY_NOTHING,
    CANCEL_REPLY_OK,
    FALLBACK_HINT_SUFFIX,
    FEEDBACK_PROMPT_SUFFIX,
    GREETING_REPLY,
    HELP_MESSAGE,
    REPHRASE_ACK,
    THANKS_REPLY,
    THREAD_CLOSED_BY_STUDENT_NOTICE,
    TRIVIAL_REPLY,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook", tags=["webhook"])

UPSERT_EVENTS = {"messages.upsert", "MESSAGES_UPSERT"}
# Evolution emite os votos de enquete como messages.update com pollUpdates.
# Algumas versões também reemitem como messages.upsert com pollUpdateMessage —
# tratamos os dois caminhos.
POLL_UPDATE_EVENTS = {"messages.update", "MESSAGES_UPDATE"}

# Resposta curta quando o aluno confirma que a dúvida foi resolvida.
# Sem poll de feedback (a Evolution v1.8.7 não entrega votos de poll —
# ver issues do Baileys); o aluno já disse "obrigado", então fecha cordial
# e oferece o atalho /coordenador como rede de segurança.
YES_ACK = (
    "Que bom que ajudei! 🎓 Sua dúvida fica registrada como resolvida.\n\n"
    "Se mudar de ideia ou tiver outra pergunta, é só me chamar. "
    "Pra falar direto com o coordenador a qualquer momento, envie "
    "*/coordenador*."
)

# Mensagem ao aluno quando a 4ª mensagem dispara escalação automática.
ESCALATION_NOTICE = (
    "Vou encaminhar sua dúvida ao coordenador. Ele responde por aqui "
    "assim que possível."
)

# Janela em que uma resposta "1"/"2"/"3" do aluno conta como voto na
# última sessão dele. 30 min é folgado pra ele ver a mensagem com as
# opções e responder.
from datetime import timedelta, timezone, datetime as _dt  # noqa: E402

_TEXT_VOTE_WINDOW = timedelta(minutes=30)


# ---------------------------------------------------------------------------
# Extração de payload
# ---------------------------------------------------------------------------

def _extract_inbound(body: dict) -> dict | None:
    """Extrai os campos úteis de um messages.upsert de texto, ou None."""
    data = body.get("data", {}) or {}
    key = data.get("key", {}) or {}
    text = (data.get("message", {}) or {}).get("conversation", "")
    remote_jid = key.get("remoteJid", "")

    if key.get("fromMe") or not text.strip() or not remote_jid:
        return None

    return {
        "message_id": key.get("id", ""),
        "remote_jid": remote_jid,
        "push_name": data.get("pushName", ""),
        "text": text,
    }


def _extract_poll_vote(body: dict) -> dict | None:
    """Extrai um voto de enquete. Lida com os dois formatos do Evolution:

    a) ``messages.update`` → ``data.update.pollUpdates[0].vote.selectedOptions``
    b) ``messages.upsert`` com ``data.message.pollUpdateMessage`` —
       estrutura ``.vote.selectedOptions`` já decriptada.

    Retorna ``{poll_id, selected: [str,...]}`` ou None se o evento não
    for um voto reconhecível.
    """
    data = body.get("data", {}) or {}

    # Formato (a): messages.update
    updates = data.get("update", {}) or {}
    poll_updates = updates.get("pollUpdates") or []
    if poll_updates:
        pu = poll_updates[0] or {}
        key = pu.get("pollCreationMessageKey") or data.get("key") or {}
        poll_id = key.get("id", "")
        selected = ((pu.get("vote") or {}).get("selectedOptions")) or []
        return {"poll_id": poll_id, "selected": [_opt_name(o) for o in selected]}

    # Formato (b): messages.upsert com pollUpdateMessage
    msg = data.get("message", {}) or {}
    poll_update = msg.get("pollUpdateMessage")
    if poll_update:
        key = (poll_update.get("pollCreationMessageKey")) or {}
        poll_id = key.get("id", "")
        vote = poll_update.get("vote") or {}
        selected = vote.get("selectedOptions") or []
        return {"poll_id": poll_id, "selected": [_opt_name(o) for o in selected]}

    return None


def _opt_name(option: dict | str) -> str:
    if isinstance(option, str):
        return option
    if isinstance(option, dict):
        return option.get("optionName") or option.get("name") or ""
    return ""


# ---------------------------------------------------------------------------
# Lookup do aluno
# ---------------------------------------------------------------------------

def _find_student(remote_jid: str, db: Session) -> Student | None:
    if "@lid" in remote_jid:
        return resolve_student_by_lid(remote_jid, db)
    phone = remote_jid.split("@", 1)[0]
    return db.query(Student).filter(Student.phone_number == phone).first()


def _log_query(db: Session, student: Student, question: str, result) -> None:
    # QueryLog.chunks_used é ARRAY(Text); result.chunks_used virou list[dict]
    # depois do fix de descritor enriquecido. Extraímos só os IDs (também
    # tolera shape antigo list[str]).
    chunk_ids = [
        c["id"] if isinstance(c, dict) else c
        for c in (result.chunks_used or [])
    ]
    db.add(QueryLog(
        phone_number=student.phone_number,
        question=question,
        answer=result.answer,
        chunks_used=chunk_ids,
        model_used=settings.llm_model,
        latency_ms=result.latency_ms,
        was_fallback=result.was_fallback,
    ))


# ---------------------------------------------------------------------------
# Enquete de fechamento
# ---------------------------------------------------------------------------

async def _apply_text_vote(
    db: Session,
    session: "QASession",
    feedback: str,
    student: "Student",
) -> dict:
    """Aplica o voto recebido por texto (1/2/3) na sessão.

    Lógica:
    - feedback ``wants_rephrase`` → NÃO fecha. Avisa o aluno pra mandar
      a pergunta reformulada. A próxima mensagem dele entra no fluxo
      normal e será classificada como rephrase.
    - feedback ``not_resolved`` → SEMPRE escala (idempotente).
    - feedback ``resolved_fully`` → grava closing_feedback. Se a sessão
      estava aberta, fecha como resolved.
    """
    # "2" (wants_rephrase) NÃO marca closing_feedback nem fecha sessão.
    # Trata como sinal intermediário e deixa o fluxo seguir.
    if feedback == "wants_rephrase":
        try:
            await evolution_client.send_text(student.phone_number, REPHRASE_ACK)
        except Exception as e:
            logger.error(f"Erro enviando ack de rephrase: {e}")
        return {"status": "text_vote_rephrase"}

    session.closing_feedback = feedback

    if feedback == "not_resolved":
        from database import Escalation
        already = (
            db.query(Escalation)
            .filter(Escalation.session_id == session.id)
            .first()
            is not None
        )
        try:
            escalation_service.create_escalation(db, session, student)
            db.commit()
        except Exception as e:
            logger.exception(f"Erro criando escalação por voto texto: {e}")
            db.rollback()
            return {"status": "text_vote_escalation_failed"}

        if not already:
            try:
                await evolution_client.send_text(
                    student.phone_number, ESCALATION_NOTICE,
                )
            except Exception as e:
                logger.error(f"Erro avisando escalação por voto: {e}")
            return {"status": "text_vote_escalated"}
        return {"status": "text_vote_already_escalated"}

    # resolved_fully (único caso restante chegando aqui).
    if session.status == "open":
        session_manager.close_as_resolved(db, session)
    db.commit()
    try:
        await evolution_client.send_text(
            student.phone_number,
            "Que bom! 🎓 Fechando como resolvido. Quando precisar de algo, é só mandar.",
        )
    except Exception as e:
        logger.error(f"Erro confirmando voto: {e}")
    return {"status": "text_vote_recorded", "feedback": feedback}


async def _send_closing_poll(db: Session, session_id, phone: str) -> None:
    """Envia a enquete de feedback e salva o id em qa_sessions.closing_poll_id."""
    try:
        poll_id = await evolution_client.send_poll(
            number=phone,
            name=session_manager.POLL_QUESTION,
            options=[label for label, _ in session_manager.POLL_OPTIONS],
            selectable_count=1,
        )
    except Exception as e:
        logger.error(f"Falha enviando poll: {e}")
        return
    if poll_id is None:
        return
    session = db.query(QASession).filter(QASession.id == session_id).first()
    if session is not None:
        session.closing_poll_id = poll_id
        db.commit()


# ---------------------------------------------------------------------------
# Handler principal de mensagens de texto
# ---------------------------------------------------------------------------

async def handle_messages_upsert(body: dict, db: Session) -> dict:
    # 1. Um upsert pode trazer um voto de enquete — trata primeiro.
    vote = _extract_poll_vote(body)
    if vote is not None:
        return await _handle_poll_vote(vote, db)

    msg = _extract_inbound(body)
    if msg is None:
        return {"status": "ignored"}

    if message_dedup.seen(msg["message_id"]):
        logger.info(f"Mensagem {msg['message_id']} já processada — ignorando duplicata")
        return {"status": "duplicate"}

    logger.info(f"Mensagem de {msg['push_name']} ({msg['remote_jid']}): {msg['text'][:60]}")

    student = _find_student(msg["remote_jid"], db)
    if not student:
        logger.warning(
            f"Mensagem de {msg['remote_jid']} sem aluno vinculado — "
            f"usuário precisa se cadastrar primeiro."
        )
        return {"status": "unknown_sender"}

    # 1.5. Voto por TEXTO. O aluno respondeu "1"/"2"/"3" depois de receber
    # uma resposta do bot com o prompt de feedback anexado. Pega a sessão
    # ABERTA do aluno e aplica como voto. Esse é o caminho principal — a
    # poll nativa do WhatsApp não funciona em Evolution v1.8.7.
    #
    # IMPORTANTE: só considera sessão ABERTA. Se a sessão recente já foi
    # fechada (resolved/escalated/abandoned), o "1"/"2"/"3" perdeu o
    # contexto — não faz sentido reaplicar voto numa sessão encerrada
    # (causaria "fechou como resolvido" sem que o aluno esperasse). Cai
    # pra triagem normal, que provavelmente trata como trivial.
    text_vote = session_manager.text_vote_for(msg["text"])
    if text_vote is not None:
        cutoff = _dt.now(timezone.utc) - _TEXT_VOTE_WINDOW
        open_session = (
            db.query(QASession)
            .filter(
                QASession.student_id == student.id,
                QASession.status == "open",
                QASession.opened_at >= cutoff,
            )
            .order_by(QASession.opened_at.desc())
            .first()
        )
        if open_session is not None:
            logger.info(
                f"Voto por texto '{msg['text']}' → feedback={text_vote} "
                f"(session {open_session.id} status=open)"
            )
            return await _apply_text_vote(db, open_session, text_vote, student)
        # Sem sessão aberta → trata como mensagem normal (cai pra triagem).
        logger.info(
            f"Voto por texto '{msg['text']}' ignorado — sem sessão aberta."
        )

    # 1.55. LIVE THREAD: se o aluno está numa conversa ao vivo com o
    # coordenador, PULAMOS toda a triagem do bot — qualquer mensagem
    # vira ThreadMessage e aparece no painel. Exceção: o comando
    # /encerrar permite o aluno sair da thread por iniciativa própria.
    live_thread = thread_service.find_live_for_student(db, student)
    if live_thread is not None:
        if message_triage.is_end_thread_command(msg["text"]):
            try:
                thread_service.close_live(db, live_thread, reason="student")
                db.commit()
            except Exception as e:
                logger.exception(f"Erro fechando live thread por /encerrar: {e}")
                db.rollback()
                return {"status": "error"}
            try:
                await evolution_client.send_text(
                    student.phone_number, THREAD_CLOSED_BY_STUDENT_NOTICE
                )
            except Exception as e:
                logger.error(f"Erro avisando aluno do /encerrar: {e}")
            return {"status": "thread_closed_by_student"}

        # Qualquer outra mensagem: registra na thread e segue. Coordenador
        # vê via polling no painel. Sem auto-resposta — quem fala agora é
        # ele, não o bot.
        try:
            thread_service.append_message(
                db, live_thread,
                direction="student",
                text=msg["text"],
            )
            db.commit()
        except Exception as e:
            logger.exception(f"Erro anexando msg do aluno na thread: {e}")
            db.rollback()
            return {"status": "error"}
        return {"status": "thread_relayed"}

    # 1.6. Triagem rápida (sem RAG, sem LLM). Saudações, trivialidades e
    # comandos /ajuda /cancelar têm respostas pré-formatadas. Evita
    # gastar uma chamada de LLM em mensagens tipo "oi". Confirmações ("obrigado")
    # são deixadas passar pra session_manager.plan_interaction tratar.
    kind = message_triage.classify(msg["text"])
    if kind == "help":
        try:
            await evolution_client.send_text(student.phone_number, HELP_MESSAGE)
        except Exception as e:
            logger.error(f"Erro enviando /ajuda: {e}")
        return {"status": "help_sent"}
    if kind == "cancel":
        cancelled = session_manager.cancel_open_session(db, student)
        db.commit()
        reply = CANCEL_REPLY_OK if cancelled else CANCEL_REPLY_NOTHING
        try:
            await evolution_client.send_text(student.phone_number, reply)
        except Exception as e:
            logger.error(f"Erro enviando /cancelar: {e}")
        return {"status": "cancelled" if cancelled else "no_session_to_cancel"}
    if kind == "greeting":
        try:
            await evolution_client.send_text(student.phone_number, GREETING_REPLY)
        except Exception as e:
            logger.error(f"Erro enviando saudação: {e}")
        return {"status": "greeting"}
    if kind == "trivial":
        try:
            await evolution_client.send_text(student.phone_number, TRIVIAL_REPLY)
        except Exception as e:
            logger.error(f"Erro enviando trivial: {e}")
        return {"status": "trivial"}
    # kind == "question" → segue o pipeline normal abaixo.

    # 2. Planeja a interação (classifica + atualiza estado da sessão).
    try:
        plan = session_manager.plan_interaction(db, student, msg["text"])
        db.commit()
    except Exception as e:
        logger.exception(f"Erro planejando interação: {e}")
        db.rollback()
        return {"status": "error"}

    # 3. ack_yes: aluno confirmou → manda ack cordial. Sem poll
    # (Evolution não entrega cliques) — o ack já fecha o ciclo e o aluno
    # pode digitar de novo se mudar de ideia, ou usar /coordenador.
    if plan.action == "ack_yes":
        try:
            await evolution_client.send_text(student.phone_number, YES_ACK)
        except Exception as e:
            logger.error(f"Erro ao enviar ack: {e}")
        return {"status": "resolved"}

    # 3b. thanks: "obrigado"/"valeu"/"blz" SEM sessão aberta — só ack
    # educado, sem chamar RAG (que daria fallback rude).
    if plan.action == "thanks":
        try:
            await evolution_client.send_text(student.phone_number, THANKS_REPLY)
        except Exception as e:
            logger.error(f"Erro ao enviar thanks: {e}")
        return {"status": "thanks"}

    # 4. escalate: limite estourado ou sentinel /coordenador → cria
    # escalação + avisa o aluno. Sem poll.
    if plan.action == "escalate":
        sess = db.query(QASession).filter(QASession.id == plan.session_id).first()
        if sess is not None:
            try:
                escalation_service.create_escalation(db, sess, student)
                db.commit()
            except Exception as e:
                logger.exception(f"Erro criando escalação: {e}")
                db.rollback()
        try:
            await evolution_client.send_text(student.phone_number, ESCALATION_NOTICE)
        except Exception as e:
            logger.error(f"Erro avisando escalação: {e}")
        return {"status": "escalated"}

    # 5. answer: roda RAG e registra a tentativa.
    # `prior_question` mantém o tópico no retrieval em follow-ups
    # ("quanto tempo tem a prova?" depois de "quando vai ser a ADA?").
    #
    # ``ask()`` é síncrono e leva alguns segundos (HTTP síncrono pra
    # LLM externa + Chroma). Chamar direto em rota async travaria o event loop e
    # bloquearia TODOS os outros requests deste worker (incluindo GETs
    # do admin). ``run_in_threadpool`` joga a chamada numa thread auxiliar
    # e libera o loop pra atender requests concorrentes.
    result = await run_in_threadpool(
        ask,
        question=msg["text"],
        prior_question=plan.prior_question,
    )
    _log_query(db, student, msg["text"], result)

    try:
        session_manager.record_attempt(
            db, plan, question=msg["text"], answer=result.answer,
            chunks_used=result.chunks_used, was_fallback=result.was_fallback,
            latency_ms=result.latency_ms,
        )
        db.commit()
    except Exception as e:
        logger.error(f"Erro ao registrar QAAttempt: {e}")
        db.rollback()

    # Anexa o prompt de feedback (1/2/3) em TODA resposta do RAG. O
    # aluno vê a resposta + as opções no mesmo balão do WhatsApp.
    # Fallback ainda recebe o hint específico no meio (sugere reformular
    # ou /coordenador) — depois vêm as opções 1/2/3 que cobrem ambos.
    if result.was_fallback:
        answer_to_send = result.answer + FALLBACK_HINT_SUFFIX + FEEDBACK_PROMPT_SUFFIX
    else:
        answer_to_send = result.answer + FEEDBACK_PROMPT_SUFFIX

    try:
        await evolution_client.send_text(student.phone_number, answer_to_send)
    except Exception as e:
        logger.error(f"Erro ao enviar mensagem: {e}")

    # NOTA HISTÓRICA: aqui antes saía uma segunda mensagem
    # ``POST_ATTEMPT3_PROMPT`` quando ``plan.attempt_number == 3``,
    # forçando uma decisão final. Hoje o ``FEEDBACK_PROMPT_SUFFIX``
    # (anexado em TODA resposta acima) já pede 1/2/3, então a 2ª
    # mensagem virou redundância visível ao aluno como "poll duplicada".
    # Se aluno responder "2" depois da attempt 3, a próxima mensagem dele
    # (>3) cai direto na escalação automática em plan_interaction.

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Handler de votos de enquete (estrutura comum aos dois formatos)
# ---------------------------------------------------------------------------

async def _handle_poll_vote(vote: dict, db: Session) -> dict:
    poll_id = vote.get("poll_id") or ""
    selected = vote.get("selected") or []
    if not poll_id or not selected:
        return {"status": "poll_ignored"}

    feedback = session_manager.feedback_from_option(selected[0])
    if feedback is None:
        logger.info(f"Poll {poll_id}: opção não reconhecida ({selected[0]!r})")
        return {"status": "poll_unknown_option"}

    session = session_manager.apply_poll_vote(db, poll_id, feedback)
    if session is None:
        logger.info(f"Poll {poll_id}: nenhuma sessão casa com esse id")
        return {"status": "poll_no_match"}

    logger.info(
        f"Poll {poll_id}: vote={feedback} | session={session.id} "
        f"status={session.status}"
    )

    # Voto "não resolveu" SEMPRE dispara escalação — independente do
    # status atual da sessão. Justificativa: o aluno está dizendo "preciso
    # de humano". Se a sessão estava resolved (ex.: ele mandou 'obrigado'
    # antes), reabrimos como escalated. Se já era escalated, o
    # create_escalation é idempotente e não duplica.
    if feedback == "not_resolved":
        student = (
            db.query(Student).filter(Student.id == session.student_id).first()
        )
        if student is None:
            logger.warning(f"Poll {poll_id}: student da session {session.id} sumiu")
            db.commit()
            return {"status": "poll_recorded_no_student"}

        # Verifica se já existia escalation antes de criar — pra não
        # mandar ack duplicado em cliques repetidos.
        from database import Escalation
        already_escalated = (
            db.query(Escalation)
            .filter(Escalation.session_id == session.id)
            .first()
            is not None
        )

        try:
            escalation_service.create_escalation(db, session, student)
            db.commit()
        except Exception as e:
            logger.exception(f"Erro criando escalação via poll: {e}")
            db.rollback()
            return {"status": "poll_escalation_failed"}

        if not already_escalated:
            try:
                await evolution_client.send_text(
                    student.phone_number,
                    "Vou encaminhar sua dúvida ao coordenador. Ele responde "
                    "por aqui assim que possível.",
                )
            except Exception as e:
                logger.error(f"Erro avisando escalação via poll: {e}")
            logger.info(
                f"Poll {poll_id}: aluno escolheu coordenador → "
                f"escalation criada (session {session.id})"
            )
            return {"status": "poll_escalated"}
        logger.info(
            f"Poll {poll_id}: aluno reclicou na opção coordenador — "
            f"escalation já existia (idempotente)"
        )
        return {"status": "poll_already_escalated"}

    # Sessão aberta + aluno disse "resolveu" → fecha como resolved agora.
    if session.status == "open" and feedback in ("resolved_fully", "resolved_partially"):
        session_manager.close_as_resolved(db, session)
        db.commit()
        logger.info(
            f"Poll {poll_id}: aluno marcou {feedback} → sessão fechada como resolved"
        )
        return {"status": "poll_resolved", "feedback": feedback}

    db.commit()
    logger.info(f"Poll {poll_id}: feedback={feedback} para session {session.id}")
    return {"status": "poll_recorded", "feedback": feedback}


# ---------------------------------------------------------------------------
# Roteador
# ---------------------------------------------------------------------------

@router.post("")
async def whatsapp_webhook(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    event = body.get("event", "unknown")
    logger.info(f"Evento: {event}")

    try:
        if event in UPSERT_EVENTS:
            return await handle_messages_upsert(body, db)
        if event in POLL_UPDATE_EVENTS:
            vote = _extract_poll_vote(body)
            if vote is not None:
                return await _handle_poll_vote(vote, db)
    except Exception as e:
        logger.exception(f"Erro processando evento {event}: {e}")
        return {"status": "error"}

    return {"status": "ignored"}
