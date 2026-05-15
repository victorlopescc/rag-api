"""Testa o router /webhook (milestone 2) — classificação + estratégias + poll."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from database import Student, get_db
from main import app
from rag_engine import RAGResponse
from services.dedup import message_dedup


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    message_dedup._seen.clear()
    # Default: aluno NÃO tem live thread aberta. Os testes que precisam
    # exercitar o caminho da thread sobrescrevem com seu próprio patch.
    # Sem esse patch global, o mock_db retorna um MagicMock truthy pra
    # find_live_for_student, fazendo TODO request entrar no relay.
    patcher = patch(
        "routers.webhook.thread_service.find_live_for_student",
        return_value=None,
    )
    patcher.start()
    try:
        yield TestClient(app)
    finally:
        patcher.stop()
        app.dependency_overrides.clear()
        message_dedup._seen.clear()


def _student(phone="5511999999999"):
    return Student(
        id=uuid.uuid4(),
        full_name="Maria",
        matricula="001",
        phone_number=phone,
        lid=None,
        active=True,
    )


def _upsert_body(
    msg_id="MSG-1",
    remote_jid="5511999999999@s.whatsapp.net",
    text="Qual a duração?",
    from_me=False,
):
    return {
        "event": "messages.upsert",
        "data": {
            "key": {"id": msg_id, "remoteJid": remote_jid, "fromMe": from_me},
            "message": {"conversation": text},
            "pushName": "Maria",
        },
    }


def _plan(action="answer", attempt=1, strategy="default",
          session_id=None, prior_intent="none"):
    return MagicMock(
        action=action,
        attempt_number=attempt,
        strategy=strategy,
        session_id=session_id or uuid.uuid4(),
        prior_intent=prior_intent,
    )


# --- fluxos triviais -------------------------------------------------------

def test_unknown_event_is_ignored(client):
    resp = client.post("/webhook", json={"event": "contacts.update", "data": {}})
    assert resp.json()["status"] == "ignored"


def test_fromme_message_is_ignored(client):
    resp = client.post("/webhook", json=_upsert_body(from_me=True))
    assert resp.json()["status"] == "ignored"


def test_empty_text_is_ignored(client):
    resp = client.post("/webhook", json=_upsert_body(text="   "))
    assert resp.json()["status"] == "ignored"


def test_unknown_sender_returns_status(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    resp = client.post("/webhook", json=_upsert_body())
    assert resp.json()["status"] == "unknown_sender"


def test_duplicate_message_is_detected(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _student()
    fake = RAGResponse(answer="ok", was_fallback=False, chunks_used=[], latency_ms=1)

    with patch("routers.webhook.ask", return_value=fake), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=_plan()), \
         patch("routers.webhook.session_manager.record_attempt"), \
         patch("routers.webhook.evolution_client.send_text", new_callable=AsyncMock):
        client.post("/webhook", json=_upsert_body(msg_id="DUP"))
        second = client.post("/webhook", json=_upsert_body(msg_id="DUP"))

    assert second.json()["status"] == "duplicate"


# --- action: answer --------------------------------------------------------

def test_answer_path_uses_strategy_from_plan(client, mock_db):
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    fake = RAGResponse(answer="4 anos", was_fallback=False, chunks_used=[], latency_ms=1)

    plan = _plan(action="answer", attempt=2, strategy="query_rewrite")
    ask_mock = MagicMock(return_value=fake)
    send = AsyncMock()

    with patch("routers.webhook.ask", ask_mock), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=plan), \
         patch("routers.webhook.session_manager.record_attempt") as rec, \
         patch("routers.webhook.evolution_client.send_text", send):
        resp = client.post("/webhook", json=_upsert_body(msg_id="M-A"))

    assert resp.json()["status"] == "ok"
    assert resp.json()["strategy"] == "query_rewrite"
    # Estratégia foi passada ao RAG.
    assert ask_mock.call_args.kwargs["strategy"] == "query_rewrite"
    rec.assert_called_once()
    send.assert_awaited_once()
    assert send.call_args.args[0] == student.phone_number


def test_log_query_extracts_ids_from_dict_chunks(client, mock_db):
    """Regressão: QueryLog.chunks_used é ARRAY(Text). Quando o RAG devolve
    list[dict] em chunks_used, o webhook precisa extrair só os IDs antes
    de gravar — senão psycopg2 quebra com 'can't adapt type dict'."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    fake = RAGResponse(
        answer="ok", was_fallback=False, latency_ms=10,
        chunks_used=[
            {"id": "c-1", "document_id": "d-1", "score": 0.8},
            {"id": "c-2", "document_id": "d-1", "score": 0.7},
        ],
    )
    plan = _plan(action="answer", attempt=1, strategy="default")

    with patch("routers.webhook.ask", return_value=fake), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=plan), \
         patch("routers.webhook.session_manager.record_attempt"), \
         patch("routers.webhook.evolution_client.send_text", new_callable=AsyncMock):
        resp = client.post("/webhook", json=_upsert_body(msg_id="LOG-TEST"))

    assert resp.json()["status"] == "ok"
    # mock_db.add foi chamado com QueryLog cujo chunks_used é list[str].
    log_call = next(
        c for c in mock_db.add.call_args_list
        if type(c.args[0]).__name__ == "QueryLog"
    )
    log = log_call.args[0]
    assert log.chunks_used == ["c-1", "c-2"]


# --- action: ack_yes -------------------------------------------------------

def test_ack_yes_sends_only_ack_no_poll(client, mock_db):
    """Após 'obrigado': ack cordial e fim. Sem poll (Evolution não entrega)."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student

    plan = _plan(action="ack_yes", attempt=0)
    send_text = AsyncMock()
    send_poll = AsyncMock(return_value="POLL-1")
    ask_mock = MagicMock()

    with patch("routers.webhook.ask", ask_mock), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=plan), \
         patch("routers.webhook.evolution_client.send_text", send_text), \
         patch("routers.webhook.evolution_client.send_poll", send_poll):
        resp = client.post("/webhook", json=_upsert_body(text="obrigado!", msg_id="THX"))

    assert resp.json()["status"] == "resolved"
    ask_mock.assert_not_called()
    send_text.assert_awaited_once()
    send_poll.assert_not_awaited()  # poll removida
    # Ack menciona o atalho /coordenador como saída opcional
    assert "/coordenador" in send_text.call_args.args[1].lower()


# --- Live thread relay -----------------------------------------------------

def test_message_in_live_thread_is_relayed_not_sent_to_bot(client, mock_db):
    """Aluno em live thread: mensagem vai pra thread, NUNCA chama RAG/triagem."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student

    fake_esc = MagicMock()  # qualquer truthy serve — find_live retorna ele
    append_mock = MagicMock()
    ask_mock = MagicMock()
    plan_mock = MagicMock()

    with patch("routers.webhook.thread_service.find_live_for_student", return_value=fake_esc), \
         patch("routers.webhook.thread_service.append_message", append_mock), \
         patch("routers.webhook.ask", ask_mock), \
         patch("routers.webhook.session_manager.plan_interaction", plan_mock):
        resp = client.post("/webhook", json=_upsert_body(text="oi coord", msg_id="THR-1"))

    assert resp.json()["status"] == "thread_relayed"
    append_mock.assert_called_once()
    ask_mock.assert_not_called()           # bot NÃO foi acionado
    plan_mock.assert_not_called()          # nem o planejador de sessão


def test_end_thread_command_closes_live_and_notifies(client, mock_db):
    """`/encerrar` durante live thread: fecha a thread e avisa o aluno."""
    from services.whatsapp import THREAD_CLOSED_BY_STUDENT_NOTICE
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student

    fake_esc = MagicMock()
    close_mock = MagicMock()
    send_text = AsyncMock()

    with patch("routers.webhook.thread_service.find_live_for_student", return_value=fake_esc), \
         patch("routers.webhook.thread_service.close_live", close_mock), \
         patch("routers.webhook.evolution_client.send_text", send_text):
        resp = client.post("/webhook", json=_upsert_body(text="/encerrar", msg_id="END-1"))

    assert resp.json()["status"] == "thread_closed_by_student"
    close_mock.assert_called_once()
    assert close_mock.call_args.kwargs["reason"] == "student"
    send_text.assert_awaited_once()
    assert send_text.call_args.args[1] == THREAD_CLOSED_BY_STUDENT_NOTICE


# --- action: thanks --------------------------------------------------------

def test_thanks_without_session_sends_polite_ack_no_rag(client, mock_db):
    """'Obrigado' sem sessão aberta: ack educado, sem RAG, sem fallback rude."""
    from services.whatsapp import THANKS_REPLY
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student

    plan = _plan(action="thanks", attempt=0)
    plan.session_id = None
    send_text = AsyncMock()
    ask_mock = MagicMock()

    with patch("routers.webhook.ask", ask_mock), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=plan), \
         patch("routers.webhook.evolution_client.send_text", send_text):
        resp = client.post("/webhook", json=_upsert_body(text="obrigado", msg_id="THX"))

    assert resp.json()["status"] == "thanks"
    ask_mock.assert_not_called()  # NUNCA chama o RAG pra um "obrigado"
    send_text.assert_awaited_once()
    sent = send_text.call_args.args[1]
    assert sent == THANKS_REPLY


# --- action: escalate ------------------------------------------------------

def test_escalate_creates_escalation_and_sends_notice(client, mock_db):
    """Escalation só envia o aviso, sem poll de feedback."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student

    plan = _plan(action="escalate", attempt=3, strategy="widen_k")
    send_text = AsyncMock()
    send_poll = AsyncMock(return_value="POLL-E")
    ask_mock = MagicMock()

    with patch("routers.webhook.ask", ask_mock), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=plan), \
         patch("routers.webhook.escalation_service.create_escalation") as create_esc, \
         patch("routers.webhook.evolution_client.send_text", send_text), \
         patch("routers.webhook.evolution_client.send_poll", send_poll):
        resp = client.post("/webhook", json=_upsert_body(msg_id="ESC"))

    assert resp.json()["status"] == "escalated"
    ask_mock.assert_not_called()
    create_esc.assert_called_once()
    send_text.assert_awaited_once()
    assert "coordenador" in send_text.call_args.args[1].lower()
    send_poll.assert_not_awaited()  # poll removida


# --- LID resolver integration ---------------------------------------------

def test_lid_message_resolved_via_resolver(client, mock_db):
    student = _student()
    fake = RAGResponse(answer="ok", was_fallback=False, chunks_used=[], latency_ms=1)

    with patch("routers.webhook.resolve_student_by_lid", return_value=student) as resolver, \
         patch("routers.webhook.ask", return_value=fake), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=_plan()), \
         patch("routers.webhook.session_manager.record_attempt"), \
         patch("routers.webhook.evolution_client.send_text", new_callable=AsyncMock):
        resp = client.post("/webhook", json=_upsert_body(
            msg_id="LID-1", remote_jid="240247:6@lid"
        ))

    assert resp.json()["status"] == "ok"
    resolver.assert_called_once()


# --- erro geral ------------------------------------------------------------

def test_webhook_handles_plan_exception(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _student()

    with patch(
        "routers.webhook.session_manager.plan_interaction",
        side_effect=RuntimeError("boom"),
    ):
        resp = client.post("/webhook", json=_upsert_body(msg_id="ERR-1"))

    assert resp.json()["status"] == "error"


# --- polls: votos via messages.update --------------------------------------

def test_poll_vote_via_messages_update_records_feedback(client, mock_db):
    fake_session = MagicMock(id=uuid.uuid4())
    with patch(
        "routers.webhook.session_manager.apply_poll_vote",
        return_value=fake_session,
    ) as apply:
        body = {
            "event": "messages.update",
            "data": {
                "key": {"id": "OUTER-KEY"},
                "update": {
                    "pollUpdates": [{
                        "pollCreationMessageKey": {"id": "POLL-1"},
                        "vote": {
                            "selectedOptions": [{"optionName": "Sim, tudo resolvido"}],
                        },
                    }],
                },
            },
        }
        resp = client.post("/webhook", json=body)

    assert resp.json()["status"] == "poll_recorded"
    assert resp.json()["feedback"] == "resolved_fully"
    apply.assert_called_once()
    assert apply.call_args.args[1] == "POLL-1"
    assert apply.call_args.args[2] == "resolved_fully"


def test_poll_vote_via_messages_upsert_pollupdate_escalates_when_no_prior(client, mock_db):
    """Voto 'Não' SEMPRE escala (idempotente). Mesmo em sessão fechada,
    se não houve escalation anterior, cria uma nova."""
    fake_session = MagicMock(id=uuid.uuid4(), status="resolved", student_id=uuid.uuid4())
    fake_student = _student()
    # Mock da query de Escalation retorna None (não havia escalation prévia)
    mock_db.query.return_value.filter.return_value.first.return_value = None

    # Quando o handler busca Student, queremos o student. Quando busca
    # Escalation, queremos None. Diferenciamos pelo argumento de query.
    def query_dispatch(model):
        result = MagicMock()
        if "Student" in str(model):
            result.filter.return_value.first.return_value = fake_student
        else:  # Escalation
            result.filter.return_value.first.return_value = None
        return result
    mock_db.query.side_effect = query_dispatch

    with patch(
        "routers.webhook.session_manager.apply_poll_vote",
        return_value=fake_session,
    ), patch(
        "routers.webhook.escalation_service.create_escalation",
    ), patch(
        "routers.webhook.evolution_client.send_text",
        new_callable=AsyncMock,
    ):
        body = {
            "event": "messages.upsert",
            "data": {
                "key": {"id": "ANY"},
                "message": {
                    "pollUpdateMessage": {
                        "pollCreationMessageKey": {"id": "POLL-B"},
                        "vote": {
                            "selectedOptions": [{"optionName": "Não, falar com coordenador"}],
                        },
                    },
                },
            },
        }
        resp = client.post("/webhook", json=body)

    assert resp.json()["status"] == "poll_escalated"


def test_poll_vote_open_session_not_resolved_triggers_escalation(client, mock_db):
    """Voto 'Não, falar com coordenador' em sessão AINDA ABERTA → escala."""
    fake_session = MagicMock(id=uuid.uuid4(), status="open", student_id=uuid.uuid4())
    fake_student = _student()

    def query_dispatch(model):
        result = MagicMock()
        if "Student" in str(model):
            result.filter.return_value.first.return_value = fake_student
        else:  # Escalation lookup → None (primeira vez)
            result.filter.return_value.first.return_value = None
        return result
    mock_db.query.side_effect = query_dispatch

    with patch(
        "routers.webhook.session_manager.apply_poll_vote",
        return_value=fake_session,
    ), patch(
        "routers.webhook.escalation_service.create_escalation",
    ) as create_esc, patch(
        "routers.webhook.evolution_client.send_text",
        new_callable=AsyncMock,
    ) as send_text:
        body = {
            "event": "messages.update",
            "data": {
                "update": {
                    "pollUpdates": [{
                        "pollCreationMessageKey": {"id": "POLL-OPEN"},
                        "vote": {
                            "selectedOptions": [{"optionName": "Não, falar com coordenador"}],
                        },
                    }],
                },
            },
        }
        resp = client.post("/webhook", json=body)

    assert resp.json()["status"] == "poll_escalated"
    create_esc.assert_called_once()
    send_text.assert_awaited_once()
    assert "coordenador" in send_text.call_args.args[1].lower()


def test_poll_vote_open_session_resolved_closes_session(client, mock_db):
    """Voto 'Resolvi' em sessão aberta fecha como resolved."""
    fake_session = MagicMock(id=uuid.uuid4(), status="open")
    with patch(
        "routers.webhook.session_manager.apply_poll_vote",
        return_value=fake_session,
    ), patch(
        "routers.webhook.session_manager.close_as_resolved",
    ) as close_resolved:
        body = {
            "event": "messages.update",
            "data": {
                "update": {
                    "pollUpdates": [{
                        "pollCreationMessageKey": {"id": "POLL-OK"},
                        "vote": {
                            "selectedOptions": [{"optionName": "Sim, tudo resolvido"}],
                        },
                    }],
                },
            },
        }
        resp = client.post("/webhook", json=body)

    assert resp.json()["status"] == "poll_resolved"
    assert resp.json()["feedback"] == "resolved_fully"
    close_resolved.assert_called_once()


def test_text_fallback_vote_3_escalates(client, mock_db):
    """Aluno responde '3' → escala via voto por texto."""
    fake_session = MagicMock(
        id=uuid.uuid4(),
        status="open",
        student_id=uuid.uuid4(),
    )
    fake_student = _student()

    def query_dispatch(model):
        result = MagicMock()
        if "QASession" in str(model):
            # Última sessão recente do aluno
            result.filter.return_value.order_by.return_value.first.return_value = fake_session
            result.filter.return_value.first.return_value = fake_session
        elif "Student" in str(model):
            result.filter.return_value.first.return_value = fake_student
        else:  # Escalation
            result.filter.return_value.first.return_value = None
        return result
    mock_db.query.side_effect = query_dispatch

    with patch(
        "routers.webhook.escalation_service.create_escalation",
    ) as create_esc, patch(
        "routers.webhook.evolution_client.send_text",
        new_callable=AsyncMock,
    ) as send_text:
        body = _upsert_body(text="3", msg_id="TXT-VOTE-3")
        resp = client.post("/webhook", json=body)

    assert resp.json()["status"] == "text_vote_escalated"
    create_esc.assert_called_once()
    send_text.assert_awaited_once()
    assert "coordenador" in send_text.call_args.args[1].lower()


def test_text_fallback_vote_1_closes_resolved(client, mock_db):
    """Aluno responde '1' → fecha como resolved e manda ack."""
    fake_session = MagicMock(
        id=uuid.uuid4(),
        status="open",
        student_id=uuid.uuid4(),
    )
    fake_student = _student()

    def query_dispatch(model):
        result = MagicMock()
        if "QASession" in str(model):
            result.filter.return_value.order_by.return_value.first.return_value = fake_session
            result.filter.return_value.first.return_value = fake_session
        elif "Student" in str(model):
            result.filter.return_value.first.return_value = fake_student
        return result
    mock_db.query.side_effect = query_dispatch

    with patch(
        "routers.webhook.session_manager.close_as_resolved",
    ) as close_resolved, patch(
        "routers.webhook.evolution_client.send_text",
        new_callable=AsyncMock,
    ) as send_text:
        body = _upsert_body(text="1", msg_id="TXT-VOTE-1")
        resp = client.post("/webhook", json=body)

    assert resp.json()["status"] == "text_vote_recorded"
    assert resp.json()["feedback"] == "resolved_fully"
    close_resolved.assert_called_once()
    send_text.assert_awaited_once()


def test_poll_vote_unknown_option_is_reported(client, mock_db):
    body = {
        "event": "messages.update",
        "data": {
            "update": {
                "pollUpdates": [{
                    "pollCreationMessageKey": {"id": "POLL-X"},
                    "vote": {"selectedOptions": [{"optionName": "talvez"}]},
                }],
            },
        },
    }
    resp = client.post("/webhook", json=body)
    assert resp.json()["status"] == "poll_unknown_option"


def test_poll_vote_with_no_matching_session(client, mock_db):
    with patch(
        "routers.webhook.session_manager.apply_poll_vote", return_value=None,
    ):
        body = {
            "event": "messages.update",
            "data": {
                "update": {
                    "pollUpdates": [{
                        "pollCreationMessageKey": {"id": "GHOST"},
                        "vote": {"selectedOptions": [{"optionName": "Sim, tudo resolvido"}]},
                    }],
                },
            },
        }
        resp = client.post("/webhook", json=body)

    assert resp.json()["status"] == "poll_no_match"


# --- triagem (greeting / trivial / help / cancel / fallback hint) ----------

def test_greeting_short_circuits_rag(client, mock_db):
    """'oi' não chama o RAG nem session_manager — só responde GREETING_REPLY."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    send_text = AsyncMock()
    plan_int = MagicMock()

    with patch("routers.webhook.ask") as ask_mock, \
         patch("routers.webhook.session_manager.plan_interaction", plan_int), \
         patch("routers.webhook.evolution_client.send_text", send_text):
        resp = client.post("/webhook", json=_upsert_body(text="oi", msg_id="GR-1"))

    assert resp.json()["status"] == "greeting"
    ask_mock.assert_not_called()
    plan_int.assert_not_called()
    send_text.assert_awaited_once()
    assert "ada" in send_text.call_args.args[1].lower()  # tem exemplos


def test_trivial_short_circuits_rag(client, mock_db):
    """Mensagem trivial sem conteúdo ('kkk') não chama o RAG."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    with patch("routers.webhook.ask") as ask_mock, \
         patch("routers.webhook.session_manager.plan_interaction") as plan_int, \
         patch(
            "routers.webhook.evolution_client.send_text", new_callable=AsyncMock,
         ) as send_text:
        resp = client.post("/webhook", json=_upsert_body(text="kkk", msg_id="TR-1"))

    assert resp.json()["status"] == "trivial"
    ask_mock.assert_not_called()
    plan_int.assert_not_called()
    send_text.assert_awaited_once()


def test_help_command_sends_help_message(client, mock_db):
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    with patch("routers.webhook.ask") as ask_mock, \
         patch(
            "routers.webhook.evolution_client.send_text", new_callable=AsyncMock,
         ) as send_text:
        resp = client.post("/webhook", json=_upsert_body(text="/ajuda", msg_id="HLP"))

    assert resp.json()["status"] == "help_sent"
    ask_mock.assert_not_called()
    send_text.assert_awaited_once()
    body = send_text.call_args.args[1]
    # /ajuda deve listar /cancelar (que é oculto na boas-vindas)
    assert "/cancelar" in body.lower()


def test_cancel_command_closes_open_session(client, mock_db):
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    with patch(
            "routers.webhook.session_manager.cancel_open_session",
            return_value=True,
         ) as cancel, \
         patch(
            "routers.webhook.evolution_client.send_text", new_callable=AsyncMock,
         ) as send_text:
        resp = client.post("/webhook", json=_upsert_body(text="/cancelar", msg_id="CXL"))

    assert resp.json()["status"] == "cancelled"
    cancel.assert_called_once()
    send_text.assert_awaited_once()
    assert "encerrada" in send_text.call_args.args[1].lower()


def test_cancel_command_with_no_open_session(client, mock_db):
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    with patch(
            "routers.webhook.session_manager.cancel_open_session",
            return_value=False,
         ), \
         patch(
            "routers.webhook.evolution_client.send_text", new_callable=AsyncMock,
         ) as send_text:
        resp = client.post("/webhook", json=_upsert_body(text="/cancelar", msg_id="CXL2"))

    assert resp.json()["status"] == "no_session_to_cancel"
    send_text.assert_awaited_once()
    assert "nenhuma" in send_text.call_args.args[1].lower()


def test_fallback_response_appends_hint(client, mock_db):
    """Quando o RAG retorna fallback, anexa o sufixo orientando o aluno."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    fake = RAGResponse(
        answer="Não encontrei essa informação nos documentos disponíveis.",
        was_fallback=True, latency_ms=1, chunks_used=[],
    )
    plan = _plan(action="answer", attempt=1, strategy="default")

    with patch("routers.webhook.ask", return_value=fake), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=plan), \
         patch("routers.webhook.session_manager.record_attempt"), \
         patch(
            "routers.webhook.evolution_client.send_text", new_callable=AsyncMock,
         ) as send_text:
        client.post("/webhook", json=_upsert_body(text="qual o preço da cantina?"))

    sent = send_text.call_args.args[1]
    assert "Não encontrei essa informação" in sent
    assert "/coordenador" in sent  # hint sufixo


def test_non_fallback_response_does_not_append_hint(client, mock_db):
    """Resposta normal NÃO recebe o sufixo de fallback."""
    student = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = student
    fake = RAGResponse(
        answer="A ADA será de 15 a 19 de junho.",
        was_fallback=False, latency_ms=1, chunks_used=[],
    )
    plan = _plan(action="answer", attempt=1, strategy="default")

    with patch("routers.webhook.ask", return_value=fake), \
         patch("routers.webhook.session_manager.plan_interaction", return_value=plan), \
         patch("routers.webhook.session_manager.record_attempt"), \
         patch(
            "routers.webhook.evolution_client.send_text", new_callable=AsyncMock,
         ) as send_text:
        client.post("/webhook", json=_upsert_body(text="quando vai ser a ada?"))

    sent = send_text.call_args.args[1]
    assert sent == "A ADA será de 15 a 19 de junho."  # sem hint
