"""Testa o router /admin (milestone 3)."""
from __future__ import annotations

import uuid
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import ARRAY, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


@compiles(UUID, "sqlite")
def _uuid_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "CHAR(36)"


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "JSON"


@compiles(ARRAY, "sqlite")
def _array_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "JSON"


from database import (  # noqa: E402
    Base, Escalation, QAAttempt, QASession, Student, get_db, utcnow,
)
from main import app  # noqa: E402


@pytest.fixture
def db():
    # TestClient runs endpoints in a worker thread; StaticPool + check_same_thread=False
    # lets the single in-memory connection be shared across threads.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


@pytest.fixture
def client(db):
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def student(db):
    s = Student(full_name="Maria", matricula="001", phone_number="5511999999999")
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


@pytest.fixture
def escalation(db, student):
    sess = QASession(student_id=student.id, status="escalated",
                     closed_at=utcnow(), closing_feedback="not_resolved")
    db.add(sess)
    db.flush()
    for i in range(1, 4):
        db.add(QAAttempt(
            session_id=sess.id, attempt_number=i,
            question=f"Q{i}", answer=f"A{i}",
            retrieval_strategy="default",
            feedback_signal="explicit_no",
        ))
    esc = Escalation(
        session_id=sess.id, student_id=student.id,
        summary="Resumo da dúvida.", status="pending",
    )
    db.add(esc)
    db.commit()
    db.refresh(esc)
    return esc


# --- auth -----------------------------------------------------------------

def test_admin_endpoints_require_api_key(client):
    resp = client.get("/admin/escalations")
    assert resp.status_code == 401


# --- list -----------------------------------------------------------------

def test_list_escalations_returns_rows(client, escalation, api_headers):
    resp = client.get("/admin/escalations", headers=api_headers)
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == str(escalation.id)
    assert items[0]["status"] == "pending"
    assert items[0]["student"]["full_name"] == "Maria"


def test_list_escalations_filters_by_status(client, escalation, api_headers, db):
    # Cria uma segunda escalação já resolvida.
    sess2 = QASession(student_id=escalation.student_id, status="escalated")
    db.add(sess2); db.flush()
    db.add(Escalation(session_id=sess2.id, student_id=escalation.student_id,
                      summary="x", status="coordinator_replied"))
    db.commit()

    resp = client.get(
        "/admin/escalations?status=pending", headers=api_headers,
    )
    items = resp.json()
    assert len(items) == 1
    assert items[0]["status"] == "pending"


# --- detail ---------------------------------------------------------------

def test_get_escalation_includes_attempts(client, escalation, api_headers):
    resp = client.get(
        f"/admin/escalations/{escalation.id}", headers=api_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["attempts"]) == 3
    assert body["attempts"][0]["attempt_number"] == 1
    assert body["closing_feedback"] == "not_resolved"


def test_get_escalation_404(client, api_headers):
    resp = client.get(
        f"/admin/escalations/{uuid.uuid4()}", headers=api_headers,
    )
    assert resp.status_code == 404


# --- patch ----------------------------------------------------------------

def test_patch_escalation_updates_label_and_notes(
    client, escalation, api_headers, db,
):
    resp = client.patch(
        f"/admin/escalations/{escalation.id}",
        json={"coordinator_label": "missing_document",
              "coordinator_notes": "falta o regulamento atualizado"},
        headers=api_headers,
    )
    assert resp.status_code == 200
    db.refresh(escalation)
    assert escalation.coordinator_label == "missing_document"
    assert escalation.coordinator_notes == "falta o regulamento atualizado"


def test_patch_escalation_sets_replied_at_when_reply_present(
    client, escalation, api_headers, db,
):
    resp = client.patch(
        f"/admin/escalations/{escalation.id}",
        json={"coordinator_reply": "Resposta oficial."},
        headers=api_headers,
    )
    assert resp.status_code == 200
    db.refresh(escalation)
    assert escalation.coordinator_reply == "Resposta oficial."
    assert escalation.replied_at is not None


def test_patch_escalation_rejects_unknown_label(
    client, escalation, api_headers,
):
    resp = client.patch(
        f"/admin/escalations/{escalation.id}",
        json={"coordinator_label": "bogus"},
        headers=api_headers,
    )
    assert resp.status_code == 422


# --- reply (milestone 4) ---------------------------------------------------

def test_reply_sends_whatsapp_and_marks_replied(
    client, escalation, api_headers, db,
):
    send = AsyncMock(return_value="MSG-REPLY")
    with patch("routers.admin.evolution_client.send_text", send):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/reply",
            json={"message": "A duração são 8 semestres.",
                  "coordinator_label": "bot_was_wrong"},
            headers=api_headers,
        )
    assert resp.status_code == 200
    db.refresh(escalation)
    assert escalation.status == "coordinator_replied"
    assert escalation.coordinator_reply == "A duração são 8 semestres."
    assert escalation.coordinator_label == "bot_was_wrong"
    assert escalation.replied_at is not None
    # O texto enviado é prefixado com a identificação do coordenador.
    sent_text = send.call_args.args[1]
    assert "Coordenação" in sent_text
    assert "A duração são 8 semestres." in sent_text


def test_reply_rejects_empty_message(client, escalation, api_headers):
    resp = client.post(
        f"/admin/escalations/{escalation.id}/reply",
        json={"message": "   "},
        headers=api_headers,
    )
    assert resp.status_code == 400


def test_reply_refuses_already_replied(client, escalation, api_headers, db):
    escalation.status = "coordinator_replied"
    db.commit()

    send = AsyncMock()
    with patch("routers.admin.evolution_client.send_text", send):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/reply",
            json={"message": "oi"},
            headers=api_headers,
        )
    assert resp.status_code == 409
    send.assert_not_called()


def test_reply_bubbles_up_evolution_failure(
    client, escalation, api_headers, db,
):
    send = AsyncMock(side_effect=RuntimeError("down"))
    with patch("routers.admin.evolution_client.send_text", send):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/reply",
            json={"message": "oi"},
            headers=api_headers,
        )
    assert resp.status_code == 502
    db.refresh(escalation)
    assert escalation.status == "pending"  # não mudou
    assert escalation.coordinator_reply is None


def test_reply_returns_502_when_evolution_returns_none(
    client, escalation, api_headers, db,
):
    send = AsyncMock(return_value=None)
    with patch("routers.admin.evolution_client.send_text", send):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/reply",
            json={"message": "oi"},
            headers=api_headers,
        )
    assert resp.status_code == 502
    db.refresh(escalation)
    assert escalation.status == "pending"


def test_reply_404_for_missing_escalation(client, api_headers):
    send = AsyncMock()
    with patch("routers.admin.evolution_client.send_text", send):
        resp = client.post(
            f"/admin/escalations/{uuid.uuid4()}/reply",
            json={"message": "oi"},
            headers=api_headers,
        )
    assert resp.status_code == 404


# --- maintenance endpoints -------------------------------------------------

def test_maintenance_close_stale_returns_count(client, api_headers):
    with patch(
        "routers.admin.maintenance.close_stale_sessions", return_value=7,
    ):
        resp = client.post(
            "/admin/maintenance/close-stale", headers=api_headers,
        )
    assert resp.status_code == 200
    assert resp.json() == {"closed": 7}


def test_maintenance_end_of_day_returns_count(client, api_headers):
    with patch(
        "routers.admin.maintenance.close_sessions_end_of_day", return_value=3,
    ):
        resp = client.post(
            "/admin/maintenance/end-of-day", headers=api_headers,
        )
    assert resp.status_code == 200
    assert resp.json() == {"closed": 3}


# ---------------------------------------------------------------------------
# Live thread endpoints
# ---------------------------------------------------------------------------

def test_start_thread_opens_live_and_notifies_student(client, escalation, api_headers, db):
    send_text = AsyncMock(return_value="MSG-OPEN")
    with patch("routers.admin.evolution_client.send_text", send_text):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/thread/start",
            headers=api_headers,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "live"
    assert body["live_opened_at"] is not None
    send_text.assert_awaited_once()
    # A mensagem de aviso menciona "/encerrar" pro aluno.
    assert "/encerrar" in send_text.call_args.args[1]


def test_start_thread_rolls_back_when_whatsapp_fails(client, escalation, api_headers, db):
    """Se o aviso ao aluno falha, o estado da escalação não fica 'live'
    (não queremos thread aberta sem o aluno saber)."""
    with patch(
        "routers.admin.evolution_client.send_text",
        AsyncMock(side_effect=RuntimeError("evolution down")),
    ):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/thread/start",
            headers=api_headers,
        )
    assert resp.status_code == 502
    db.refresh(escalation)
    assert escalation.status == "pending"  # não virou 'live'


def test_start_thread_conflicts_with_existing_live(client, escalation, api_headers, db, student):
    """Aluno já tem outra escalação em status live → 409."""
    other_sess = QASession(student_id=student.id, status="escalated")
    db.add(other_sess); db.flush()
    other = Escalation(
        session_id=other_sess.id, student_id=student.id,
        summary="x", status="live",
        live_opened_at=utcnow(),
    )
    db.add(other); db.commit()

    with patch("routers.admin.evolution_client.send_text", AsyncMock()):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/thread/start",
            headers=api_headers,
        )
    assert resp.status_code == 409


def test_get_thread_returns_messages_in_order(client, escalation, api_headers, db):
    from database import ThreadMessage
    # Forja uma thread já encerrada com 3 mensagens.
    escalation.status = "resolved"
    escalation.live_opened_at = utcnow() - timedelta(hours=1)
    escalation.live_closed_at = utcnow()
    db.add_all([
        ThreadMessage(escalation_id=escalation.id, direction="coordinator", text="primeira"),
        ThreadMessage(escalation_id=escalation.id, direction="student", text="segunda"),
        ThreadMessage(escalation_id=escalation.id, direction="coordinator", text="terceira"),
    ])
    db.commit()

    resp = client.get(
        f"/admin/escalations/{escalation.id}/thread", headers=api_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "resolved"
    assert [m["text"] for m in body["messages"]] == ["primeira", "segunda", "terceira"]


def test_post_thread_message_sends_to_whatsapp_and_persists(client, escalation, api_headers, db):
    escalation.status = "live"
    escalation.live_opened_at = utcnow()
    escalation.last_activity_at = utcnow()
    db.commit()

    send_text = AsyncMock(return_value="MSG-ABC")
    with patch("routers.admin.evolution_client.send_text", send_text):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/thread/messages",
            json={"text": "Olá aluno, vou te ajudar"},
            headers=api_headers,
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["direction"] == "coordinator"
    assert body["text"] == "Olá aluno, vou te ajudar"
    # Mensagem prefixada com a identificação da coordenação foi enviada.
    sent_text = send_text.call_args.args[1]
    assert "Coordenação" in sent_text
    assert "Olá aluno" in sent_text


def test_post_thread_message_rejects_when_not_live(client, escalation, api_headers):
    # escalation.status default = "pending" — não está live.
    resp = client.post(
        f"/admin/escalations/{escalation.id}/thread/messages",
        json={"text": "tentando enviar"},
        headers=api_headers,
    )
    assert resp.status_code == 409


def test_post_thread_message_rejects_empty(client, escalation, api_headers, db):
    escalation.status = "live"
    db.commit()
    resp = client.post(
        f"/admin/escalations/{escalation.id}/thread/messages",
        json={"text": "   "},
        headers=api_headers,
    )
    assert resp.status_code == 400


def test_close_thread_marks_resolved_and_notifies(client, escalation, api_headers, db):
    escalation.status = "live"
    escalation.live_opened_at = utcnow()
    db.commit()

    send_text = AsyncMock(return_value="MSG-CLOSE")
    with patch("routers.admin.evolution_client.send_text", send_text):
        resp = client.post(
            f"/admin/escalations/{escalation.id}/thread/close",
            headers=api_headers,
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"
    send_text.assert_awaited_once()


def test_close_thread_rejects_when_not_live(client, escalation, api_headers):
    # status pending — não dá pra fechar o que não foi aberto.
    resp = client.post(
        f"/admin/escalations/{escalation.id}/thread/close",
        headers=api_headers,
    )
    assert resp.status_code == 409


def test_maintenance_close_stale_threads_endpoint(client, api_headers):
    with patch(
        "routers.admin.maintenance.close_stale_live_threads",
        new_callable=AsyncMock,
        return_value=2,
    ):
        resp = client.post(
            "/admin/maintenance/close-stale-threads", headers=api_headers,
        )
    assert resp.status_code == 200
    assert resp.json() == {"closed": 2}
