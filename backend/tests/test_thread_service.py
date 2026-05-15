"""Testa o ciclo de vida da live thread (start_live, close_live, append, lista, stale)."""
from datetime import timedelta

import pytest
from sqlalchemy import ARRAY, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker


# Compatibilidade PG → SQLite (mesmo padrão usado em test_session_manager).
@compiles(UUID, "sqlite")
def _uuid_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "CHAR(36)"


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "JSON"


@compiles(ARRAY, "sqlite")
def _array_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "JSON"


from database import Base, Escalation, QAAttempt, QASession, Student, utcnow  # noqa: E402
from services.thread_service import (  # noqa: E402
    LIVE_TIMEOUT,
    ThreadConflictError,
    ThreadStateError,
    append_message,
    close_live,
    close_stale_live_threads,
    find_live_for_student,
    find_stale_live_threads,
    list_messages,
    start_live,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def student(db):
    s = Student(full_name="Maria", matricula="001", phone_number="5511999999999")
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# --- fixtures locais (db, student vêm de conftest) -------------------------

def _make_escalation(
    db, student, *, status="pending", last_activity_at=None,
) -> Escalation:
    """Cria session+attempt+escalation pra um aluno. Reusa o conftest existente
    mas garante que a Escalation tem session_id válida."""
    sess = QASession(student_id=student.id, status="escalated")
    db.add(sess)
    db.flush()
    att = QAAttempt(
        session_id=sess.id,
        attempt_number=1,
        question="q",
        answer="a",
        retrieval_strategy="default",
        was_fallback=False,
    )
    db.add(att)
    esc = Escalation(
        session_id=sess.id,
        student_id=student.id,
        summary="s",
        status=status,
        last_activity_at=last_activity_at,
    )
    db.add(esc)
    db.flush()
    return esc


# ---------------------------------------------------------------------------
# start_live
# ---------------------------------------------------------------------------

def test_start_live_from_pending_sets_status_and_timestamps(db, student):
    esc = _make_escalation(db, student, status="pending")
    out = start_live(db, esc)
    assert out.status == "live"
    assert out.live_opened_at is not None
    assert out.last_activity_at is not None


def test_start_live_from_coordinator_replied_is_allowed(db, student):
    """Coordenador pode iniciar conversa ao vivo MESMO depois de já ter
    mandado uma resposta única (modo legado)."""
    esc = _make_escalation(db, student, status="coordinator_replied")
    out = start_live(db, esc)
    assert out.status == "live"


def test_start_live_from_resolved_raises(db, student):
    esc = _make_escalation(db, student, status="resolved")
    with pytest.raises(ThreadStateError):
        start_live(db, esc)


def test_start_live_conflicts_when_student_has_other_live_thread(db, student):
    first = _make_escalation(db, student, status="pending")
    start_live(db, first)
    db.flush()

    second = _make_escalation(db, student, status="pending")
    with pytest.raises(ThreadConflictError):
        start_live(db, second)


# ---------------------------------------------------------------------------
# close_live
# ---------------------------------------------------------------------------

def test_close_live_by_coordinator_marks_resolved(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    out = close_live(db, esc, reason="coordinator")
    assert out.status == "resolved"
    assert out.live_closed_at is not None


def test_close_live_by_student_marks_abandoned(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    out = close_live(db, esc, reason="student")
    assert out.status == "abandoned"


def test_close_live_by_timeout_marks_abandoned(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    out = close_live(db, esc, reason="timeout")
    assert out.status == "abandoned"


def test_close_live_when_not_live_raises(db, student):
    esc = _make_escalation(db, student, status="pending")
    with pytest.raises(ThreadStateError):
        close_live(db, esc, reason="coordinator")


# ---------------------------------------------------------------------------
# append_message + list_messages
# ---------------------------------------------------------------------------

def test_append_coordinator_message_updates_last_activity(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    activity_before = esc.last_activity_at
    # força um pequeno gap pra detectar a atualização
    esc.last_activity_at = activity_before - timedelta(minutes=1)
    db.flush()

    append_message(db, esc, direction="coordinator", text="oi aluno")
    assert esc.last_activity_at > activity_before - timedelta(minutes=1)


def test_append_student_message_does_NOT_update_last_activity(db, student):
    """last_activity_at é reset DO COORDENADOR — o aluno mandar
    mensagem não deve segurar a thread aberta indefinidamente."""
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    # Força um valor antigo pra detectar.
    old = utcnow() - timedelta(hours=20)
    esc.last_activity_at = old
    db.flush()

    append_message(db, esc, direction="student", text="oi coord")
    assert esc.last_activity_at == old  # permanece igual


def test_list_messages_returns_chronological_order(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    append_message(db, esc, direction="coordinator", text="primeira")
    append_message(db, esc, direction="student", text="segunda")
    append_message(db, esc, direction="coordinator", text="terceira")

    msgs = list_messages(db, esc)
    assert [m.text for m in msgs] == ["primeira", "segunda", "terceira"]


# ---------------------------------------------------------------------------
# find_live_for_student
# ---------------------------------------------------------------------------

def test_find_live_returns_escalation_when_live(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    out = find_live_for_student(db, student)
    assert out is not None
    assert out.id == esc.id


def test_find_live_returns_none_when_no_live(db, student):
    _make_escalation(db, student, status="pending")
    assert find_live_for_student(db, student) is None


def test_find_live_returns_none_after_close(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    close_live(db, esc, reason="coordinator")
    assert find_live_for_student(db, student) is None


# ---------------------------------------------------------------------------
# find_stale + close_stale
# ---------------------------------------------------------------------------

def test_find_stale_returns_live_threads_with_old_last_activity(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    # Força last_activity_at pra muito atrás.
    esc.last_activity_at = utcnow() - LIVE_TIMEOUT - timedelta(hours=1)
    db.flush()

    stale = find_stale_live_threads(db)
    assert any(s.id == esc.id for s in stale)


def test_find_stale_ignores_fresh_threads(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    # last_activity_at é agora (recém-aberta) — não deveria estar stale.
    stale = find_stale_live_threads(db)
    assert not any(s.id == esc.id for s in stale)


def test_close_stale_marks_threads_as_abandoned(db, student):
    esc = _make_escalation(db, student, status="pending")
    start_live(db, esc)
    esc.last_activity_at = utcnow() - LIVE_TIMEOUT - timedelta(hours=1)
    db.flush()

    closed = close_stale_live_threads(db)
    assert len(closed) == 1
    db.refresh(esc)
    assert esc.status == "abandoned"
    assert esc.live_closed_at is not None
