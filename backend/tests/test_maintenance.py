"""Testa services.maintenance: fechamento de sessões ociosas / fim-de-dia."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import ARRAY, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker


@compiles(UUID, "sqlite")
def _uuid_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "CHAR(36)"


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "JSON"


@compiles(ARRAY, "sqlite")
def _array_sqlite(_type, _compiler, **_kw):  # pragma: no cover
    return "JSON"


from database import Base, QAAttempt, QASession, Student, utcnow  # noqa: E402
from services import session_manager  # noqa: E402
from services.maintenance import (  # noqa: E402
    close_sessions_end_of_day,
    close_stale_sessions,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


@pytest.fixture
def student(db):
    s = Student(full_name="Maria", matricula="001", phone_number="5511999999999")
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _make_session(db, student, *, opened_at=None, with_attempt_at=None):
    sess = QASession(student_id=student.id, status="open")
    if opened_at is not None:
        sess.opened_at = opened_at
    db.add(sess)
    db.flush()
    if with_attempt_at is not None:
        att = QAAttempt(
            session_id=sess.id, attempt_number=1,
            question="Q", answer="A", retrieval_strategy="default",
        )
        att.created_at = with_attempt_at
        db.add(att)
    db.commit()
    db.refresh(sess)
    return sess


# --- close_stale_sessions --------------------------------------------------

def test_close_stale_closes_only_idle_beyond_timeout(db, student):
    fresh = _make_session(db, student)  # opened_at ~ now
    stale = _make_session(
        db, student,
        opened_at=utcnow() - session_manager.SESSION_IDLE_TIMEOUT - timedelta(minutes=5),
    )

    n = close_stale_sessions(db)

    db.refresh(fresh)
    db.refresh(stale)
    assert n == 1
    assert fresh.status == "open"
    assert stale.status == "abandoned"


def test_close_stale_considers_last_attempt_time(db, student):
    # Sessão antiga, mas com tentativa recente → NÃO é stale.
    sess = _make_session(
        db, student,
        opened_at=utcnow() - timedelta(days=2),
        with_attempt_at=utcnow() - timedelta(minutes=10),
    )
    assert close_stale_sessions(db) == 0
    db.refresh(sess)
    assert sess.status == "open"


def test_close_stale_noop_when_nothing_to_close(db):
    assert close_stale_sessions(db) == 0


# --- close_sessions_end_of_day --------------------------------------------

def test_end_of_day_closes_sessions_opened_before_today(db, student):
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    today_morning = datetime.now(timezone.utc).replace(
        hour=8, minute=0, second=0, microsecond=0,
    )
    old = _make_session(db, student, opened_at=yesterday)
    new = _make_session(db, student, opened_at=today_morning)

    n = close_sessions_end_of_day(db)

    db.refresh(old)
    db.refresh(new)
    assert n == 1
    assert old.status == "abandoned"
    assert new.status == "open"


def test_end_of_day_respects_custom_now(db, student):
    # Passa `now` futuro — tudo vira "de ontem".
    past = utcnow() - timedelta(hours=1)
    sess = _make_session(db, student, opened_at=past)

    future = utcnow() + timedelta(days=1)
    n = close_sessions_end_of_day(db, now=future)

    db.refresh(sess)
    assert n == 1
    assert sess.status == "abandoned"


def test_end_of_day_marks_feedback_timeout(db, student):
    yesterday = utcnow() - timedelta(days=1)
    sess = _make_session(
        db, student, opened_at=yesterday, with_attempt_at=yesterday,
    )
    close_sessions_end_of_day(db)
    att = db.query(QAAttempt).filter(QAAttempt.session_id == sess.id).first()
    assert att.feedback_signal == "timeout"
