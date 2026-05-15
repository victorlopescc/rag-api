"""Testa o escalation_service: summary via LLM + persistência idempotente."""
from __future__ import annotations

from unittest.mock import patch

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


from database import Base, Escalation, QAAttempt, QASession, Student  # noqa: E402
from services import escalation_service  # noqa: E402
from services.escalation_service import (  # noqa: E402
    create_escalation,
    summarize_attempts,
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


@pytest.fixture
def session_with_3_attempts(db, student):
    sess = QASession(student_id=student.id, status="open")
    db.add(sess)
    db.flush()
    for i in range(1, 4):
        db.add(QAAttempt(
            session_id=sess.id,
            attempt_number=i,
            question=f"Q{i}",
            answer=f"A{i} (provavelmente errada)",
            retrieval_strategy=["default", "query_rewrite", "widen_k"][i - 1],
            feedback_signal="explicit_no" if i < 3 else None,
        ))
    db.commit()
    db.refresh(sess)
    return sess


# --- summarize_attempts ----------------------------------------------------

def test_summarize_attempts_empty_returns_placeholder():
    assert "sem tentativas" in summarize_attempts([])


def test_summarize_attempts_uses_llm_output():
    att = QAAttempt(attempt_number=1, question="Q", answer="A",
                    retrieval_strategy="default")
    with patch("services.escalation_service.generate",
               return_value="Resumo gerado pelo LLM."):
        out = summarize_attempts([att])
    assert out == "Resumo gerado pelo LLM."


def test_summarize_attempts_falls_back_when_llm_raises():
    att = QAAttempt(attempt_number=1, question="Qual a duração?",
                    answer="A errada", retrieval_strategy="default")
    with patch("services.escalation_service.generate",
               side_effect=RuntimeError("ollama down")):
        out = summarize_attempts([att])
    assert "indisponível" in out.lower()
    assert "Qual a duração?" in out


def test_summarize_attempts_fallback_on_empty_llm():
    att = QAAttempt(attempt_number=1, question="Q", answer="A",
                    retrieval_strategy="default")
    with patch("services.escalation_service.generate", return_value="   "):
        out = summarize_attempts([att])
    assert "indisponível" in out.lower()


# --- create_escalation -----------------------------------------------------

def test_create_escalation_persists_row_and_escalates_session(
    db, student, session_with_3_attempts,
):
    with patch("services.escalation_service.generate", return_value="RESUMO"):
        esc = create_escalation(db, session_with_3_attempts, student)
        db.commit()

    assert esc.id is not None
    assert esc.summary == "RESUMO"
    assert esc.status == "pending"
    assert esc.student_id == student.id
    # Sessão marcada como escalada.
    db.refresh(session_with_3_attempts)
    assert session_with_3_attempts.status == "escalated"
    assert session_with_3_attempts.closed_at is not None


def test_create_escalation_is_idempotent(
    db, student, session_with_3_attempts,
):
    with patch("services.escalation_service.generate", return_value="R1"):
        e1 = create_escalation(db, session_with_3_attempts, student)
        db.commit()
    with patch("services.escalation_service.generate", return_value="R2"):
        e2 = create_escalation(db, session_with_3_attempts, student)
        db.commit()

    assert e1.id == e2.id
    # Não duplicou.
    assert db.query(Escalation).count() == 1
    # Summary continua o da primeira chamada (não sobrescreve).
    assert e2.summary == "R1"


def test_create_escalation_preserves_session_if_already_escalated(
    db, student, session_with_3_attempts,
):
    # Já fecha a sessão como escalated fora do service.
    session_with_3_attempts.status = "escalated"
    db.commit()
    closed_before = session_with_3_attempts.closed_at

    with patch("services.escalation_service.generate", return_value="R"):
        create_escalation(db, session_with_3_attempts, student)
        db.commit()

    db.refresh(session_with_3_attempts)
    # close_as_escalated não é chamado de novo (mantém status + closed_at).
    assert session_with_3_attempts.status == "escalated"
    assert session_with_3_attempts.closed_at == closed_before


def test_render_attempts_includes_strategy_and_feedback(
    db, student, session_with_3_attempts,
):
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "ok"

    with patch("services.escalation_service.generate", side_effect=fake_generate):
        create_escalation(db, session_with_3_attempts, student)

    prompt = captured["prompt"]
    assert "Tentativa 1" in prompt and "Tentativa 3" in prompt
    assert "query_rewrite" in prompt
    assert "widen_k" in prompt
    assert "explicit_no" in prompt
