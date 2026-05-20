"""Tests for /admin/analytics endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

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
    Base, Chunk, Document, Escalation, QAAttempt, QASession, Student, get_db, utcnow,
)
from main import app  # noqa: E402


@pytest.fixture
def db():
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
def api_headers():
    from config import settings
    return {"X-API-Key": settings.api_secret_key}


@pytest.fixture
def seeded(db):
    """Seeds a realistic mini-dataset for analytics tests.

    - 1 student
    - 3 sessions: 1 resolved, 1 escalated, 1 abandoned
    - Attempts with varied strategies and fallback flags
    - 1 escalation with reply within 2h
    - 2 documents (one used, one never retrieved)
    """
    student = Student(full_name="Ana", matricula="001", phone_number="5511999999999")
    db.add(student)
    db.flush()

    doc_a = Document(filename="ppc.pdf", file_type="pdf", category="ppc", status="indexed")
    doc_b = Document(filename="tcc.pdf", file_type="pdf", category="tcc", status="indexed")
    db.add_all([doc_a, doc_b])
    db.flush()

    now = datetime.now(timezone.utc)

    # Resolved session — single attempt, explicit_yes.
    s1 = QASession(
        student_id=student.id, status="resolved",
        opened_at=now - timedelta(days=2), closed_at=now - timedelta(days=2),
        closing_feedback="resolved_fully",
    )
    db.add(s1); db.flush()
    db.add(QAAttempt(
        session_id=s1.id, attempt_number=1,
        question="carga horária?", answer="3200h",
        was_fallback=False, latency_ms=900,
        feedback_signal="explicit_yes", resolved=True,
        retrieved_chunks=[{"document_id": str(doc_a.id), "score": 0.7}],
        created_at=now - timedelta(days=2),
    ))

    # Escalated session — 3 attempts, all fallback, then an escalation.
    s2 = QASession(
        student_id=student.id, status="escalated",
        opened_at=now - timedelta(days=1), closed_at=now - timedelta(days=1),
        closing_feedback="not_resolved",
    )
    db.add(s2); db.flush()
    for i, sig in [
        (1, "explicit_no"),
        (2, "implicit_rephrase"),
        (3, "timeout"),
    ]:
        db.add(QAAttempt(
            session_id=s2.id, attempt_number=i,
            question=f"Q{i}", answer="Não encontrei...",
            was_fallback=True, latency_ms=1500 + i * 100,
            feedback_signal=sig,
            retrieved_chunks=[{"document_id": str(doc_a.id), "score": 0.1}],
            created_at=now - timedelta(days=1),
        ))
    esc = Escalation(
        session_id=s2.id, student_id=student.id,
        summary="Aluno tentou 3x sem sucesso.",
        status="coordinator_replied",
        coordinator_label="missing_document",
        created_at=now - timedelta(days=1),
        replied_at=now - timedelta(days=1) + timedelta(hours=2),
    )
    db.add(esc)

    # Abandoned session.
    s3 = QASession(
        student_id=student.id, status="abandoned",
        opened_at=now - timedelta(hours=5), closed_at=now - timedelta(hours=4),
    )
    db.add(s3); db.flush()

    db.commit()
    return {"student": student, "doc_a": doc_a, "doc_b": doc_b}


# ---------------------------------------------------------------------------

def test_overview_without_key_is_401(client, seeded):
    r = client.get("/admin/analytics/overview")
    assert r.status_code == 401


def test_overview_kpis(client, seeded, api_headers):
    r = client.get("/admin/analytics/overview", headers=api_headers)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["total_sessions"] == 3
    assert d["sessions_resolved"] == 1
    assert d["sessions_escalated"] == 1
    assert d["sessions_abandoned"] == 1
    # 1 resolved / 3 closed
    assert d["resolution_rate"] == pytest.approx(1 / 3, rel=1e-3)
    assert d["escalation_rate"] == pytest.approx(1 / 3, rel=1e-3)
    # 4 attempts total, 3 fallback (session 2)
    assert d["total_attempts"] == 4
    assert d["fallback_attempts"] == 3
    assert d["fallback_rate"] == pytest.approx(0.75, rel=1e-3)
    assert d["total_escalations"] == 1
    assert d["pending_escalations"] == 0
    # reply at +2h => 120 minutes
    assert d["avg_reply_minutes"] == pytest.approx(120.0, rel=1e-2)
    assert d["avg_latency_ms"] is not None


def test_overview_respects_since_until(client, seeded, api_headers):
    # A narrow window in the distant past should see zero sessions.
    r = client.get(
        "/admin/analytics/overview",
        headers=api_headers,
        params={"since": "1999-01-01T00:00:00+00:00",
                "until": "2000-01-01T00:00:00+00:00"},
    )
    assert r.status_code == 200
    assert r.json()["total_sessions"] == 0


def test_overview_rejects_inverted_range(client, seeded, api_headers):
    r = client.get(
        "/admin/analytics/overview",
        headers=api_headers,
        params={"since": "2030-01-01T00:00:00+00:00",
                "until": "2020-01-01T00:00:00+00:00"},
    )
    assert r.status_code == 400


def test_strategies_endpoint_was_removed(client, seeded, api_headers):
    """Endpoint /admin/analytics/strategies foi removido junto com a lógica
    de múltiplas estratégias de retrieval. Garante que retorna 404."""
    r = client.get("/admin/analytics/strategies", headers=api_headers)
    assert r.status_code == 404


def test_escalations_report(client, seeded, api_headers):
    r = client.get("/admin/analytics/escalations", headers=api_headers)
    assert r.status_code == 200
    d = r.json()
    assert d["total"] == 1
    assert d["by_label"][0]["label"] == "missing_document"
    assert d["by_label"][0]["count"] == 1
    # replied at +2h → "1-6h" bucket
    buckets = {b["bucket"]: b["count"] for b in d["reply_time_buckets"]}
    assert buckets["1-6h"] == 1
    assert buckets["pending"] == 0
    # closing feedback pulled from linked session
    assert any(c["feedback"] == "not_resolved" for c in d["closing_feedback"])


def test_documents_report(client, seeded, api_headers):
    r = client.get("/admin/analytics/documents", headers=api_headers)
    assert r.status_code == 200
    d = r.json()
    # doc_a is used, doc_b is never retrieved
    used_filenames = [r["filename"] for r in d["rows"]]
    never_filenames = [r["filename"] for r in d["never_retrieved"]]
    assert "ppc.pdf" in used_filenames
    assert "tcc.pdf" in never_filenames
    ppc_row = next(r for r in d["rows"] if r["filename"] == "ppc.pdf")
    # doc_a appears once in s1 (not fallback) + once per attempt in s2 (3 fallback).
    # But we dedupe per-attempt, so 4 attempts mentioning doc_a.
    assert ppc_row["attempts_used"] == 4
    assert ppc_row["fallback_attempts"] == 3
    assert ppc_row["fallback_rate"] == pytest.approx(0.75, rel=1e-3)


def test_timeseries_buckets_by_day(client, seeded, api_headers):
    r = client.get("/admin/analytics/timeseries", headers=api_headers)
    assert r.status_code == 200
    d = r.json()
    # Three days should appear (some may be the same day depending on clock).
    assert len(d["points"]) >= 2
    totals = {"sessions_opened": 0, "attempts": 0, "fallback_attempts": 0}
    for p in d["points"]:
        totals["sessions_opened"] += p["sessions_opened"]
        totals["attempts"] += p["attempts"]
        totals["fallback_attempts"] += p["fallback_attempts"]
    assert totals["sessions_opened"] == 3
    assert totals["attempts"] == 4
    assert totals["fallback_attempts"] == 3


def test_export_overview_csv(client, seeded, api_headers):
    r = client.get("/admin/analytics/export/overview.csv", headers=api_headers)
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    body = r.text
    assert "total_sessions" in body
    assert "3" in body  # total_sessions


def test_documents_report_handles_legacy_string_chunks(db, client, api_headers):
    """Dados antigos gravavam retrieved_chunks como ``list[str]`` (só chroma_id).
    O endpoint deve resolver via tabela ``chunks`` para mapear doc_id."""
    student = Student(full_name="Bia", matricula="002", phone_number="5511888888888")
    doc = Document(filename="legacy.pdf", file_type="pdf", category="ppc", status="indexed")
    db.add_all([student, doc]); db.flush()

    db.add(Chunk(
        document_id=doc.id, chunk_index=0,
        content="x", chroma_id="legacy-chroma-1",
    ))
    db.flush()

    sess = QASession(student_id=student.id, status="resolved")
    db.add(sess); db.flush()
    db.add(QAAttempt(
        session_id=sess.id, attempt_number=1,
        question="q", answer="a",
        was_fallback=False, latency_ms=100,
        retrieved_chunks=["legacy-chroma-1"],  # shape antigo
    ))
    db.commit()

    r = client.get("/admin/analytics/documents", headers=api_headers)
    assert r.status_code == 200
    rows = r.json()["rows"]
    used = {row["filename"]: row for row in rows}
    assert "legacy.pdf" in used
    assert used["legacy.pdf"]["attempts_used"] == 1


def test_topics_by_category_and_terms(client, seeded, api_headers):
    r = client.get("/admin/analytics/topics", headers=api_headers)
    assert r.status_code == 200
    d = r.json()
    # seeded tem 4 tentativas — todas apontam pra doc_a (category=ppc).
    cats = {c["category"]: c for c in d["by_category"]}
    assert "ppc" in cats
    assert cats["ppc"]["attempts"] == 4
    assert cats["ppc"]["fallback_attempts"] == 3
    # escalações: só s2 é escalated, e atttempt_number=1 é da default → 1
    assert cats["ppc"]["escalations"] == 1

    # top_terms: "carga" e "horaria" devem aparecer vindos de "carga horária?"
    terms = {t["term"]: t["count"] for t in d["top_terms"]}
    assert "carga" in terms
    assert "horaria" in terms  # acento removido


def test_topics_terms_carry_dominant_category(client, seeded, api_headers):
    """Cada TopicTerm deve vir com a categoria dominante do doc recuperado."""
    r = client.get("/admin/analytics/topics", headers=api_headers)
    assert r.status_code == 200
    by_term = {t["term"]: t for t in r.json()["top_terms"]}
    # No fixture, TODAS as tentativas apontam pra doc_a (category=ppc),
    # então todos os termos extraídos devem ter category="ppc".
    assert by_term["carga"]["category"] == "ppc"
    assert by_term["horaria"]["category"] == "ppc"


def test_topics_strips_stopwords_and_short_words(client, seeded, api_headers):
    r = client.get("/admin/analytics/topics", headers=api_headers)
    terms = {t["term"]: t["count"] for t in r.json()["top_terms"]}
    # stopwords não podem aparecer
    for sw in ["de", "o", "a", "para", "com"]:
        assert sw not in terms


def test_export_topics_csv(client, seeded, api_headers):
    r = client.get("/admin/analytics/export/topics.csv", headers=api_headers)
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "category" in r.text or "term" in r.text


def test_export_unknown_section_is_422(client, seeded, api_headers):
    # FastAPI path enum → 422
    r = client.get("/admin/analytics/export/bogus.csv", headers=api_headers)
    assert r.status_code == 422
