"""Analytics endpoints for the TCC dashboard.

All routes are under ``/admin/analytics`` and protected by ``X-API-Key``.

Each endpoint accepts optional ISO-8601 ``since`` and ``until`` timestamps.
Defaults cover "all time" when both are omitted. Aggregates are computed with
SQLAlchemy so they work on both Postgres (production) and SQLite (tests).
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import and_, case, func
from sqlalchemy.orm import Session

from auth import require_api_key
from database import (
    Chunk,
    Document,
    Escalation,
    QAAttempt,
    QASession,
    Student,
    get_db,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/analytics",
    tags=["admin-analytics"],
    dependencies=[Depends(require_api_key)],
)


# ---------------------------------------------------------------------------
# Date-range helper
# ---------------------------------------------------------------------------

def _range(since: datetime | None, until: datetime | None) -> tuple[datetime, datetime]:
    """Normalises an optional (since, until) pair.

    If both are omitted we return a 10-year window ending now, which is
    effectively "all time" for a project that just started.
    """
    now = datetime.now(timezone.utc)
    end = until or now
    start = since or (end - timedelta(days=365 * 10))
    if start > end:
        raise HTTPException(status_code=400, detail="since > until")
    return start, end


def _between(col, start: datetime, end: datetime):
    return and_(col >= start, col <= end)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class OverviewKPIs(BaseModel):
    window_start: str
    window_end: str

    total_sessions: int
    sessions_resolved: int
    sessions_escalated: int
    sessions_abandoned: int
    sessions_open: int

    resolution_rate: float      # resolved / (resolved + escalated + abandoned)
    escalation_rate: float      # escalated / (resolved + escalated + abandoned)

    total_attempts: int
    fallback_attempts: int
    fallback_rate: float        # fallback / total_attempts

    avg_latency_ms: float | None
    p95_latency_ms: float | None

    total_escalations: int
    pending_escalations: int
    avg_reply_minutes: float | None
    median_reply_minutes: float | None


class StrategyRow(BaseModel):
    strategy: str
    attempts: int
    fallback: int
    fallback_rate: float
    explicit_yes: int
    explicit_no: int
    implicit_rephrase: int
    implicit_new_topic: int
    timeout: int
    avg_latency_ms: float | None


class FeedbackByAttempt(BaseModel):
    attempt_number: int
    explicit_yes: int
    explicit_no: int
    implicit_rephrase: int
    implicit_new_topic: int
    timeout: int
    no_signal: int


class StrategyReport(BaseModel):
    per_strategy: list[StrategyRow]
    feedback_by_attempt: list[FeedbackByAttempt]


class LabelCount(BaseModel):
    label: str
    count: int


class StatusCount(BaseModel):
    status: str
    count: int


class ReplyBucket(BaseModel):
    bucket: str  # "<1h" | "1-6h" | "6-24h" | ">24h" | "pending"
    count: int


class ClosingFeedbackCount(BaseModel):
    feedback: str  # resolved_fully | resolved_partially | not_resolved | no_vote
    count: int


class EscalationsReport(BaseModel):
    total: int
    by_label: list[LabelCount]
    by_status: list[StatusCount]
    reply_time_buckets: list[ReplyBucket]
    closing_feedback: list[ClosingFeedbackCount]


class DocumentRow(BaseModel):
    document_id: str | None
    filename: str
    category: str | None
    status: str
    attempts_used: int
    fallback_attempts: int
    fallback_rate: float
    avg_score: float | None


class DocumentsReport(BaseModel):
    rows: list[DocumentRow]
    never_retrieved: list[DocumentRow]


class TimeSeriesPoint(BaseModel):
    date: str               # YYYY-MM-DD
    sessions_opened: int
    sessions_resolved: int
    sessions_escalated: int
    attempts: int
    fallback_attempts: int


class TimeSeriesReport(BaseModel):
    points: list[TimeSeriesPoint]


class TopicCategoryRow(BaseModel):
    category: str              # "ppc" | "tcc" | ... | "sem_categoria"
    attempts: int
    fallback_attempts: int
    fallback_rate: float
    escalations: int           # sessões escaladas cuja 1ª tentativa caiu nessa categoria


class TopicTerm(BaseModel):
    term: str
    count: int
    # Categoria do documento mais associado a esse termo (derivada do primeiro
    # chunk recuperado em cada tentativa onde o termo aparece). ``None`` se
    # o termo só veio de perguntas sem chunk recuperado.
    category: str | None = None


class TopicsReport(BaseModel):
    by_category: list[TopicCategoryRow]
    top_terms: list[TopicTerm]
    total_attempts: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return round(a / b, 4) if b else 0.0


def _count_by(db: Session, col, start, end, filter_col, values: list[str]) -> dict[str, int]:
    """Generic "count grouped by one column within a time range" helper."""
    q = (
        db.query(col, func.count())
        .filter(_between(filter_col, start, end))
        .group_by(col)
    )
    result = {v: 0 for v in values}
    for k, n in q.all():
        key = k if k is not None else ""
        if key in result:
            result[key] = n
    return result


def _percentile(values: list[float], p: float) -> float | None:
    """Cheap percentile (no numpy dep). Returns None on empty."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


# ---------------------------------------------------------------------------
# Overview — row of KPIs
# ---------------------------------------------------------------------------

@router.get("/overview", response_model=OverviewKPIs)
def overview(
    since: datetime | None = None,
    until: datetime | None = None,
    db: Session = Depends(get_db),
):
    start, end = _range(since, until)

    # Sessions within the window (by opened_at).
    sess_counts = _count_by(
        db, QASession.status, start, end, QASession.opened_at,
        ["open", "resolved", "escalated", "abandoned"],
    )
    total_sessions = sum(sess_counts.values())
    closed = sess_counts["resolved"] + sess_counts["escalated"] + sess_counts["abandoned"]

    # Attempts within the window.
    attempt_q = db.query(
        func.count(QAAttempt.id),
        func.sum(case((QAAttempt.was_fallback.is_(True), 1), else_=0)),
        func.avg(QAAttempt.latency_ms),
    ).filter(_between(QAAttempt.created_at, start, end))
    row = attempt_q.one()
    total_attempts = int(row[0] or 0)
    fallback_attempts = int(row[1] or 0)
    avg_latency = float(row[2]) if row[2] is not None else None

    # p95 latency — pulled as a list so we stay SQLite-compatible.
    lats = [
        int(x[0])
        for x in db.query(QAAttempt.latency_ms)
        .filter(_between(QAAttempt.created_at, start, end))
        .filter(QAAttempt.latency_ms.isnot(None))
        .all()
    ]
    p95 = _percentile(lats, 0.95)

    # Escalations within window.
    esc_rows = db.query(Escalation).filter(
        _between(Escalation.created_at, start, end)
    ).all()
    total_escalations = len(esc_rows)
    pending = sum(1 for e in esc_rows if e.status == "pending")
    reply_deltas_min: list[float] = []
    for e in esc_rows:
        if e.replied_at and e.created_at:
            reply_deltas_min.append((e.replied_at - e.created_at).total_seconds() / 60.0)
    avg_reply = round(sum(reply_deltas_min) / len(reply_deltas_min), 2) if reply_deltas_min else None
    median_reply = _percentile(reply_deltas_min, 0.5)
    median_reply = round(median_reply, 2) if median_reply is not None else None

    return OverviewKPIs(
        window_start=start.isoformat(),
        window_end=end.isoformat(),
        total_sessions=total_sessions,
        sessions_resolved=sess_counts["resolved"],
        sessions_escalated=sess_counts["escalated"],
        sessions_abandoned=sess_counts["abandoned"],
        sessions_open=sess_counts["open"],
        resolution_rate=_safe_div(sess_counts["resolved"], closed),
        escalation_rate=_safe_div(sess_counts["escalated"], closed),
        total_attempts=total_attempts,
        fallback_attempts=fallback_attempts,
        fallback_rate=_safe_div(fallback_attempts, total_attempts),
        avg_latency_ms=round(avg_latency, 2) if avg_latency is not None else None,
        p95_latency_ms=round(p95, 2) if p95 is not None else None,
        total_escalations=total_escalations,
        pending_escalations=pending,
        avg_reply_minutes=avg_reply,
        median_reply_minutes=median_reply,
    )


# ---------------------------------------------------------------------------
# Retry-strategy report (the core M2 thesis question)
# ---------------------------------------------------------------------------

_FEEDBACK_SIGNALS = [
    "explicit_yes", "explicit_no",
    "implicit_rephrase", "implicit_new_topic", "timeout",
]


@router.get("/strategies", response_model=StrategyReport)
def strategies(
    since: datetime | None = None,
    until: datetime | None = None,
    db: Session = Depends(get_db),
):
    start, end = _range(since, until)

    # Per-strategy aggregates.
    rows = (
        db.query(
            QAAttempt.retrieval_strategy,
            func.count(QAAttempt.id),
            func.sum(case((QAAttempt.was_fallback.is_(True), 1), else_=0)),
            func.avg(QAAttempt.latency_ms),
        )
        .filter(_between(QAAttempt.created_at, start, end))
        .group_by(QAAttempt.retrieval_strategy)
        .all()
    )

    per_strategy: list[StrategyRow] = []
    for strategy, attempts, fallback, avg_lat in rows:
        signal_counts = dict(
            db.query(QAAttempt.feedback_signal, func.count())
            .filter(_between(QAAttempt.created_at, start, end))
            .filter(QAAttempt.retrieval_strategy == strategy)
            .group_by(QAAttempt.feedback_signal)
            .all()
        )
        per_strategy.append(
            StrategyRow(
                strategy=strategy or "default",
                attempts=int(attempts or 0),
                fallback=int(fallback or 0),
                fallback_rate=_safe_div(int(fallback or 0), int(attempts or 0)),
                explicit_yes=int(signal_counts.get("explicit_yes", 0)),
                explicit_no=int(signal_counts.get("explicit_no", 0)),
                implicit_rephrase=int(signal_counts.get("implicit_rephrase", 0)),
                implicit_new_topic=int(signal_counts.get("implicit_new_topic", 0)),
                timeout=int(signal_counts.get("timeout", 0)),
                avg_latency_ms=round(float(avg_lat), 2) if avg_lat is not None else None,
            )
        )
    per_strategy.sort(key=lambda r: r.attempts, reverse=True)

    # Feedback distribution per attempt number (1, 2, 3).
    fb_by_attempt: list[FeedbackByAttempt] = []
    for n in (1, 2, 3):
        counts = dict(
            db.query(QAAttempt.feedback_signal, func.count())
            .filter(_between(QAAttempt.created_at, start, end))
            .filter(QAAttempt.attempt_number == n)
            .group_by(QAAttempt.feedback_signal)
            .all()
        )
        fb_by_attempt.append(
            FeedbackByAttempt(
                attempt_number=n,
                explicit_yes=int(counts.get("explicit_yes", 0)),
                explicit_no=int(counts.get("explicit_no", 0)),
                implicit_rephrase=int(counts.get("implicit_rephrase", 0)),
                implicit_new_topic=int(counts.get("implicit_new_topic", 0)),
                timeout=int(counts.get("timeout", 0)),
                no_signal=int(counts.get(None, 0)),
            )
        )

    return StrategyReport(per_strategy=per_strategy, feedback_by_attempt=fb_by_attempt)


# ---------------------------------------------------------------------------
# Escalations report
# ---------------------------------------------------------------------------

@router.get("/escalations", response_model=EscalationsReport)
def escalations_report(
    since: datetime | None = None,
    until: datetime | None = None,
    db: Session = Depends(get_db),
):
    start, end = _range(since, until)

    esc = db.query(Escalation).filter(
        _between(Escalation.created_at, start, end)
    ).all()
    total = len(esc)

    label_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    buckets = {"<1h": 0, "1-6h": 0, "6-24h": 0, ">24h": 0, "pending": 0}

    for e in esc:
        label = e.coordinator_label or "não rotulada"
        label_counts[label] = label_counts.get(label, 0) + 1

        status_counts[e.status] = status_counts.get(e.status, 0) + 1

        if e.replied_at and e.created_at:
            hours = (e.replied_at - e.created_at).total_seconds() / 3600.0
            if hours < 1:
                buckets["<1h"] += 1
            elif hours < 6:
                buckets["1-6h"] += 1
            elif hours < 24:
                buckets["6-24h"] += 1
            else:
                buckets[">24h"] += 1
        else:
            buckets["pending"] += 1

    # Closing feedback from qa_sessions linked to these escalations.
    session_ids = [e.session_id for e in esc]
    if session_ids:
        fb_rows = dict(
            db.query(QASession.closing_feedback, func.count())
            .filter(QASession.id.in_(session_ids))
            .group_by(QASession.closing_feedback)
            .all()
        )
    else:
        fb_rows = {}
    closing_fb = [
        ClosingFeedbackCount(feedback=k or "no_vote", count=int(v))
        for k, v in fb_rows.items()
    ]

    return EscalationsReport(
        total=total,
        by_label=[
            LabelCount(label=k, count=v)
            for k, v in sorted(label_counts.items(), key=lambda x: -x[1])
        ],
        by_status=[
            StatusCount(status=k, count=v)
            for k, v in sorted(status_counts.items(), key=lambda x: -x[1])
        ],
        reply_time_buckets=[
            ReplyBucket(bucket=k, count=v) for k, v in buckets.items()
        ],
        closing_feedback=sorted(closing_fb, key=lambda c: -c.count),
    )


# ---------------------------------------------------------------------------
# Documents usage
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=DocumentsReport)
def documents_report(
    since: datetime | None = None,
    until: datetime | None = None,
    db: Session = Depends(get_db),
):
    start, end = _range(since, until)

    # Pull all attempts in window, iterate JSONB in Python. Portable + fine
    # for thesis-sized data (<100k attempts).
    attempts = (
        db.query(QAAttempt.retrieved_chunks, QAAttempt.was_fallback)
        .filter(_between(QAAttempt.created_at, start, end))
        .all()
    )

    # Dados legados: ``retrieved_chunks`` pode ser ``list[str]`` (só chroma_ids)
    # em vez de ``list[dict]``. Nesses casos resolvemos chunk_id → document_id
    # pela tabela ``chunks`` (coluna ``chroma_id``). Fazemos um único lookup
    # em batch pra não bater no banco N vezes por request.
    legacy_ids: set[str] = set()
    for retrieved, _fallback in attempts:
        if not retrieved:
            continue
        for c in retrieved:
            if isinstance(c, str):
                legacy_ids.add(c)

    chroma_to_doc: dict[str, str] = {}
    if legacy_ids:
        for chroma_id, doc_id in (
            db.query(Chunk.chroma_id, Chunk.document_id)
            .filter(Chunk.chroma_id.in_(legacy_ids))
            .all()
        ):
            if chroma_id and doc_id is not None:
                chroma_to_doc[chroma_id] = str(doc_id)

    # {doc_id: {"attempts": n, "fallback": n, "scores": [float]}}
    agg: dict[str, dict] = {}
    for retrieved, fallback in attempts:
        if not retrieved:
            continue
        seen_in_attempt: set[str] = set()
        for c in retrieved:
            if isinstance(c, dict):
                doc_id = c.get("document_id") or c.get("doc_id")
                score = c.get("score")
            elif isinstance(c, str):
                doc_id = chroma_to_doc.get(c)
                score = None
            else:
                continue
            if not doc_id:
                continue
            doc_id = str(doc_id)
            if doc_id in seen_in_attempt:
                continue  # dedupe per attempt
            seen_in_attempt.add(doc_id)
            d = agg.setdefault(doc_id, {"attempts": 0, "fallback": 0, "scores": []})
            d["attempts"] += 1
            if fallback:
                d["fallback"] += 1
            if isinstance(score, (int, float)):
                d["scores"].append(float(score))

    all_docs = {str(d.id): d for d in db.query(Document).all()}

    rows: list[DocumentRow] = []
    for doc_id, stats in agg.items():
        doc = all_docs.get(doc_id)
        rows.append(
            DocumentRow(
                document_id=doc_id,
                filename=doc.filename if doc else "(documento removido)",
                category=doc.category if doc else None,
                status=doc.status if doc else "deleted",
                attempts_used=stats["attempts"],
                fallback_attempts=stats["fallback"],
                fallback_rate=_safe_div(stats["fallback"], stats["attempts"]),
                avg_score=round(sum(stats["scores"]) / len(stats["scores"]), 4)
                if stats["scores"] else None,
            )
        )
    rows.sort(key=lambda r: r.attempts_used, reverse=True)

    used_ids = set(agg.keys())
    never = [
        DocumentRow(
            document_id=str(d.id),
            filename=d.filename,
            category=d.category,
            status=d.status,
            attempts_used=0,
            fallback_attempts=0,
            fallback_rate=0.0,
            avg_score=None,
        )
        for d in all_docs.values() if str(d.id) not in used_ids
    ]
    never.sort(key=lambda r: r.filename)

    return DocumentsReport(rows=rows, never_retrieved=never)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

@router.get("/timeseries", response_model=TimeSeriesReport)
def timeseries(
    since: datetime | None = None,
    until: datetime | None = None,
    db: Session = Depends(get_db),
):
    start, end = _range(since, until)

    # Python-side bucketing keeps us portable across SQLite/Postgres.
    days: dict[str, dict[str, int]] = {}

    def bucket(d: datetime) -> str:
        return d.date().isoformat()

    for opened_at, status in db.query(QASession.opened_at, QASession.status).filter(
        _between(QASession.opened_at, start, end)
    ):
        if opened_at is None:
            continue
        key = bucket(opened_at)
        slot = days.setdefault(key, {
            "sessions_opened": 0, "sessions_resolved": 0, "sessions_escalated": 0,
            "attempts": 0, "fallback_attempts": 0,
        })
        slot["sessions_opened"] += 1
        if status == "resolved":
            slot["sessions_resolved"] += 1
        elif status == "escalated":
            slot["sessions_escalated"] += 1

    for created_at, fb in db.query(QAAttempt.created_at, QAAttempt.was_fallback).filter(
        _between(QAAttempt.created_at, start, end)
    ):
        if created_at is None:
            continue
        key = bucket(created_at)
        slot = days.setdefault(key, {
            "sessions_opened": 0, "sessions_resolved": 0, "sessions_escalated": 0,
            "attempts": 0, "fallback_attempts": 0,
        })
        slot["attempts"] += 1
        if fb:
            slot["fallback_attempts"] += 1

    points = [
        TimeSeriesPoint(date=k, **v)
        for k, v in sorted(days.items())
    ]
    return TimeSeriesReport(points=points)


# ---------------------------------------------------------------------------
# Topics — mais frequentes nas perguntas dos alunos
# ---------------------------------------------------------------------------

# Stopwords em português (mínimo viável; não usamos NLTK pra manter portátil).
_STOPWORDS_PT: frozenset[str] = frozenset("""
a ao aos as à às com como da das de do dos e é em entre era eram essa essas
esse esses esta estas este estes eu for foi foram formos fosse fossem há isso
isto já la las lhe lhes lo los mas me mesma mesmo meu meus minha minhas muito
na nas não nem no nos nossa nossas nosso nossos num numa o os ou para pela pelas
pelo pelos por porque qual quais quando que quem são se seja sejam sem ser será
serão seria seriam seu seus só sob sobre sua suas também te tem tém tinha tinham
toda todas todo todos tu tua tuas um uma umas uns você vocês vos
pra pro pras pros aqui aí onde tudo nada algum alguma alguns algumas
coisa coisas vez vezes ter tenho tive teve tiver tinha tu ei oi olá
posso pode podem poderia gostaria queria quero sabe saber favor
""".split())


def _normalize_term(w: str) -> str:
    """lowercase + remove acentos + tira pontuação das pontas."""
    import unicodedata
    w = w.strip().lower()
    if not w:
        return ""
    # remove acentos
    w = "".join(
        ch for ch in unicodedata.normalize("NFKD", w)
        if not unicodedata.combining(ch)
    )
    # mantém só letras/dígitos
    w = "".join(ch for ch in w if ch.isalnum())
    return w


def _tokenize_question(q: str) -> list[str]:
    import re
    # split em não-alfanumérico (preserva acentuados antes do _normalize_term)
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", q or "")
    out: list[str] = []
    for w in words:
        norm = _normalize_term(w)
        if len(norm) < 3:
            continue
        if norm in _STOPWORDS_PT:
            continue
        if norm.isdigit():
            continue
        out.append(norm)
    return out


@router.get("/topics", response_model=TopicsReport)
def topics(
    since: datetime | None = None,
    until: datetime | None = None,
    top_n: int = Query(25, ge=5, le=100),
    db: Session = Depends(get_db),
):
    """Mais perguntado pelos alunos, em duas visões:

    - **por categoria de documento**: agrupa cada tentativa pela categoria do
      primeiro chunk recuperado (fallback: "sem_categoria"). Mostra volume +
      taxa de fallback + nº de escalações por categoria.
    - **top termos**: contagem dos termos mais frequentes nas perguntas, com
      stopwords em PT removidas.
    """
    start, end = _range(since, until)

    # Carrega as tentativas da janela (+ sessão, pra cruzar com escalated).
    attempts = (
        db.query(
            QAAttempt.question,
            QAAttempt.retrieved_chunks,
            QAAttempt.was_fallback,
            QAAttempt.attempt_number,
            QASession.status,
        )
        .join(QASession, QAAttempt.session_id == QASession.id)
        .filter(_between(QAAttempt.created_at, start, end))
        .all()
    )

    # Lookup chroma_id → document_id, pra suportar dados legados (list[str]).
    legacy_ids: set[str] = set()
    for _q, retrieved, _fb, _n, _st in attempts:
        if not retrieved:
            continue
        for c in retrieved:
            if isinstance(c, str):
                legacy_ids.add(c)
    chroma_to_doc: dict[str, str] = {}
    if legacy_ids:
        for chroma_id, doc_id in (
            db.query(Chunk.chroma_id, Chunk.document_id)
            .filter(Chunk.chroma_id.in_(legacy_ids)).all()
        ):
            if chroma_id and doc_id is not None:
                chroma_to_doc[chroma_id] = str(doc_id)

    # doc_id → category
    doc_to_cat = {
        str(d.id): (d.category or "sem_categoria")
        for d in db.query(Document).all()
    }

    def first_category(retrieved) -> str:
        if not retrieved:
            return "sem_categoria"
        for c in retrieved:
            if isinstance(c, dict):
                did = c.get("document_id") or c.get("doc_id")
            elif isinstance(c, str):
                did = chroma_to_doc.get(c)
            else:
                did = None
            if did:
                return doc_to_cat.get(str(did), "sem_categoria")
        return "sem_categoria"

    # Agrega por categoria.
    cats: dict[str, dict] = {}
    term_counts: dict[str, int] = {}
    # Pra cada termo, quantas vezes apareceu em cada categoria — depois a
    # "categoria dominante" é a de maior contagem.
    term_by_cat: dict[str, dict[str, int]] = {}

    for question, retrieved, was_fallback, attempt_number, sess_status in attempts:
        cat = first_category(retrieved)
        slot = cats.setdefault(cat, {"attempts": 0, "fallback": 0, "esc": 0})
        slot["attempts"] += 1
        if was_fallback:
            slot["fallback"] += 1
        # Conta "escalada" só na 1ª tentativa da sessão pra não duplicar.
        if attempt_number == 1 and sess_status == "escalated":
            slot["esc"] += 1

        for t in _tokenize_question(question or ""):
            term_counts[t] = term_counts.get(t, 0) + 1
            bucket = term_by_cat.setdefault(t, {})
            bucket[cat] = bucket.get(cat, 0) + 1

    by_cat = [
        TopicCategoryRow(
            category=cat,
            attempts=v["attempts"],
            fallback_attempts=v["fallback"],
            fallback_rate=_safe_div(v["fallback"], v["attempts"]),
            escalations=v["esc"],
        )
        for cat, v in cats.items()
    ]
    by_cat.sort(key=lambda r: r.attempts, reverse=True)

    def dominant_cat(term: str) -> str | None:
        buckets = term_by_cat.get(term)
        if not buckets:
            return None
        # Ignora "sem_categoria" no desempate se houver alternativa real.
        real = {k: v for k, v in buckets.items() if k != "sem_categoria"}
        pool = real if real else buckets
        return max(pool.items(), key=lambda kv: kv[1])[0]

    top_terms = [
        TopicTerm(term=t, count=c, category=dominant_cat(t))
        for t, c in sorted(term_counts.items(), key=lambda x: -x[1])[:top_n]
    ]

    return TopicsReport(
        by_category=by_cat,
        top_terms=top_terms,
        total_attempts=sum(v["attempts"] for v in cats.values()),
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

ExportSection = Literal[
    "overview", "strategies", "escalations", "documents", "timeseries", "topics"
]


def _csv_response(rows: list[dict], filename: str) -> StreamingResponse:
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    data = buf.getvalue()
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export/{section}.csv")
def export_csv(
    section: ExportSection,
    since: datetime | None = None,
    until: datetime | None = None,
    db: Session = Depends(get_db),
):
    if section == "overview":
        data = overview(since, until, db).model_dump()
        return _csv_response([data], "overview.csv")

    if section == "strategies":
        rep = strategies(since, until, db)
        return _csv_response(
            [r.model_dump() for r in rep.per_strategy]
            + [r.model_dump() for r in rep.feedback_by_attempt],
            "strategies.csv",
        )

    if section == "escalations":
        rep = escalations_report(since, until, db)
        flat = (
            [{"type": "label", **r.model_dump()} for r in rep.by_label]
            + [{"type": "status", **r.model_dump()} for r in rep.by_status]
            + [{"type": "reply_bucket", **r.model_dump()} for r in rep.reply_time_buckets]
            + [{"type": "closing_fb", **r.model_dump()} for r in rep.closing_feedback]
        )
        return _csv_response(flat, "escalations.csv")

    if section == "documents":
        rep = documents_report(since, until, db)
        flat = (
            [{"bucket": "used", **r.model_dump()} for r in rep.rows]
            + [{"bucket": "never", **r.model_dump()} for r in rep.never_retrieved]
        )
        return _csv_response(flat, "documents.csv")

    if section == "timeseries":
        rep = timeseries(since, until, db)
        return _csv_response([p.model_dump() for p in rep.points], "timeseries.csv")

    if section == "topics":
        rep = topics(since, until, 100, db)
        # Schema unificado pras duas views (category + term) — csv.DictWriter
        # exige fieldnames consistentes entre linhas.
        fields = [
            "type", "category", "attempts", "fallback_attempts", "fallback_rate",
            "escalations", "term", "count",
        ]
        flat: list[dict] = []
        for r in rep.by_category:
            d = r.model_dump()
            flat.append({
                "type": "category",
                "category": d["category"],
                "attempts": d["attempts"],
                "fallback_attempts": d["fallback_attempts"],
                "fallback_rate": d["fallback_rate"],
                "escalations": d["escalations"],
                "term": "", "count": "",
            })
        for r in rep.top_terms:
            flat.append({
                "type": "term",
                "category": "", "attempts": "", "fallback_attempts": "",
                "fallback_rate": "", "escalations": "",
                "term": r.term, "count": r.count,
            })
        # Usa DictWriter manualmente pra manter a ordem das colunas.
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="topics.csv"'},
        )

    raise HTTPException(status_code=400, detail="Seção desconhecida.")
