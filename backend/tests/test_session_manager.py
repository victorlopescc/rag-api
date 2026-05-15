"""Testa o session_manager (milestone 2) — plan_interaction + record_attempt
+ lifecycle helpers + mapeamento de voto da enquete.

Usa SQLite in-memory com UUID/JSONB/ARRAY mapeados para tipos portáveis.
O classificador via LLM é mockado (Ollama não roda nos testes).
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import ARRAY, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker


# --- Compatibilidade PG → SQLite -------------------------------------------
@compiles(UUID, "sqlite")
def _uuid_sqlite(_type, _compiler, **_kw):  # pragma: no cover - glue
    return "CHAR(36)"


@compiles(JSONB, "sqlite")
def _jsonb_sqlite(_type, _compiler, **_kw):  # pragma: no cover - glue
    return "JSON"


@compiles(ARRAY, "sqlite")
def _array_sqlite(_type, _compiler, **_kw):  # pragma: no cover - glue
    return "JSON"


# ---------------------------------------------------------------------------

from database import Base, QAAttempt, QASession, Student, utcnow  # noqa: E402
from services import session_manager  # noqa: E402
from services.session_manager import (  # noqa: E402
    POLL_OPTIONS,
    SESSION_IDLE_TIMEOUT,
    apply_poll_vote,
    classify_fast,
    close_as_abandoned,
    close_as_escalated,
    close_as_resolved,
    feedback_from_option,
    is_escalation_sentinel,
    plan_interaction,
    record_attempt,
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
def llm_classifier():
    """Mocka o classificador LLM com um valor programável por teste."""
    with patch("services.session_manager._classify") as m:
        # Default: regex fast-path fallback behavior (tests override as needed).
        m.side_effect = _default_classify_side_effect
        yield m


def _default_classify_side_effect(message, prior):
    # Reproduz: regex "yes" sempre passa; resto é "unclear".
    return classify_fast(message) if classify_fast(message) == "yes" else "unclear"


# --- classify_fast ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "sim", "obrigado", "obrigada", "blz", "vlw", "valeu", "resolvido",
    "perfeito", "ajudou", "consegui", "1", "ok", "show", "top",
    "obrigado!!!", "sim obrigado", "blz vlw",
])
def test_classify_fast_yes(text):
    assert classify_fast(text) == "yes"


@pytest.mark.parametrize("text", [
    "", "não", "nao", "2", "qual a duração?",
    "não entendi, pode explicar?",
    "obrigado pela explicação mas ainda tenho uma dúvida",
    "sim, mas e quanto ao TCC?",
])
def test_classify_fast_unclear(text):
    assert classify_fast(text) == "unclear"


# --- plan_interaction: primeira mensagem -----------------------------------

def test_plan_without_open_session_opens_new(db, student, llm_classifier):
    plan = plan_interaction(db, student, "Qual a duração do curso?")
    db.commit()

    assert plan.action == "answer"
    assert plan.attempt_number == 1
    assert plan.strategy == "default"
    assert plan.prior_intent == "none"
    # Uma sessão aberta, zero tentativas (record_attempt é separado).
    assert db.query(QASession).count() == 1
    assert db.query(QAAttempt).count() == 0


def test_plan_thanks_without_open_session_returns_thanks_action(db, student, llm_classifier):
    """'Obrigado' isolado sem sessão aberta NÃO deve abrir sessão nem ir
    pro RAG (que daria fallback rude). Devolve action=thanks."""
    plan = plan_interaction(db, student, "obrigado")
    db.commit()

    assert plan.action == "thanks"
    assert plan.session_id is None
    assert plan.attempt_number == 0
    # Nenhuma sessão criada — não há nada pra responder.
    assert db.query(QASession).count() == 0


def test_plan_thanks_variants_all_recognized(db, student, llm_classifier):
    """Variações comuns de gentileza são classificadas como thanks."""
    for greeting in ("obrigado!", "valeu", "blz", "ok", "sim"):
        plan = plan_interaction(db, student, greeting)
        db.commit()
        assert plan.action == "thanks", f"falha em {greeting!r}"
        assert db.query(QASession).count() == 0


# --- plan_interaction: yes fecha sessão ------------------------------------

def test_plan_yes_closes_session(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()

    p2 = plan_interaction(db, student, "obrigado!")
    db.commit()

    assert p2.action == "ack_yes"
    assert p2.attempt_number == 0
    sess = db.query(QASession).first()
    assert sess.status == "resolved"
    assert sess.closed_at is not None

    last = db.query(QAAttempt).first()
    assert last.feedback_signal == "explicit_yes"
    assert last.resolved is True


# --- plan_interaction: no / rephrase escalam attempt_number ----------------

def test_plan_no_keeps_session_and_bumps_strategy(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()

    llm_classifier.side_effect = lambda m, prior: "no"
    p2 = plan_interaction(db, student, "não entendi")
    db.commit()

    assert p2.action == "answer"
    assert p2.attempt_number == 2
    assert p2.strategy == "query_rewrite"
    assert p2.session_id == p1.session_id

    last = db.query(QAAttempt).first()
    assert last.feedback_signal == "explicit_no"
    assert last.resolved is False


def test_plan_rephrase_also_bumps(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()

    llm_classifier.side_effect = lambda m, prior: "rephrase"
    p2 = plan_interaction(db, student, "quero dizer, a duração em semestres")
    db.commit()

    assert p2.attempt_number == 2
    assert p2.strategy == "query_rewrite"
    last = db.query(QAAttempt).first()
    assert last.feedback_signal == "implicit_rephrase"


def test_prior_question_is_attached_when_continuing_session(db, student, llm_classifier):
    """SessionPlan deve carregar a pergunta anterior pra o RAG manter o tópico."""
    p1 = plan_interaction(db, student, "quando vai ser a ADA?")
    record_attempt(db, p1, question="quando vai ser a ADA?", answer="15 a 19 de junho.")
    db.commit()

    # 1ª da sessão não tem prior_question
    assert p1.prior_question is None

    # 2ª pergunta na mesma sessão (rephrase) com mensagem VAGA →
    # prior_question = pergunta anterior (anchor de tópico).
    # Se a nova mensagem tivesse sigla/keyword detectável, _choose_prior
    # descartaria pra evitar contaminar o retrieval.
    llm_classifier.side_effect = lambda m, prior: "rephrase"
    p2 = plan_interaction(db, student, "quanto tempo eu tenho?")
    assert p2.prior_question == "quando vai ser a ADA?"


def test_prior_question_carried_over_even_on_new_topic(db, student, llm_classifier):
    """Em new_topic com mensagem nova VAGA, mantemos prior_question pro RAG.

    Motivação: o intent classifier é conservador e marca como
    'new_topic' qualquer pergunta com vocabulário diferente. Quando a
    mensagem nova é vaga (sem sigla nem keyword de domínio), faz
    sentido manter o anchor anterior pra evitar fallback por perda de
    contexto. Se a nova mensagem TEM sigla/keyword (ex.: "calculadora"
    aciona ADA), aí _choose_prior descarta o prior — caso testado em
    test_prior_dropped_when_new_message_has_acronym.
    """
    p1 = plan_interaction(db, student, "quando vai ser a ADA?")
    record_attempt(db, p1, question="quando vai ser a ADA?", answer="15 a 19 de junho.")
    db.commit()

    llm_classifier.side_effect = lambda m, prior: "new_topic"
    p2 = plan_interaction(db, student, "qual o prazo?")  # mensagem vaga
    assert p2.session_id != p1.session_id
    assert p2.prior_question == "quando vai ser a ADA?"


def test_prior_dropped_when_new_message_has_different_acronym(
    db, student, llm_classifier,
):
    """Aluno passa de ADA pra TCC → prior_question não arrasta o ADA."""
    p1 = plan_interaction(db, student, "quanto vale a ADA?")
    record_attempt(db, p1, question="quanto vale a ADA?", answer="5 pontos.")
    db.commit()

    llm_classifier.side_effect = lambda m, prior: "new_topic"
    p2 = plan_interaction(db, student, "quem orienta o TCC?")
    assert p2.session_id != p1.session_id
    assert p2.prior_question is None  # ADA não arrasta pra TCC


def test_prior_question_lookback_5min_window(db, student, llm_classifier):
    """Lookback curto de 5 min: pergunta recente do mesmo aluno em
    sessão fechada ainda conta como prior_question."""
    from services.session_manager import close_as_resolved
    p1 = plan_interaction(db, student, "quando vai ser a ADA?")
    record_attempt(db, p1, question="quando vai ser a ADA?", answer="15-19 jun.")
    sess = db.query(QASession).filter_by(id=p1.session_id).one()
    close_as_resolved(db, sess)
    db.commit()

    # Mensagem vaga (sem sigla/keyword) — mantém prior do lookback.
    p2 = plan_interaction(db, student, "qual o prazo?")
    assert p2.session_id != p1.session_id
    assert p2.prior_question == "quando vai ser a ADA?"


def test_prior_question_lookback_expires_after_5min(db, student, llm_classifier, monkeypatch):
    """Pergunta de mais de 5 min atrás NÃO entra como prior_question."""
    from datetime import timedelta
    import services.session_manager as sm
    from services.session_manager import close_as_resolved

    p1 = plan_interaction(db, student, "qual a duração do curso?")
    record_attempt(db, p1, question="qual a duração do curso?", answer="4 anos.")
    sess = db.query(QASession).filter_by(id=p1.session_id).one()
    close_as_resolved(db, sess)
    db.commit()

    # Faz a pergunta antiga "envelhecer" 10 min.
    real_now = sm.utcnow
    monkeypatch.setattr(sm, "utcnow", lambda: real_now() + timedelta(minutes=10))

    p2 = plan_interaction(db, student, "posso usar calculadora?")
    assert p2.prior_question is None


def test_plan_unclear_treated_as_rephrase(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()

    llm_classifier.side_effect = lambda m, prior: "unclear"
    p2 = plan_interaction(db, student, "?")
    db.commit()

    assert p2.attempt_number == 2


def test_third_attempt_uses_widen_k(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()

    llm_classifier.side_effect = lambda m, prior: "rephrase"
    p2 = plan_interaction(db, student, "Q2")
    record_attempt(db, p2, question="Q2", answer="A2")
    db.commit()

    p3 = plan_interaction(db, student, "Q3")
    db.commit()

    assert p3.attempt_number == 3
    assert p3.strategy == "widen_k"


def test_fourth_attempt_escalates(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()
    llm_classifier.side_effect = lambda m, prior: "rephrase"
    for _ in range(2):
        p = plan_interaction(db, student, "de novo")
        record_attempt(db, p, question="x", answer="y")
        db.commit()

    p4 = plan_interaction(db, student, "ainda não")
    db.commit()

    assert p4.action == "escalate"


# --- plan_interaction: new_topic encerra a sessão e abre outra --------------

def test_plan_new_topic_opens_new_session(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()

    llm_classifier.side_effect = lambda m, prior: "new_topic"
    p2 = plan_interaction(db, student, "outra coisa")
    db.commit()

    assert p2.action == "answer"
    assert p2.attempt_number == 1
    assert p2.session_id != p1.session_id

    sessions = db.query(QASession).order_by(QASession.opened_at).all()
    assert len(sessions) == 2
    assert sessions[0].status == "abandoned"
    assert sessions[1].status == "open"

    prev = db.query(QAAttempt).first()
    assert prev.feedback_signal == "implicit_new_topic"


# --- escalation sentinel ---------------------------------------------------

@pytest.mark.parametrize("text", [
    "coordenador",
    "Coordenador",
    "/coordenador",
    "/coord",
    "humano",
    "atendimento humano",
    "falar com coordenador",
    "Falar com o coordenador",
    "quero coordenador",
    "quero falar com coordenador",
    "coordenador?",
    "  coordenador  ",
])
def test_is_escalation_sentinel_matches(text):
    assert is_escalation_sentinel(text) is True


@pytest.mark.parametrize("text", [
    "Quem é o coordenador da ADA?",
    "como falar com o coordenador",  # tem texto extra
    "preciso de ajuda",
    "humanos podem fazer ada?",
    "coord",  # match parcial não conta
    "",
    "   ",
])
def test_is_escalation_sentinel_does_not_match(text):
    assert is_escalation_sentinel(text) is False


def test_plan_sentinel_escalates_immediately(db, student, llm_classifier):
    """Sentinel pula classificação e vai direto pra escalate."""
    p = plan_interaction(db, student, "/coordenador")
    db.commit()
    assert p.action == "escalate"
    assert p.prior_intent == "sentinel_escalation"
    # llm_classifier nem foi chamado
    llm_classifier.assert_not_called()


def test_plan_sentinel_works_without_open_session(db, student, llm_classifier):
    """Sentinel cria sessão nova se necessário, pra anexar a escalação."""
    p = plan_interaction(db, student, "coordenador")
    db.commit()
    assert p.action == "escalate"
    sessions = db.query(QASession).all()
    assert len(sessions) == 1
    assert sessions[0].id == p.session_id


# --- text vote fallback ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("1", "resolved_fully"),
    ("2", "resolved_partially"),
    ("3", "not_resolved"),
    (" 1 ", "resolved_fully"),  # trim
    ("4", None),
    ("0", None),
    ("sim", None),  # texto não-numérico não conta como voto
    ("", None),
    ("12", None),
])
def test_text_vote_for(text, expected):
    from services.session_manager import text_vote_for
    assert text_vote_for(text) == expected


# --- _choose_prior — prior_question topic conflict --------------------------

def test_choose_prior_drops_when_new_message_has_acronym():
    """Pergunta nova menciona TCC → não anexa prior sobre ADA."""
    from services.session_manager import _choose_prior
    assert _choose_prior("Quanto vale a ADA?", "Quem orienta o TCC?") is None


def test_choose_prior_drops_when_categories_conflict():
    """Prior fala de ADA, nova fala de TCC → descarta prior."""
    from services.session_manager import _choose_prior
    assert _choose_prior("Quando vai ser a ADA?", "Qual a duração do TCC?") is None


def test_choose_prior_drops_when_new_message_already_has_acronym_even_same_category():
    """Nova mensagem já tem ADA → não precisa de prior, ela carrega o tópico."""
    from services.session_manager import _choose_prior
    assert _choose_prior("Quando vai ser a ADA?", "Posso usar calculadora na ADA?") is None


def test_choose_prior_keeps_when_new_is_vague():
    """Nova é vaga ('quanto vale a prova?'), prior tem ADA → mantém prior pra contexto."""
    from services.session_manager import _choose_prior
    # Nota: 'prova' agora aciona category=ADA via keyword. Então as duas
    # caem em ADA. Mantém.
    assert _choose_prior("Quando vai ser a ADA?", "quanto tempo eu tenho?") == "Quando vai ser a ADA?"


def test_choose_prior_returns_none_when_candidate_is_none():
    from services.session_manager import _choose_prior
    assert _choose_prior(None, "qualquer pergunta") is None


# --- record_attempt --------------------------------------------------------

def test_record_attempt_persists_all_fields(db, student, llm_classifier):
    plan = plan_interaction(db, student, "Q1")
    record_attempt(
        db, plan,
        question="Q1", answer="A1",
        chunks_used=[
            {"id": "c1", "document_id": "d1", "score": 0.9},
            {"id": "c2", "document_id": "d1", "score": 0.8},
        ],
        was_fallback=False,
        latency_ms=123,
    )
    db.commit()

    att = db.query(QAAttempt).one()
    assert att.attempt_number == 1
    assert att.retrieval_strategy == "default"
    assert att.retrieved_chunks == [
        {"id": "c1", "document_id": "d1", "score": 0.9},
        {"id": "c2", "document_id": "d1", "score": 0.8},
    ]
    assert att.latency_ms == 123


def test_record_attempt_refuses_ack_plan(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()
    p2 = plan_interaction(db, student, "obrigado")

    with pytest.raises(ValueError):
        record_attempt(db, p2, question="obrigado", answer="x")


# --- sessão vencida --------------------------------------------------------

def test_stale_session_is_abandoned(db, student, llm_classifier):
    p1 = plan_interaction(db, student, "Q1")
    record_attempt(db, p1, question="Q1", answer="A1")
    db.commit()

    old = db.query(QASession).first()
    old.opened_at = utcnow() - SESSION_IDLE_TIMEOUT - timedelta(minutes=1)
    db.query(QAAttempt).first().created_at = old.opened_at
    db.commit()

    p2 = plan_interaction(db, student, "Q2")
    db.commit()

    sessions = db.query(QASession).order_by(QASession.opened_at).all()
    assert sessions[0].status == "abandoned"
    assert sessions[1].status == "open"
    assert p2.attempt_number == 1


# --- close_as_* ------------------------------------------------------------

def test_close_as_resolved_stores_poll_id(db, student, llm_classifier):
    p = plan_interaction(db, student, "Q")
    record_attempt(db, p, question="Q", answer="A")
    db.commit()

    sess = db.query(QASession).first()
    close_as_resolved(db, sess, poll_id="POLL-1")
    db.commit()

    assert sess.status == "resolved"
    assert sess.closing_poll_id == "POLL-1"


def test_close_as_escalated_stores_poll_id(db, student, llm_classifier):
    p = plan_interaction(db, student, "Q")
    record_attempt(db, p, question="Q", answer="A")
    db.commit()

    sess = db.query(QASession).first()
    close_as_escalated(db, sess, poll_id="POLL-E")
    db.commit()

    assert sess.status == "escalated"
    assert sess.closing_poll_id == "POLL-E"


def test_close_as_abandoned_marks_timeout(db, student, llm_classifier):
    p = plan_interaction(db, student, "Q")
    record_attempt(db, p, question="Q", answer="A")
    db.commit()

    sess = db.query(QASession).first()
    close_as_abandoned(db, sess)
    db.commit()

    att = db.query(QAAttempt).first()
    assert sess.status == "abandoned"
    assert att.feedback_signal == "timeout"


# --- poll mapping + apply_poll_vote ----------------------------------------

def test_feedback_from_option_matches_labels():
    for label, feedback in POLL_OPTIONS:
        assert feedback_from_option(label) == feedback
        assert feedback_from_option(label.upper()) == feedback


def test_feedback_from_option_unknown_returns_none():
    assert feedback_from_option("outra coisa") is None
    assert feedback_from_option("") is None


def test_apply_poll_vote_updates_session(db, student, llm_classifier):
    p = plan_interaction(db, student, "Q")
    record_attempt(db, p, question="Q", answer="A")
    sess = db.query(QASession).first()
    close_as_resolved(db, sess, poll_id="POLL-XYZ")
    db.commit()

    updated = apply_poll_vote(db, "POLL-XYZ", "resolved_fully")
    db.commit()

    assert updated is not None
    assert updated.id == sess.id
    db.refresh(sess)
    assert sess.closing_feedback == "resolved_fully"


def test_apply_poll_vote_missing_session_returns_none(db):
    assert apply_poll_vote(db, "POLL-GHOST", "resolved_fully") is None
