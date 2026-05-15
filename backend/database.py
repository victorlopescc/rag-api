import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, Column, String, Integer, Boolean,
    Text, ARRAY, DateTime, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency para injetar sessão do banco nas rotas FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def utcnow():
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    category = Column(String)
    description = Column(Text)
    file_size = Column(Integer)
    total_chunks = Column(Integer, default=0)
    status = Column(String, nullable=False, default="pending")
    error_msg = Column(Text)
    uploaded_by = Column(String, default="coordinator")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Student(Base):
    __tablename__ = "students"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lid = Column(String, nullable=True, index=True)
    full_name = Column(String, nullable=False)
    matricula = Column(String, nullable=False)
    phone_number = Column(String, nullable=False, unique=True)
    pending_welcome_id = Column(String, nullable=True, index=True)
    active = Column(Boolean, default=True)
    data_consent = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chroma_id = Column(String, unique=True)
    token_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_number = Column(String)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    chunks_used = Column(ARRAY(Text))
    model_used = Column(String)
    latency_ms = Column(Integer)
    was_fallback = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# =============================================================================
# Avaliação do bot (thesis data) — sessões de perguntas, tentativas e escalações.
#
# Uma QASession agrupa as tentativas sobre um mesmo tópico. Fecha quando o
# aluno confirma resolução, quando o bot esgota as 3 tentativas (escalação)
# ou por timeout (6h de ociosidade / fim do dia).
#
# Cada mensagem do aluno que dispara uma resposta do RAG vira uma QAAttempt.
# Se a 3ª tentativa falhar, uma Escalation é criada para o coordenador.
# =============================================================================


class QASession(Base):
    __tablename__ = "qa_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True),
                        ForeignKey("students.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    # open | resolved | abandoned | escalated
    status = Column(String, nullable=False, default="open", index=True)
    topic_summary = Column(Text)  # preenchido por LLM ao fechar (milestone 3)
    opened_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    closed_at = Column(DateTime(timezone=True))
    # Voto do aluno na enquete enviada no fechamento:
    # resolved_fully | resolved_partially | not_resolved | null (sem voto)
    closing_feedback = Column(String)
    # key.id da poll enviada (para casar votos do webhook).
    closing_poll_id = Column(String, index=True)

    attempts = relationship(
        "QAAttempt", back_populates="session",
        cascade="all, delete-orphan", order_by="QAAttempt.attempt_number",
    )
    escalation = relationship(
        "Escalation", back_populates="session",
        uselist=False, cascade="all, delete-orphan",
    )


class QAAttempt(Base):
    __tablename__ = "qa_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True),
                        ForeignKey("qa_sessions.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    attempt_number = Column(Integer, nullable=False)  # 1..3
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    # default | query_rewrite | widen_k  (milestone 2 usa as 3; milestone 1 só default)
    retrieval_strategy = Column(String, nullable=False, default="default")
    retrieved_chunks = Column(JSONB)  # [{id, score, document_id}, ...]
    was_fallback = Column(Boolean, default=False)
    latency_ms = Column(Integer)
    # explicit_yes | explicit_no | implicit_rephrase | implicit_new_topic | timeout | null
    feedback_signal = Column(String, index=True)
    resolved = Column(Boolean)  # null = desconhecido
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    session = relationship("QASession", back_populates="attempts")


class Escalation(Base):
    __tablename__ = "escalations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True),
                        ForeignKey("qa_sessions.id", ondelete="CASCADE"),
                        nullable=False, unique=True)
    student_id = Column(UUID(as_uuid=True),
                        ForeignKey("students.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    summary = Column(Text, nullable=False)  # LLM-generated
    # Estados:
    #   pending               → criada, coordenador ainda não viu
    #   coordinator_replied   → modo legado: 1 resposta única (sem thread)
    #   resolved_by_bot_later → aluno voltou e o bot resolveu
    #   live                  → conversa ao vivo aberta entre aluno e coordenador
    #   resolved              → thread encerrada pelo coordenador
    #   abandoned             → thread encerrada por timeout / aluno
    status = Column(String, nullable=False, default="pending", index=True)
    coordinator_reply = Column(Text)
    # Rótulos do coordenador para análise da tese:
    # bot_was_wrong | missing_document | student_misunderstood | other
    coordinator_label = Column(String)
    coordinator_notes = Column(Text)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    replied_at = Column(DateTime(timezone=True))
    # Ciclo de vida da live thread (todos NULL no fluxo legado de
    # "1 resposta única"; preenchidos quando a thread é iniciada).
    live_opened_at = Column(DateTime(timezone=True))
    live_closed_at = Column(DateTime(timezone=True))
    last_activity_at = Column(DateTime(timezone=True))

    session = relationship("QASession", back_populates="escalation")
    thread_messages = relationship(
        "ThreadMessage",
        back_populates="escalation",
        cascade="all, delete-orphan",
        order_by="ThreadMessage.sent_at",
    )


class ThreadMessage(Base):
    """Uma mensagem trocada durante uma live thread aluno↔coordenador.

    ``direction = 'student'``: aluno mandou pelo WhatsApp.
    ``direction = 'coordinator'``: coordenador mandou pelo painel.
    """
    __tablename__ = "thread_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    escalation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("escalations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    direction = Column(String, nullable=False)  # 'student' | 'coordinator'
    text = Column(Text, nullable=False)
    sent_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    # ID retornado pelo Evolution quando a mensagem foi enviada ao
    # WhatsApp (só preenchido em mensagens 'coordinator' que vão pro
    # aluno). Útil pra correlação com ACKs.
    evolution_msg_id = Column(String)

    escalation = relationship("Escalation", back_populates="thread_messages")
