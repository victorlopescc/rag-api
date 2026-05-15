-- =============================================
-- RAG System — Schema inicial
-- =============================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documentos enviados pelo coordenador
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx', 'txt')),
    category    TEXT,                          -- ex: 'regulamento', 'grade', 'edital'
    description TEXT,
    file_size   INTEGER,                       -- em bytes
    total_chunks INTEGER DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'indexed', 'error')),
    error_msg   TEXT,
    uploaded_by TEXT DEFAULT 'coordinator',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chunks gerados a partir dos documentos
CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,             -- posição do chunk no doc
    content      TEXT NOT NULL,                -- texto do chunk
    chroma_id    TEXT UNIQUE,                  -- ID correspondente no ChromaDB
    token_count  INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Alunos cadastrados que usam o bot via WhatsApp
CREATE TABLE IF NOT EXISTS students (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lid                 TEXT,                          -- LID do WhatsApp (vinculado após o 1º ACK)
    full_name           TEXT NOT NULL,
    matricula           TEXT NOT NULL,
    phone_number        TEXT NOT NULL UNIQUE,
    pending_welcome_id  TEXT,                          -- key.id da mensagem de boas-vindas (usado p/ resolver LID)
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    data_consent        BOOLEAN NOT NULL DEFAULT TRUE, -- opt-in para uso dos dados no TCC
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Log de perguntas feitas via WhatsApp (auditoria)
CREATE TABLE IF NOT EXISTS query_logs (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_number  TEXT,                        -- número do usuário (anonimizável)
    question      TEXT NOT NULL,
    answer        TEXT,
    chunks_used   TEXT[],                      -- IDs dos chunks usados na resposta
    model_used    TEXT,
    latency_ms    INTEGER,
    was_fallback  BOOLEAN DEFAULT FALSE,       -- TRUE se não encontrou contexto
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================
-- Avaliação do bot (TCC) — sessões, tentativas, escalações
-- =============================================

-- Uma sessão agrupa as tentativas do bot sobre um mesmo tópico.
CREATE TABLE IF NOT EXISTS qa_sessions (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id       UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    status           TEXT NOT NULL DEFAULT 'open'
                     CHECK (status IN ('open', 'resolved', 'abandoned', 'escalated')),
    topic_summary    TEXT,
    opened_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at        TIMESTAMPTZ,
    closing_feedback TEXT CHECK (closing_feedback IN
                     ('resolved_fully', 'resolved_partially', 'not_resolved')),
    closing_poll_id  TEXT
);

-- Cada (pergunta, resposta) dentro de uma sessão.
CREATE TABLE IF NOT EXISTS qa_attempts (
    id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id         UUID NOT NULL REFERENCES qa_sessions(id) ON DELETE CASCADE,
    attempt_number     INTEGER NOT NULL CHECK (attempt_number BETWEEN 1 AND 3),
    question           TEXT NOT NULL,
    answer             TEXT NOT NULL,
    retrieval_strategy TEXT NOT NULL DEFAULT 'default'
                       CHECK (retrieval_strategy IN ('default', 'query_rewrite', 'widen_k')),
    retrieved_chunks   JSONB,
    was_fallback       BOOLEAN DEFAULT FALSE,
    latency_ms         INTEGER,
    feedback_signal    TEXT CHECK (feedback_signal IN
                       ('explicit_yes', 'explicit_no', 'implicit_rephrase',
                        'implicit_new_topic', 'timeout', 'cancelled_by_user')),
    resolved           BOOLEAN,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Escalações para o coordenador, geradas depois da 3ª tentativa falhar.
CREATE TABLE IF NOT EXISTS escalations (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id        UUID NOT NULL UNIQUE REFERENCES qa_sessions(id) ON DELETE CASCADE,
    student_id        UUID NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    summary           TEXT NOT NULL,
    -- Estados:
    --   pending               → criada, coordenador ainda não viu
    --   coordinator_replied   → coordenador mandou UMA resposta (modo legado, sem thread)
    --   resolved_by_bot_later → aluno voltou e o bot resolveu
    --   live                  → conversa ao vivo aberta entre aluno e coordenador
    --   resolved              → thread encerrada pelo coordenador
    --   abandoned             → thread encerrada por timeout ou pedido do aluno
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'coordinator_replied', 'resolved_by_bot_later',
                                        'live', 'resolved', 'abandoned')),
    coordinator_reply TEXT,
    coordinator_label TEXT CHECK (coordinator_label IN
                      ('bot_was_wrong', 'missing_document',
                       'student_misunderstood', 'other')),
    coordinator_notes TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    replied_at        TIMESTAMPTZ,
    -- Ciclo de vida da live thread:
    live_opened_at    TIMESTAMPTZ,
    live_closed_at    TIMESTAMPTZ,
    last_activity_at  TIMESTAMPTZ
);

-- Mensagens trocadas durante uma thread ao vivo aluno↔coordenador.
-- Cada mensagem do aluno (vinda do WhatsApp) e do coordenador (vinda
-- do painel) é registrada aqui em ordem cronológica.
CREATE TABLE IF NOT EXISTS thread_messages (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    escalation_id    UUID NOT NULL REFERENCES escalations(id) ON DELETE CASCADE,
    direction        TEXT NOT NULL CHECK (direction IN ('student', 'coordinator')),
    text             TEXT NOT NULL,
    sent_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    evolution_msg_id TEXT
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_documents_status   ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_phone   ON query_logs(phone_number);
CREATE INDEX IF NOT EXISTS idx_query_logs_date    ON query_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_students_lid       ON students(lid);
CREATE INDEX IF NOT EXISTS idx_students_pending   ON students(pending_welcome_id);
CREATE INDEX IF NOT EXISTS idx_qa_sessions_student     ON qa_sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_qa_sessions_status      ON qa_sessions(status);
CREATE INDEX IF NOT EXISTS idx_qa_sessions_open_only   ON qa_sessions(student_id) WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_qa_sessions_poll_id     ON qa_sessions(closing_poll_id) WHERE closing_poll_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_qa_attempts_session     ON qa_attempts(session_id);
CREATE INDEX IF NOT EXISTS idx_qa_attempts_feedback    ON qa_attempts(feedback_signal);
CREATE INDEX IF NOT EXISTS idx_escalations_status      ON escalations(status);
CREATE INDEX IF NOT EXISTS idx_escalations_student     ON escalations(student_id);
CREATE INDEX IF NOT EXISTS idx_escalations_live_by_student
    ON escalations(student_id) WHERE status = 'live';
CREATE INDEX IF NOT EXISTS idx_thread_messages_escalation
    ON thread_messages(escalation_id, sent_at);

-- Trigger para atualizar updated_at automaticamente
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_students_updated_at
    BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();