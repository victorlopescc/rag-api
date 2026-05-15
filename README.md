# RAG — Coordenação de Ciência da Computação

Assistente via WhatsApp para alunos do curso de Ciência da Computação da
PUC Minas. Responde perguntas sobre regulamentos, grade curricular,
TCC e correlatos usando **RAG (Retrieval-Augmented Generation)** com
LLM local — sem custo de API e sem dados saindo da infraestrutura da
universidade.

Quando o bot não dá conta da dúvida, o caso é escalado para um
coordenador humano via portal web, com possibilidade de **conversa ao
vivo** (mensagens do WhatsApp passam direto pelo painel, sem o bot
intermediar).

---

## Arquitetura

```
        WhatsApp                                 ┌────────────┐
           │                                     │   Ollama   │
           ▼                                     │ (LLM local)│
    ┌─────────────┐     ┌──────────┐             └──────┬─────┘
    │  Evolution  │◀───▶│ Mongo +  │                    │
    │    API      │     │  Redis   │             ┌──────▼─────┐
    └──────┬──────┘     └──────────┘             │  ChromaDB  │
           │  webhook (POST /webhook)            │ (denso)    │
           ▼                                     └──────▲─────┘
    ┌──────────────┐                                    │
    │   FastAPI    │──── BM25 in-memory ────────────────┤
    │   backend    │──── Cross-encoder reranker ────────┘
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  PostgreSQL  │  alunos, sessões, escalações, thread messages
    └──────────────┘
```

Frontend separado em [`rag-portal`](https://github.com/) — React 19 +
Mantine 9, consome a API.

Todos os serviços rodam localmente — nenhuma dependência paga.

---

## Componentes do retrieval

1. **Busca densa** — embeddings `nomic-embed-text` (Ollama) indexados
   no ChromaDB, distância de cosseno.
2. **Busca lexical BM25** — `rank-bm25` em memória, reconstruído a
   cada ingestão.
3. **Sobreposição lexical crua** — contagem de tokens distintos em
   comum, complementa BM25 em corpora pequenos onde o IDF satura.
4. **Fusão por Reciprocal Rank Fusion (RRF)** — combina os rankings
   acima sem necessidade de calibrar pesos.
5. **Reranker cross-encoder** — `mmarco-mMiniLMv2-L12-H384-v1` da
   `sentence-transformers`, rodando em CPU; reordena os top-50 do RRF
   e devolve os 10 melhores ao prompt.

---

## Pré-requisitos

| Ferramenta | Versão mínima | Link |
|---|---|---|
| Docker + Docker Compose | 24+ | https://docs.docker.com/get-docker |
| Python | 3.11+ | https://python.org |
| Ollama | 0.1+ | https://ollama.ai |

**Hardware recomendado:** 16 GB de RAM (Qwen2.5 7B Instruct ocupa
~5 GB residentes durante a inferência; ChromaDB + Postgres + Mongo +
Redis + Evolution + reranker somam mais ~3 GB).

**Para expor o webhook do WhatsApp durante o desenvolvimento local**,
use um túnel público (ngrok, cloudflared, etc.) apontando para
`http://localhost:8000/webhook`.

---

## Setup local

### 1. Suba a infra com Docker

```bash
cp infra/.env.example infra/.env
# edite infra/.env preenchendo POSTGRES_PASSWORD, API_SECRET_KEY e
# EVOLUTION_API_KEY com valores aleatórios fortes.

docker compose -f infra/docker-compose.yml up -d
```

Sobe PostgreSQL, MongoDB, Redis, ChromaDB e Evolution API.

### 2. Crie o ambiente Python e instale deps

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 3. Baixe os modelos do Ollama

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M     # ~4.4 GB
ollama pull nomic-embed-text               # ~270 MB
```

### 4. Suba o backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Confira a saúde

```bash
curl http://localhost:8000/health
```

---

## Configuração da Evolution API

Após subir os containers:

1. Acesse o painel da Evolution em `http://localhost:8080`
2. Crie uma instância (ex.: `coordenacao`) e escaneie o QR code
3. Configure o webhook apontando para o backend:

```bash
bash infra/configure_evolution_webhook.sh
```

O script registra o webhook nos eventos `MESSAGES_UPSERT`,
`MESSAGES_UPDATE`, `CONTACTS_UPSERT` e `CONTACTS_UPDATE`.

---

## Variáveis de ambiente

Veja `infra/.env.example` para a lista completa. As principais:

| Variável | Descrição |
|---|---|
| `OLLAMA_LLM_MODEL` | Modelo do Ollama (default: `qwen2.5:7b-instruct-q4_K_M`) |
| `OLLAMA_EMBED_MODEL` | Modelo de embedding (default: `nomic-embed-text`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Parâmetros do chunker (500 / 80) |
| `SIMILARITY_THRESHOLD` | Score mínimo no modo sem reranker (0.20) |
| `MAX_CHUNKS_RETRIEVED` | Top-K final passado ao LLM (10) |
| `ENABLE_RERANKER` | Liga/desliga o cross-encoder (default: true) |
| `RERANKER_MIN_SCORE` | Probabilidade mínima após sigmoid (0.05) |
| `ENABLE_BM25` | Liga/desliga o BM25 (default: true) |
| `API_SECRET_KEY` | Chave X-API-Key para o painel admin |

---

## Cadastro de alunos

Os alunos se cadastram em `https://<seu-domínio>/registro` (página
do frontend), informando nome, matrícula e telefone. O backend dispara
a mensagem de boas-vindas pelo WhatsApp via Evolution API.

---

## Scripts úteis

```bash
# Re-indexa todos os documentos a partir do texto reconstruído dos
# chunks atuais (útil quando você muda o chunker).
cd backend && python -m scripts.reindex

# Re-indexa a partir dos arquivos originais (PDF/DOCX), resultado
# mais limpo quando você tem os arquivos.
python -m scripts.reindex_from_files \
    ~/Downloads/PPC.pdf \
    ~/Downloads/Regulamento_TCC.pdf \
    ~/Downloads/Resolucao_ADA.pdf

# Resetar TODA a infraestrutura (containers + volumes + chunks).
bash infra/reset_all.sh --yes
```

---

## Estrutura de pastas

```
backend/
├── auth.py                  # Validação de X-API-Key
├── config.py                # Pydantic settings (lê infra/.env)
├── database.py              # Models SQLAlchemy
├── main.py                  # FastAPI app + CORS + warmup
├── rag_engine.py            # Orquestra retrieval → LLM → resposta
├── pipeline/
│   ├── acronyms.py          # Expansão e detecção de siglas (ADA, TCC, PPC...)
│   ├── bm25_index.py        # BM25 in-memory + sobreposição lexical crua
│   ├── chunker.py           # Splitter recursivo por separadores
│   ├── embedder.py          # Chamadas ao Ollama embeddings
│   ├── extractor.py         # PDF (PyMuPDF) + DOCX
│   ├── ingestor.py          # Pipeline de ingestão completa
│   ├── llm.py               # Cliente do Ollama generate
│   ├── prompt_builder.py    # Constrói o prompt final
│   ├── reranker.py          # Cross-encoder mmarco-mMiniLMv2 (CPU)
│   ├── retrieval_strategies.py  # 3 estratégias + RRF
│   └── vector_store.py      # Wrapper do ChromaDB
├── routers/
│   ├── admin.py             # Escalações + thread + manutenção
│   ├── admin_analytics.py   # KPIs + relatórios
│   ├── documents.py         # Upload / listar / chunks preview
│   ├── query.py             # POST /query direto (testes)
│   ├── users.py             # Cadastro de alunos
│   └── webhook.py           # Recebe do WhatsApp via Evolution
├── services/
│   ├── dedup.py             # Idempotência por msg_id
│   ├── escalation_service.py # Resumo automático da escalação
│   ├── evolution_client.py  # Wrapper HTTP da Evolution
│   ├── evolution_mongo.py   # Lê messageUpdate pra resolver LID
│   ├── intent_classifier.py # Classifica mensagem (yes / rephrase / ...)
│   ├── lid_resolver.py      # Mapeia LID opaco → phone real
│   ├── maintenance.py       # Cron de sessões e threads stale
│   ├── message_triage.py    # Saudações / trivialidades / comandos
│   ├── session_manager.py   # Ciclo de vida da QASession
│   ├── thread_service.py    # Live thread aluno↔coordenador
│   └── whatsapp.py          # Textos fixos e helpers
├── scripts/                 # Scripts auxiliares (reindex)
└── tests/                   # Suite de testes (pytest)

infra/
├── docker-compose.yml       # PostgreSQL, MongoDB, Redis, ChromaDB, Evolution
├── init.sql                 # Schema completo
├── .env.example
├── reset_all.sh             # Apaga tudo e sobe de novo
└── configure_evolution_webhook.sh
```

---

## Rodando os testes

```bash
cd backend && python -m pytest -q
```

Testes usam SQLite in-memory para os fluxos que envolvem persistência;
Ollama, Evolution API e Chroma são mockados.

---

## Licença

Projeto desenvolvido como Trabalho de Conclusão de Curso de Ciência da
Computação na PUC Minas.
