"""Fixtures globais dos testes.

Configura variáveis de ambiente mínimas antes de importar `config` para
que os testes rodem sem depender de um `.env` válido nem de serviços
externos (Postgres, Mongo, Ollama, Chroma).
"""
import os
import sys
from pathlib import Path

# Garante que o diretório `backend/` esteja no sys.path para que os imports
# `from config import ...`, `from database import ...` etc. funcionem.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Defaults sensatos para testes. Sobrescrevem qualquer .env lido.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_LLM_MODEL", "mistral-test")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "embed-test")
os.environ.setdefault("CHROMA_PERSIST_PATH", "./chroma_test")
os.environ.setdefault("CHUNK_SIZE", "500")
os.environ.setdefault("CHUNK_OVERLAP", "50")
os.environ.setdefault("SIMILARITY_THRESHOLD", "0.4")
os.environ.setdefault("MAX_CHUNKS_RETRIEVED", "5")
os.environ.setdefault("API_SECRET_KEY", "test-secret")
os.environ.setdefault("API_PORT", "8000")
os.environ.setdefault("EVOLUTION_API_URL", "http://localhost:8080")
os.environ.setdefault("EVOLUTION_API_KEY", "test-key")
os.environ.setdefault("EVOLUTION_INSTANCE", "test-instance")
os.environ.setdefault("EVOLUTION_MONGO_URI", "mongodb://localhost:27017")
os.environ.setdefault("EVOLUTION_MONGO_DB", "evolution-test")

import pytest  # noqa: E402


@pytest.fixture
def mock_db():
    """DB mock genérico para rotas — `query().filter().first()` etc."""
    from unittest.mock import MagicMock
    db = MagicMock()
    return db


@pytest.fixture
def api_headers():
    """Header com API key válida para rotas protegidas."""
    return {"X-API-Key": "test-secret"}
