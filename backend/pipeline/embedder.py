"""
Gera embeddings chamando o Ollama local.
Mantém uma instância singleton do cliente HTTP para reusar conexões.

Defensivo contra:
- chunks vazios / só whitespace (Ollama responde 500)
- caracteres de controle (NUL, etc.) que vêm de PDFs mal extraídos
- 500s transitórios do Ollama (retry com backoff)
- chunks muito longos (truncamos pelo limite de chars do modelo)
"""

import logging
import time

import httpx

from config import settings

logger = logging.getLogger(__name__)

_client: httpx.Client | None = None

# Limite defensivo de tamanho. nomic-embed-text aceita ~8192 tokens; ~30k
# caracteres é folgado pra qualquer chunk razoável e evita 500 por overflow.
_MAX_CHARS = 30_000

_RETRIES = 2
_BACKOFF_SECONDS = 1.5


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=60.0,
        )
    return _client


def _sanitize(text: str) -> str:
    """Remove caracteres de controle e espaços redundantes."""
    if not text:
        return ""
    # Tira NUL e demais control chars (mantém \n, \t).
    cleaned = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or ord(ch) >= 32
    )
    cleaned = cleaned.strip()
    if len(cleaned) > _MAX_CHARS:
        logger.warning(
            "Chunk com %d chars excede limite (%d); truncando.",
            len(cleaned), _MAX_CHARS,
        )
        cleaned = cleaned[:_MAX_CHARS]
    return cleaned


def embed_text(text: str) -> list[float]:
    """
    Gera o embedding de um único texto.
    Retorna lista de floats (vetor de ~768 dimensões).

    Levanta ``ValueError`` se o texto ficar vazio depois de sanitizar.
    Faz retry em 500/conexão antes de propagar.
    """
    sanitized = _sanitize(text)
    if not sanitized:
        raise ValueError("Texto vazio após sanitização — não dá pra gerar embedding.")

    client = _get_client()
    last_err: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            response = client.post(
                "/api/embeddings",
                json={
                    "model": settings.ollama_embed_model,
                    "prompt": sanitized,
                    # Por padrão Ollama usa num_ctx=2048 mesmo em modelos
                    # de embedding que aguentam mais. nomic-embed-text
                    # suporta 8192 — explicitamos pra evitar
                    # "input length exceeds the context length".
                    "options": {"num_ctx": 8192},
                },
            )
            if response.status_code >= 500:
                # Ollama 500 — loga o corpo (geralmente diz o motivo) e retry.
                body_preview = (response.text or "")[:300]
                logger.warning(
                    "Ollama %d em embed (tentativa %d/%d). Body: %s",
                    response.status_code, attempt + 1, _RETRIES + 1, body_preview,
                )
                response.raise_for_status()
            response.raise_for_status()
            return response.json()["embedding"]
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            if attempt < _RETRIES:
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise
    # Pra fazer o type-checker feliz; nunca chega aqui.
    raise last_err if last_err else RuntimeError("embed_text: estado impossível")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos.

    Processa um a um (Ollama não tem endpoint batch nativo). Em caso de
    erro, a exceção propagada inclui o índice do chunk problemático e
    um preview do conteúdo, pra facilitar debug.
    """
    out: list[list[float]] = []
    for i, t in enumerate(texts):
        try:
            out.append(embed_text(t))
        except Exception as e:
            preview = (t or "").strip().replace("\n", " ")[:120]
            logger.error(
                "Falha no embedding do chunk %d/%d (%d chars): %s | preview: %r",
                i + 1, len(texts), len(t or ""), e, preview,
            )
            raise RuntimeError(
                f"Embedding falhou no chunk {i + 1}/{len(texts)}: {e}"
            ) from e
    return out