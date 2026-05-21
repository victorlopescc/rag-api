"""Gera embeddings via LiteLLM (provider definido em ``settings.embed_model``).

Padrão atual: ``gemini/gemini-embedding-001`` com ``dimensions=768``
(o modelo suporta Matryoshka shrinking — manda 3072 nativo, devolve
projeção 768 mantendo qualidade próxima). Free tier do Gemini é 1500
RPM em embeddings, folgado pra ingestão de docs típicos.

Defensivo contra:
- chunks vazios / só whitespace (a API rejeita)
- caracteres de controle (NUL, etc.) que vêm de PDFs mal extraídos
- erros transitórios (retry com backoff)
- chunks muito longos (truncamos pelo limite de chars do modelo)

Atenção: trocar o modelo de embedding invalida vetores já indexados no
ChromaDB. Espaços semânticos de modelos diferentes não são compatíveis,
então uma migração de embed_model exige re-indexar todos os documentos.
"""
import logging
import time

import litellm
from litellm.exceptions import APIError, RateLimitError, Timeout

from config import settings

logger = logging.getLogger(__name__)

# Limite defensivo de tamanho. ``gemini-embedding-001`` aceita até 2048
# tokens (~8k chars). 7000 chars cobre qualquer chunk razoável e evita
# erro por overflow no provider.
_MAX_CHARS = 7_000

# Dimensão alvo do vetor (Matryoshka shrinking do gemini-embedding-001).
# 768 é um sweet spot histórico (era a dimensão do nomic-embed-text), e
# o Chroma armazena/calcula cosseno em ~25% do custo de 3072 com perda
# de qualidade desprezível em benchmarks.
_EMBED_DIM = 768

_RETRIES = 2
_BACKOFF_SECONDS = 1.5


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
    """Gera o embedding de um único texto.

    Retorna lista de floats (``_EMBED_DIM`` dimensões).

    Levanta ``ValueError`` se o texto ficar vazio depois de sanitizar.
    Faz retry em erros transitórios antes de propagar.
    """
    sanitized = _sanitize(text)
    if not sanitized:
        raise ValueError("Texto vazio após sanitização — não dá pra gerar embedding.")

    last_err: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            response = litellm.embedding(
                model=settings.embed_model,
                input=[sanitized],
                dimensions=_EMBED_DIM,
                api_key=settings.gemini_api_key,
            )
            return response["data"][0]["embedding"]
        except (APIError, RateLimitError, Timeout) as e:
            last_err = e
            logger.warning(
                "Embed erro (tentativa %d/%d): %s: %s",
                attempt + 1, _RETRIES + 1, type(e).__name__, e,
            )
            if attempt < _RETRIES:
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise
    # Pra fazer o type-checker feliz; nunca chega aqui.
    raise last_err if last_err else RuntimeError("embed_text: estado impossível")


# Tamanho do batch pra ``embed_batch``. O Gemini aceita até 100 inputs
# por chamada; usamos 50 pra balancear throughput vs blast radius (se
# 1 chunk com problema fizer o batch inteiro falhar, perdemos só 50).
_BATCH_SIZE = 50


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Gera embeddings em lotes pra reduzir round-trips.

    Antes (Ollama local), cada chamada era ~50ms na GPU local. Com API
    externa, cada round-trip custa 200-500ms — processar 1000 chunks
    sequenciais ficou inviável (~5 min). Batching de 50 reduz isso a
    ~20 chamadas, levando ~10s no total.

    Em caso de erro num batch, cai pra single-call em cada chunk daquele
    batch — preserva a granularidade de log do código antigo (essencial
    pra identificar qual chunk corrompido derrubou a ingestão do PPC).
    """
    out: list[list[float]] = []
    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        # Sanitiza ANTES de mandar. Itens vazios viram None temporário —
        # vamos preencher a posição depois pra manter ordem com chunks_text.
        sanitized: list[str | None] = []
        for t in batch:
            s = _sanitize(t)
            sanitized.append(s if s else None)

        non_empty_idx = [i for i, s in enumerate(sanitized) if s]
        if not non_empty_idx:
            # Lote inteiro vazio — propaga erro com índice global.
            raise ValueError(
                f"Lote {batch_start}-{batch_start + len(batch)} ficou vazio após sanitização."
            )

        non_empty_texts = [sanitized[i] for i in non_empty_idx]

        try:
            response = litellm.embedding(
                model=settings.embed_model,
                input=non_empty_texts,
                dimensions=_EMBED_DIM,
                api_key=settings.gemini_api_key,
            )
            embeddings = [d["embedding"] for d in response["data"]]
            # Reordena resultados pra alinhar com a ordem do batch original.
            batch_out: list[list[float] | None] = [None] * len(batch)
            for local_i, emb in zip(non_empty_idx, embeddings):
                batch_out[local_i] = emb
            # Itens que estavam vazios não deveriam chegar aqui — o ingestor
            # filtra antes — mas se chegarem, fail-fast com mensagem clara.
            for local_i, emb in enumerate(batch_out):
                if emb is None:
                    global_i = batch_start + local_i
                    raise RuntimeError(
                        f"Chunk {global_i + 1}/{len(texts)} vazio chegou no embed_batch."
                    )
                out.append(emb)
        except Exception as batch_err:
            # Batch inteiro falhou (timeout, item corrompido, etc.).
            # Cai pra single-call pra (a) salvar o que ainda dá e (b)
            # identificar precisamente qual chunk causou o problema.
            logger.warning(
                "Lote %d-%d falhou: %s. Tentando chunk por chunk...",
                batch_start, batch_start + len(batch), batch_err,
            )
            for local_i, t in enumerate(batch):
                global_i = batch_start + local_i
                try:
                    out.append(embed_text(t))
                except Exception as e:
                    preview = (t or "").strip().replace("\n", " ")[:120]
                    logger.error(
                        "Falha no embedding do chunk %d/%d (%d chars): %s | preview: %r",
                        global_i + 1, len(texts), len(t or ""), e, preview,
                    )
                    raise RuntimeError(
                        f"Embedding falhou no chunk {global_i + 1}/{len(texts)}: {e}"
                    ) from e
    return out
