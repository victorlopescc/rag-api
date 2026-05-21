"""LLM-as-reranker — usa o próprio Gemini 3 Flash pra ranquear chunks por
relevância à query.

Por que existe (e por que NÃO é mais cross-encoder local)
---------------------------------------------------------
Antes: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` rodando local via
``sentence-transformers``. Funcionava razoavelmente em texto natural,
mas tinha 2 problemas:

1. **Formato tabular confunde**: chunks tipo
   ``- Per: 2; Disciplina: AEDs2; CH: 120; ...`` recebiam score baixo
   porque o modelo foi treinado em prosa, não em campos chave-valor.
   Vimos casos onde o chunk LITERALMENTE correto era cortado.
2. **500MB de RAM ocupados pra cada worker uvicorn** + ~120MB no disco
   pelo modelo + dependência pesada (``torch``, ``sentence-transformers``).

Hoje: o Gemini 3 Flash já é o LLM principal do RAG. Ele é muito mais
capaz de avaliar relevância semântica que o mmarco, entende formato
tabular nativamente, e custa ~$0.001 por chamada (alguns milésimos de
real). Adicionar uma chamada extra pra reranking ficou trivial.

Como funciona
-------------
1. Recebe os top-N chunks do retrieval híbrido (N ~= ``reranker_input_k``).
2. Manda pro Gemini um prompt: "Aqui está uma pergunta e N trechos;
   retorne JSON com os índices ordenados por relevância, com score 0-10".
3. Parseia o JSON, aplica ``min_score``, retorna top_k.
4. Em qualquer falha (JSON inválido, timeout, etc.), faz fallback
   gracioso devolvendo os chunks na ordem original — pra que o pipeline
   nunca derrube uma query por causa do reranker.

Custo & latência
----------------
Por query: ~3-5k tokens de input + ~300 de output. Em Gemini 3 Flash:
~$0.001/call e ~0.8-1.5s de latência adicional. Aceitável pra UX de
WhatsApp.
"""
from __future__ import annotations

import json
import logging
import re

import litellm

from config import settings

logger = logging.getLogger(__name__)


# Limita o tamanho do preview de cada chunk no prompt. Mais que isso
# infla o prompt sem proporcionalmente melhorar a decisão do reranker
# (a parte mais relevante do chunk geralmente é o início ou centro,
# raramente os últimos 300 chars).
_CHUNK_PREVIEW_CHARS = 500


_PROMPT_TEMPLATE = """Avalie a relevância de cada trecho para responder a pergunta.

PERGUNTA:
{query}

TRECHOS:
{chunks_block}

Atribua a CADA trecho uma nota de 0 a 10:
- 0-2: irrelevante
- 3-5: pouco relevante (toca no assunto mas não responde)
- 6-8: relevante (contém a informação)
- 9-10: altamente relevante (contém a resposta direta)

Não invente, não infira — avalie só pela presença de informação útil.

Responda APENAS um JSON neste formato exato, sem comentários ou markdown:
{{"ranking": [{{"id": 0, "score": 9}}, {{"id": 3, "score": 7}}, ...]}}

Inclua TODOS os trechos no ranking. Ordene do mais relevante ao menos."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_prompt(query: str, chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        preview = (c.get("content") or "")[:_CHUNK_PREVIEW_CHARS]
        # Sem newlines internos pra preservar formato do prompt;
        # substituímos por " | " pra manter legibilidade.
        preview = preview.replace("\n", " | ")
        lines.append(f"[{i}] {preview}")
    return _PROMPT_TEMPLATE.format(query=query, chunks_block="\n\n".join(lines))


def _parse_ranking(raw: str, expected_n: int) -> list[tuple[int, float]] | None:
    """Extrai ``[(id, score_0_to_1), ...]`` do JSON do LLM. None em erro.

    Tolerante a ruído ao redor (markdown, prefixos) — pega o primeiro
    bloco ``{...}`` válido.
    """
    m = _JSON_BLOCK_RE.search(raw or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    ranking = obj.get("ranking")
    if not isinstance(ranking, list):
        return None

    out: list[tuple[int, float]] = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        idx = item.get("id")
        score = item.get("score")
        if not isinstance(idx, int) or idx < 0 or idx >= expected_n:
            continue
        try:
            s = float(score)
        except (TypeError, ValueError):
            continue
        # Normaliza score 0-10 → 0-1. Clamp defensivo.
        s_norm = max(0.0, min(10.0, s)) / 10.0
        out.append((idx, s_norm))
    return out if out else None


def rerank(
    query: str,
    chunks: list[dict],
    *,
    top_k: int,
    min_score: float = 0.0,
) -> list[dict]:
    """Reordena ``chunks`` por relevância à ``query`` e devolve os top_k.

    Cada chunk de saída tem dois campos novos:
      - ``rerank_score``: probabilidade [0,1] de relevância (do LLM).
      - ``embed_score``:  o score original do RRF (preservado pra debug).

    O campo ``score`` passa a refletir ``rerank_score``.

    ``min_score`` filtra chunks abaixo desse limiar antes de cortar em top_k.
    Default 0.0 = sem filtro (decisão fica com o caller).

    Em caso de falha (LLM timeout, JSON inválido, etc.), devolve os
    chunks na ordem original (limitado a top_k) — fail-soft pra não
    quebrar a query do aluno.
    """
    if not chunks:
        return []

    # Importa aqui pra quebrar dependência circular potencial e respeitar
    # o helper que sabe disable thinking em Gemini 3.
    from pipeline.llm import _extra_kwargs_for, _temperature_for

    prompt = _build_prompt(query, chunks)
    try:
        response = litellm.completion(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=_temperature_for(settings.llm_model),
            max_tokens=512,
            api_key=settings.gemini_api_key,
            timeout=30.0,
            **_extra_kwargs_for(settings.llm_model),
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"LLM rerank falhou ({type(e).__name__}: {e}); usando ordem original")
        return _fallback_order(chunks, top_k)

    ranking = _parse_ranking(raw, expected_n=len(chunks))
    if ranking is None:
        logger.warning(f"LLM rerank devolveu JSON inválido; usando ordem original. Raw: {raw[:200]!r}")
        return _fallback_order(chunks, top_k)

    enriched: list[dict] = []
    seen_ids: set[int] = set()
    for idx, score in ranking:
        if idx in seen_ids:
            continue
        seen_ids.add(idx)
        if score < min_score:
            continue
        c = chunks[idx]
        enriched.append({
            **c,
            "embed_score": c.get("score"),
            "rerank_score": score,
            "score": score,
        })

    # Garantia defensiva: se o LLM ignorou alguns chunks, anexa eles ao
    # final com score 0. Evita perder candidatos só porque o reranker
    # foi preguiçoso. Eles ficam atrás dos chunks ranqueados.
    if len(seen_ids) < len(chunks):
        for i, c in enumerate(chunks):
            if i not in seen_ids:
                enriched.append({
                    **c,
                    "embed_score": c.get("score"),
                    "rerank_score": 0.0,
                    "score": 0.0,
                })

    enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
    return enriched[:top_k]


def _fallback_order(chunks: list[dict], top_k: int) -> list[dict]:
    """Mantém a ordem do retrieval híbrido quando o LLM falha. Preserva
    o score original e marca ``rerank_score=None`` pra deixar claro no
    log/debug que o reranker foi pulado nesta query."""
    out = []
    for c in chunks[:top_k]:
        out.append({
            **c,
            "embed_score": c.get("score"),
            "rerank_score": None,
        })
    return out


def warmup() -> None:
    """Antes carregava o cross-encoder local. Agora é no-op — não há
    modelo pra inicializar. Mantemos a função pra preservar o contrato
    do ``main.py`` (chamada de startup) sem precisar mexer no caller."""
    return
