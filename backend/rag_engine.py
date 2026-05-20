"""
RAG Engine — orquestra a busca semântica e a geração da resposta.

Fluxo:
  pergunta → retrieve híbrido (denso + BM25 + RRF) → filtra por score
           → monta prompt → LLM → resposta
"""

import logging
import time
from dataclasses import dataclass

from pipeline.retrieval import retrieve
from pipeline.prompt_builder import build_prompt, FALLBACK_MESSAGE
from pipeline.llm import generate
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer:      str
    was_fallback: bool
    # Descritores dos chunks usados: [{"id", "document_id", "score"}, ...].
    # O ``id`` é o chroma_id; ``document_id`` pode ser None se o metadata
    # não tiver sido gravado. Mantemos como list[dict] (não list[str]) para
    # que o analytics consiga atribuir uso por documento.
    chunks_used: list[dict]
    latency_ms:  int


def ask(
    question: str,
    category: str | None = None,
    prior_question: str | None = None,
) -> RAGResponse:
    """
    Responde uma pergunta usando RAG.

    Parâmetros:
        question: pergunta do usuário
        category: filtra a busca por categoria de documento (opcional)
        prior_question: pergunta anterior na MESMA sessão (mantém o tópico
                  no retrieval; o prompt do LLM continua vendo só ``question``).
                  Útil pra "Quanto tempo eu tenho pra prova?" depois de
                  "Quando vai ser a ADA?" — sozinha a 2ª pergunta perderia o
                  contexto entre os 970 chunks de PPC.
    """
    start = time.time()

    # Para retrieval, anexa a pergunta anterior pra carregar o tópico.
    # O LLM, mais abaixo, vê só a pergunta atual.
    retrieve_question = (
        f"{prior_question}\n{question}" if prior_question else question
    )
    logger.info(
        f"RAG: '{question[:60]}...' "
        f"(prior_q={'sim' if prior_question else 'não'})"
    )
    chunks = retrieve(retrieve_question, category=category)

    logger.info(f"{len(chunks)} chunks recuperados.")

    # Threshold: depende de quem produziu os scores.
    #   - Reranker ON: ``score`` = probabilidade calibrada do cross-encoder
    #     [0,1]. Usamos ``reranker_min_score`` (default 0.05).
    #   - Reranker OFF: ``score`` = similaridade cosseno + boosts ad-hoc.
    #     Usamos o ``similarity_threshold`` histórico (0.20).
    threshold = (
        settings.reranker_min_score
        if settings.enable_reranker
        else settings.similarity_threshold
    )
    relevant = [c for c in chunks if c["score"] >= threshold]
    logger.info(f"{len(relevant)} chunks acima do threshold ({threshold}).")

    # Limita ao top-K. Manter chunks demais no prompt vira ruído — o LLM
    # pequeno (8B) tende a misturar informações ou pegar a primeira frase
    # parecida em vez do trecho realmente certo. ``max_chunks_retrieved``
    # já é o teto operacional (k=8 por default).
    relevant = sorted(relevant, key=lambda c: c["score"], reverse=True)
    relevant = relevant[: settings.max_chunks_retrieved]

    # Se nenhum chunk for relevante, retorna fallback sem chamar o LLM
    if not relevant:
        logger.warning("Nenhum chunk relevante encontrado — ativando fallback.")
        return RAGResponse(
            answer=FALLBACK_MESSAGE,
            was_fallback=True,
            chunks_used=[],
            latency_ms=int((time.time() - start) * 1000),
        )

    # Monta o prompt com os chunks relevantes
    prompt = build_prompt(question, relevant)

    # Chama o LLM
    logger.info("Chamando o LLM...")
    answer = generate(prompt)

    # Verifica se o LLM mesmo assim indicou que não encontrou a informação
    was_fallback = FALLBACK_MESSAGE.lower() in answer.lower()

    latency = int((time.time() - start) * 1000)
    logger.info(f"Resposta gerada em {latency}ms.")

    return RAGResponse(
        answer=answer,
        was_fallback=was_fallback,
        chunks_used=[
            {
                "id": c["id"],
                "document_id": (c.get("metadata") or {}).get("document_id"),
                "score": c["score"],
            }
            for c in relevant
        ],
        latency_ms=latency,
    )
