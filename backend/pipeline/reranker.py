"""Cross-encoder reranker — reordena os top-K do retrieval por relevância
semântica fina (query↔chunk).

Por que existe
--------------
Embeddings densos (nomic-embed-text) ranqueiam por similaridade média do
significado. O problema: cabeçalhos e intros dos documentos têm
similaridade alta com QUASE qualquer query do mesmo domínio, soterrando
chunks que LITERALMENTE respondem a pergunta. Um cross-encoder olha
query+chunk juntos e atribui um score calibrado de relevância — sem os
patterns hardcoded que tínhamos antes (boost lexical, expansões
conceituais), o que torna o sistema doc-agnóstico.

Privacidade
-----------
O modelo baixa do HuggingFace UMA vez na instalação e roda 100% offline
depois. Sem chamada de API externa em runtime. Cabe a constraint de
privacidade do sistema.

Custo
-----
mmarco-mMiniLMv2-L12-H384-v1: ~120MB no disco, ~500MB RAM em runtime,
~80-150ms pra ranquear 25 pares em CPU. No nosso budget de 16GB com
qwen14b (~10-11GB) sobra folga.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

# CRÍTICO: precisa vir ANTES de qualquer import que carregue torch.
# Sem isso, ``torch`` importa o runtime CUDA mesmo configurando
# ``device="cpu"`` no CrossEncoder, e o runtime sozinho reserva
# 200-500MB de VRAM só do contexto CUDA. Em GPUs apertadas (ex.: RTX
# 4050 com 6GB e Windows + Chrome consumindo ~1GB de baseline), esses
# 500MB são suficientes pra fazer o Ollama crashar com "llama runner
# process has terminated: CUDA error" ao carregar o LLM.
# Setar a env var pra string vazia diz ao torch "não vejo nenhuma
# GPU" — torch não inicializa CUDA, não reserva VRAM, e o reranker
# roda em CPU (que é o que queríamos de qualquer jeito).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

logger = logging.getLogger(__name__)

# Carregamento do modelo é caro (~2-5s no primeiro uso). Mantemos
# um singleton com lock pra inicialização thread-safe — uvicorn com
# múltiplos workers só baixa/carrega uma vez por worker.
_model: Any = None
_model_lock = threading.Lock()


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:  # double-checked locking
            return _model
        # Import preguiçoso pra que o módulo seja importável mesmo sem
        # sentence-transformers instalado (testes que mockam não pagam
        # a dep). Se não tem, propaga o ImportError com contexto.
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers não está instalado. "
                "Rode `pip install sentence-transformers` ou desligue o "
                "reranker via ENABLE_RERANKER=false no .env."
            ) from e

        from config import settings
        logger.info(
            f"Carregando cross-encoder: {settings.reranker_model} "
            f"(device={settings.reranker_device})"
        )
        # CRÍTICO: forçamos device explicitamente. Default do
        # sentence-transformers é CUDA se disponível, o que faz o
        # reranker BRIGAR por VRAM com o qwen14b já carregado no Ollama
        # e gerar "CUDA error" / OOM no LLM. Em CPU, o reranker roda
        # em ~80-150ms por batch de 25 (aceitável) e não toca a GPU.
        _model = CrossEncoder(
            settings.reranker_model,
            max_length=512,
            device=settings.reranker_device,
        )
        logger.info("Cross-encoder carregado.")
        return _model


def _sigmoid(x: float) -> float:
    """Logit → probabilidade [0,1]. Trabalhamos em probabilidades pra
    que o threshold downstream tenha sentido absoluto (0.5 = "modelo
    está em dúvida"; <0.05 = "claramente irrelevante")."""
    # Clamp pra evitar overflow em logits muito grandes.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def rerank(
    query: str,
    chunks: list[dict],
    *,
    top_k: int,
    min_score: float = 0.0,
) -> list[dict]:
    """Reordena ``chunks`` por relevância à ``query`` e devolve os top_k.

    Cada chunk de saída tem dois campos novos preservados:
      - ``rerank_score``: probabilidade [0,1] da relevância (sigmoid do logit).
      - ``embed_score``:  o score original do embedding (preservado pra debug).

    O campo ``score`` passa a refletir o ``rerank_score``, pra que o
    downstream (filtros e ordenação) opere sobre a sinal mais forte.

    ``min_score`` filtra chunks abaixo desse limiar antes de cortar em top_k.
    Default 0.0 = sem filtro (decisão fica com o caller).
    """
    if not chunks:
        return []
    model = _load_model()
    pairs = [(query, (c.get("content") or "")) for c in chunks]
    raw_scores = model.predict(pairs)  # numpy array (real) ou list (mock)
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    enriched: list[dict] = []
    for c, logit in zip(chunks, raw_scores):
        prob = _sigmoid(float(logit))
        if prob < min_score:
            continue
        enriched.append({
            **c,
            "embed_score": c.get("score"),
            "rerank_score": prob,
            "score": prob,
        })
    enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
    return enriched[:top_k]


def warmup() -> None:
    """Força o load do modelo no startup (evita primeira request lenta).

    Chamável idempotentemente. Se ``enable_reranker`` está OFF na config,
    é no-op.
    """
    from config import settings
    if not settings.enable_reranker:
        return
    try:
        _load_model()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Reranker warmup falhou (pipeline cai pro ranking sem rerank): {e}")
