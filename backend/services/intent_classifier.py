"""Classificador de intenção da mensagem do aluno dentro de uma sessão.

Duas camadas:

1. Fast-path (regex) em ``classify_fast`` — herdado do session_manager
   (apenas "yes" é detectado aqui de forma barata).
2. LLM em ``classify_with_llm`` — dada a última pergunta e resposta
   do bot + a nova mensagem do aluno, decide entre
   ``yes | no | rephrase | new_topic | unclear``.

A fachada ``classify`` combina as duas: tenta o fast-path; se for
inconclusivo e houver contexto anterior, cai no LLM. Nunca levanta —
em caso de erro na API devolve ``unclear`` e loga.

Ser robusto importa mais do que ser preciso: o custo de um
``new_topic`` confundido com ``rephrase`` é só "gasta uma tentativa
no mesmo tema"; não corrompe dados.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

import litellm

from config import settings
from services.session_manager import classify_fast

logger = logging.getLogger(__name__)

Intent = Literal["yes", "no", "rephrase", "new_topic", "unclear"]
VALID: set[Intent] = {"yes", "no", "rephrase", "new_topic", "unclear"}


@dataclass
class Prior:
    """Último turno do bot dentro da sessão, para dar contexto ao LLM."""
    question: str
    answer: str


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "Você classifica a intenção da nova mensagem de um aluno que conversa "
    "com um chatbot acadêmico. Leia a pergunta anterior, a resposta do bot "
    "e a mensagem nova. Decida APENAS uma categoria:\n\n"
    "- yes: o aluno confirma resolução (ex: 'obrigado', 'resolveu', 'ajudou').\n"
    "- no: o aluno diz EXPLICITAMENTE que não foi respondido ou contesta a "
    "resposta (ex: 'não foi isso', 'não entendi', 'isso está errado').\n"
    "- rephrase: o aluno repete a MESMA pergunta com palavras diferentes "
    "porque não foi respondida (ex: pergunta 'quanto dura?' depois 'qual a "
    "duração?'). Apenas a MESMA pergunta reformulada.\n"
    "- new_topic: TODA pergunta factual nova, mesmo que sobre o mesmo "
    "documento/assunto. Se o aluno acabou de perguntar 'quando é a prova?' "
    "e agora pergunta 'posso usar calculadora?', isso é new_topic — são "
    "perguntas distintas, não reformulação.\n"
    "- unclear: não dá pra decidir.\n\n"
    "REGRA: se a nova mensagem é uma pergunta factual diferente da "
    "anterior (mesmo no mesmo tema), classifique como new_topic, NÃO "
    "rephrase. Rephrase é só quando a pergunta é literalmente a mesma "
    "reformulada.\n\n"
    "Responda APENAS um JSON: {\"intent\": \"<categoria>\"}"
)


def _build_prompt(prior: Prior, new_message: str) -> str:
    return (
        f"{_SYSTEM}\n\n"
        f"Pergunta anterior do aluno: {prior.question}\n"
        f"Resposta do bot: {prior.answer}\n"
        f"Nova mensagem do aluno: {new_message}\n\n"
        f"JSON:"
    )


# ---------------------------------------------------------------------------
# Chamada à LLM (via litellm — provider configurável)
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_intent(raw: str) -> Intent:
    """Extrai o campo intent do retorno do LLM. Tolerante a ruído ao redor."""
    candidates: list[str] = []
    m = _JSON_RE.search(raw)
    if m:
        candidates.append(m.group(0))
    candidates.append(raw)

    for text in candidates:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        val = str(obj.get("intent", "")).strip().lower()
        if val in VALID:
            return val  # type: ignore[return-value]

    # Fallback: busca a palavra-chave no texto cru (modelo pode ter escrito
    # sem JSON). Preferência pela PRIMEIRA keyword encontrada para evitar
    # matches falsos em definições que listam todas as categorias.
    lower = raw.lower()
    first: tuple[int, Intent] | None = None
    for key in ("yes", "no", "rephrase", "new_topic", "unclear"):
        idx = lower.find(key)
        if idx != -1 and (first is None or idx < first[0]):
            first = (idx, key)  # type: ignore[assignment]
    return first[1] if first else "unclear"


def classify_with_llm(prior: Prior, new_message: str) -> Intent:
    """Chama a LLM externa e devolve a intenção. ``unclear`` em erro.

    Idealmente seria ``temperature=0`` pra ser determinístico — mesma
    entrada, mesma classificação. Mas a família Gemini 3 exige
    ``temperature ~= 1.0`` (ver nota em ``pipeline.llm._temperature_for``),
    então usamos o mesmo helper aqui. A variabilidade que isso introduz
    é limitada pela escolha enxuta de intents válidas + prompt curto.

    NÃO aplicamos o pós-processamento de ``pipeline.llm.generate`` aqui
    porque o output é JSON estruturado e as regexes de limpeza removeriam
    as chaves.
    """
    from pipeline.llm import _extra_kwargs_for, _temperature_for
    try:
        response = litellm.completion(
            model=settings.llm_model,
            messages=[
                {"role": "user", "content": _build_prompt(prior, new_message)},
            ],
            temperature=_temperature_for(settings.llm_model),
            # 256 tokens com thinking desligado é folgado pro JSON
            # curto que esperamos ({"intent": "yes"}). Antes era 80, mas
            # Gemini 3 podia consumir todo o orçamento em thinking e
            # devolver string vazia → classifier caía em ``unclear``.
            max_tokens=256,
            api_key=settings.gemini_api_key,
            timeout=30.0,
            **_extra_kwargs_for(settings.llm_model),
        )
        raw = (response.choices[0].message.content or "")
        return _parse_intent(raw)
    except Exception as e:  # pragma: no cover - log path
        logger.warning(f"Intent classifier falhou: {e}")
        return "unclear"


# ---------------------------------------------------------------------------
# Fachada
# ---------------------------------------------------------------------------

def classify(new_message: str, prior: Prior | None = None) -> Intent:
    """Combina regex fast-path e LLM.

    - Sem ``prior`` (primeira mensagem da sessão): só roda o fast-path.
      Se não for "yes", retorna ``unclear``.
    - Com ``prior``: fast-path primeiro; em ``unclear`` cai no LLM.
    """
    if classify_fast(new_message) == "yes":
        return "yes"
    if prior is None:
        return "unclear"
    return classify_with_llm(prior, new_message)
