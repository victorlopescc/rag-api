"""Geração de texto via LiteLLM.

Provider e modelo são definidos em ``settings.llm_model`` (padrão atual:
``gemini/gemini-2.5-flash``). Pra trocar de provider, basta mudar a
string + setar a API key correspondente — o resto do código nem fica
sabendo.
"""
import logging
import re
import time

import litellm
from litellm.exceptions import APIError, RateLimitError, Timeout

from config import settings

logger = logging.getLogger(__name__)

# Mensagem usada quando o pós-processamento poda toda a resposta a ponto de
# sobrar lixo. Duplicada aqui de pipeline.prompt_builder pra evitar import
# circular.
FALLBACK_PLACEHOLDER = "Não encontrei essa informação nos documentos disponíveis."


def _temperature_for(model: str) -> float:
    """Define a temperatura por família de modelo.

    A família Gemini 3 EXIGE ``temperature ~= 1.0``. O próprio LiteLLM avisa:
    "Setting temperature < 1.0 for Gemini 3 models can cause infinite loops,
    degraded reasoning performance, and failure on complex tasks."

    Modelos antigos (Gemini 2.5, Qwen, Claude, OpenAI) funcionam melhor
    com temperatura baixa pra RAG factual — alucinação aqui vai direto
    pro WhatsApp do aluno.
    """
    if "gemini-3" in model:
        return 1.0
    return 0.1


def _extra_kwargs_for(model: str) -> dict:
    """Args extras por família de modelo (passados ao litellm.completion).

    Gemini 3 vem com "thinking tokens" ligados por default — o modelo
    pensa internamente antes de responder, e ESSE pensamento consome
    o orçamento de ``max_tokens``. Em testes vimos casos onde 1024 tokens
    viraram 540 de pensamento + 122 de texto visível, cortando a resposta.

    Pra RAG factual, raciocínio extenso não ajuda — queremos extração
    direta dos chunks. Setamos ``reasoning_effort=disable`` pra zerar
    o overhead de thinking e liberar todo o orçamento pro texto.
    """
    if "gemini-3" in model:
        return {"reasoning_effort": "disable"}
    return {}


# Remove blocos de caracteres CJK (chinês/japonês/coreano), cirílico,
# árabe, etc. Modelos multilíngues ocasionalmente vazam tokens em outros
# idiomas no meio de respostas em PT (mais comum no Qwen, raríssimo no
# Gemini — mantemos o filtro como salvaguarda barata).
_NON_LATIN_RE = re.compile(
    r"[　-〿぀-ゟ゠-ヿ㐀-䶿"
    r"一-鿿＀-￯Ѐ-ӿ؀-ۿऀ-ॿ]+"
)

# Preâmbulos que o modelo insiste em colocar mesmo quando o prompt proíbe.
# Removemos defensivamente após a geração.
_PREAMBLE_RE = re.compile(
    r"^\s*(com base (no|nos|na|nas) (contexto|documento|informa[çc][õo]es|trechos?)"
    r"|de acordo com (o|os|a|as) (contexto|documento|informa[çc][õo]es|trechos?)"
    r"|segundo (o|os|a|as) (contexto|documento|informa[çc][õo]es|trechos?)"
    r"|conforme (o|os|a|as) (contexto|documento|informa[çc][õo]es|trechos?))"
    r"[^\.\n]*[\.\n]\s*",
    flags=re.IGNORECASE,
)

# Meta-comentários entre parênteses que vazam o raciocínio do modelo:
#   (Ambiguidade resolvida: ...)
#   (Supus que ...)
#   (Considerando que ...)
#   (Não encontrei essa informação...)  <-- quando aparece DENTRO de outra resposta
# Removemos esses parênteses sem afetar citações de fonte tipo
# "(Resolução ADA, §3.1)", que são curtas e não começam com verbo/marcador.
_META_PAREN_RE = re.compile(
    r"\s*\(\s*(?:"
    r"ambig[uü]idade(?:\s+resolvida)?"
    r"|sup[oõu]?[sn][\s\w]*?\s+que"
    r"|considerando\s+que"
    r"|interpret(?:ei|ando|a[çc][ãa]o)"
    r"|n[ãa]o\s+encontrei\s+essa\s+informa[çc][ãa]o"
    r"|assumindo\s+que"
    r")[^\)]*\)\.?\s*",
    flags=re.IGNORECASE,
)

# Frases inteiras de meta-divagação que o modelo às vezes produz, mesmo
# quando o prompt proíbe. Usadas como hedge quando ele não tem certeza.
# Removemos a frase inteira (de pontuação a pontuação).
_META_PHRASE_RE = re.compile(
    r"(?:"
    r"n[ãa]o\s+há\s+(?:uma\s+)?(?:resposta|informa[çc][õo]es?)\s+(?:espec[íi]ficas?|expl[íi]citas?)"
    r"[^.!?\n]*[.!?\n]\s*"
    r"|no\s+entanto,?\s+(?:posso|é\s+poss[íi]vel)\s+inferir"
    r"[^.!?\n]*[.!?\n]\s*"
    r"|com\s+base\s+(?:no|nos|na|nas)\s+(?:contexto|trechos?|documentos?|informa[çc][õo]es)"
    r"[^.!?\n]*[.!?\n]?\s*"
    r")",
    flags=re.IGNORECASE,
)


def generate(prompt: str) -> str:
    """Envia o prompt para a LLM externa e retorna o texto pós-processado.

    Usa parâmetros conservadores (``temperature=0.1``) porque este é um RAG
    factual — alucinação aqui vai direto pro WhatsApp do aluno.

    Faz até 2 retries com backoff em erros transitórios da API (5xx,
    rate limit, timeout). LiteLLM normaliza esses erros entre providers.
    """
    last_err: Exception | None = None
    text: str = ""
    for attempt in range(3):
        try:
            response = litellm.completion(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=_temperature_for(settings.llm_model),
                top_p=0.9,
                # 2048 tokens (~6kB de texto em PT-BR). Cobre respostas
                # longas (listas de disciplinas, resumos de escalação)
                # com margem confortável. O Gemini é mais "tagarela" que
                # o Qwen e a família 3 tinha thinking que consumia
                # orçamento — desligamos via ``_extra_kwargs_for``.
                max_tokens=2048,
                api_key=settings.gemini_api_key,
                timeout=90.0,
                **_extra_kwargs_for(settings.llm_model),
            )
            text = (response.choices[0].message.content or "").strip()
            break
        except (APIError, RateLimitError, Timeout) as e:
            last_err = e
            logger.warning(
                f"LLM erro (tentativa {attempt + 1}/3): "
                f"{type(e).__name__}: {e}"
            )
            if attempt < 2:
                # Backoff curto: 1s, 2s. Total worst case ~3s + tempos
                # de request.
                time.sleep(1.0 * (attempt + 1))
                continue
            raise

    # Remove preâmbulos do tipo "Com base no contexto, ..." que o modelo
    # insiste em gerar mesmo quando o system prompt proíbe.
    text = _PREAMBLE_RE.sub("", text).strip()
    # Remove meta-comentários entre parênteses (ambiguidade, suposições, etc.)
    text = _META_PAREN_RE.sub(" ", text).strip()
    # Remove frases inteiras de hedge ("Não há resposta específica..." etc.)
    text = _META_PHRASE_RE.sub("", text).strip()
    # Remove tokens em outros idiomas (CJK, cirílico, etc.) — salvaguarda.
    text = _NON_LATIN_RE.sub("", text).strip()
    # Limpa espaços duplicados que possam ter sobrado.
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Se sobrou só pontuação/lixo depois das limpezas, devolve fallback.
    if len(text) < 3 or text.strip(".,;: \n\t") == "":
        text = FALLBACK_PLACEHOLDER
    return text
