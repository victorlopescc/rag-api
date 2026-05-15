"""Chama o modelo LLM no Ollama e retorna a resposta como string."""
import re

import httpx

from config import settings

_client: httpx.Client | None = None

# Mensagem usada quando o pós-processamento poda toda a resposta a ponto de
# sobrar lixo. Duplicada aqui de pipeline.prompt_builder pra evitar import
# circular.
FALLBACK_PLACEHOLDER = "Não encontrei essa informação nos documentos disponíveis."


# Remove blocos de caracteres CJK (chinês/japonês/coreano), cirílico,
# árabe, etc. Vimos qwen2.5:14b ocasionalmente vazar tokens em chinês
# no meio de respostas em PT — provavelmente "language drift" pós-stop.
# Mantemos apenas latim, dígitos, pontuação e símbolos comuns.
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


def _get_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        # Timeout p/ qwen2.5:7b — cabe na GPU e responde em ~3-8s.
        # 90s dá folga pra primeiro request (cold start do Ollama) e
        # pra prompts longos sem deixar o usuário pendurado.
        _client = httpx.Client(base_url=settings.ollama_base_url, timeout=90.0)
    return _client


def generate(prompt: str) -> str:
    """Envia o prompt para o Ollama e retorna o texto gerado.

    Usa parâmetros conservadores (temperature baixa) porque este é um RAG
    factual — alucinação aqui vai direto pro WhatsApp do aluno.

    Faz até 2 retries com backoff em erro 5xx do Ollama. Com qwen2.5:7b
    cabendo na VRAM da RTX 4050 (6GB), os erros 5xx ficaram raros —
    eram comuns no qwen14b por OOM/CUDA. Mantemos o retry como
    salvaguarda mas com backoff curto pra não pendurar 3min em falhas.
    """
    import time

    payload = {
        "model": settings.ollama_llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,       # factual, quase determinístico
            "top_p": 0.9,
            "repeat_penalty": 1.1,    # evita bullets/frases repetidas
            # qwen2.5:7b-q4_K_M (~4.4GB) + KV cache + ativações precisam
            # caber em ~5.3GB úteis da RTX 4050 (6GB total - ~800MB que
            # Windows/Chrome reservam). 4096 dá folga e cobre 10 chunks
            # de 500 tokens + system prompt sem truncar.
            "num_ctx": 4096,
            "num_predict": 320,       # respostas curtas mas com folga
            "stop": ["PERGUNTA:", "CONTEXTO:", "\n---"],
        },
    }
    client = _get_client()
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = client.post("/api/generate", json=payload)
            if response.status_code >= 500:
                body = (response.text or "")[:300]
                logger = __import__("logging").getLogger(__name__)
                logger.warning(
                    f"Ollama {response.status_code} em generate "
                    f"(tentativa {attempt + 1}/3). Body: {body}"
                )
                response.raise_for_status()
            response.raise_for_status()
            break
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = e
            # Backoff curto: 1s, 2s. Total worst case ~3s + tempos
            # de request, bem longe dos 3min do setup anterior.
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise
    text = response.json()["response"].strip()
    # Remove preâmbulos do tipo "Com base no contexto, ..." que o modelo
    # insiste em gerar mesmo quando o system prompt proíbe.
    text = _PREAMBLE_RE.sub("", text).strip()
    # Remove meta-comentários entre parênteses (ambiguidade, suposições, etc.)
    # — proibidos pelo prompt mas o modelo às vezes ainda gera.
    text = _META_PAREN_RE.sub(" ", text).strip()
    # Remove frases inteiras de hedge ("Não há resposta específica..." etc.)
    text = _META_PHRASE_RE.sub("", text).strip()
    # Remove tokens em outros idiomas (CJK, cirílico, etc.) que o
    # qwen2.5:14b ocasionalmente vaza no meio de respostas em PT.
    text = _NON_LATIN_RE.sub("", text).strip()
    # Limpa espaços duplicados que possam ter sobrado.
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Se sobrou só pontuação/lixo depois das limpezas, devolve fallback.
    if len(text) < 3 or text.strip(".,;: \n\t") == "":
        text = FALLBACK_PLACEHOLDER
    return text
