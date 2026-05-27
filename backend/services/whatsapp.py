"""Utilidades de WhatsApp: normalização de LID, número e textos fixos."""

import re
import secrets
import urllib.parse

WELCOME_MESSAGE = """Olá, {name}! 👋

Seu cadastro foi realizado com sucesso. Agora você pode tirar dúvidas sobre o curso de Ciência da Computação direto por aqui.

📚 *Você pode perguntar coisas como:*
• Quando vai ser a ADA?
• Quanto vale a prova?
• Quais são as regras do TCC?
• Posso usar calculadora?

💡 Pra falar direto com o coordenador a qualquer momento, envie */coordenador*.
🆘 Pra ver mais opções e dicas, envie */ajuda*."""


# Mensagem completa de ajuda — disparada por /ajuda. Inclui /cancelar
# (que é intencionalmente omitido na mensagem de boas-vindas pra não
# poluir o primeiro contato).
HELP_MESSAGE = """🆘 *Como usar o assistente da coordenação*

📚 *Posso responder dúvidas sobre:*
• ADA — Avaliação de Desempenho Acadêmico
• PPC — Projeto Pedagógico do Curso
• Regulamento do TCC
• Datas, prazos e regras gerais do curso

❓ *Exemplos de perguntas que funcionam bem:*
• Quando vai ser a ADA?
• Quanto vale a prova?
• Posso usar calculadora?
• Qual a duração mínima do TCC?
• Sou aluno do 5º período, vou fazer online?

🔁 *Se eu não entender de primeira:*
Tente reformular com mais detalhes. Posso tentar até 3 vezes — se nada resolver, te encaminho ao coordenador automaticamente.

🧑‍🏫 *Comandos especiais:*
• */ajuda* — mostra esta mensagem
• */coordenador* — falar direto com o coordenador
• */cancelar* — encerra a conversa atual (útil se você se confundiu e quer começar de novo)"""


# Resposta a saudações isoladas ("oi", "td bem"). Curta, cordial,
# já mostra exemplos pra orientar.
GREETING_REPLY = """Oi! 👋 Estou aqui pra responder dúvidas sobre o curso de Ciência da Computação.

Tente algo como:
• *Quando vai ser a ADA?*
• *Quanto vale a prova?*

Pra ver mais dicas, envie */ajuda*."""


# Resposta a trivialidades / mensagens curtas sem sentido ("ok", "?", "kkk").
# Não envia à LLM, devolve um nudge pro aluno mandar uma pergunta de fato.
TRIVIAL_REPLY = """Hmm, não entendi essa mensagem 🤔

Tente mandar uma dúvida por escrito, por exemplo:
• *Posso usar calculadora na ADA?*
• *Quem orienta o TCC?*

Se precisar de ajuda, envie */ajuda*."""


# Resposta ao /cancelar.
CANCEL_REPLY_OK = """Conversa encerrada. ✅

Quando quiser tirar outra dúvida, é só me mandar uma mensagem."""

CANCEL_REPLY_NOTHING = """Você não tem nenhuma conversa aberta no momento. 🙂

Se quiser começar uma, é só mandar sua dúvida — ou envie */ajuda* pra ver dicas."""


# Resposta a "obrigado", "valeu", "blz" etc. quando NÃO há sessão aberta.
# Sem isso, o sistema mandava esses textos pro RAG e voltava com
# fallback ("não encontrei essa informação"), o que ficava rude.
THANKS_REPLY = """De nada! 😊

Quando tiver outra dúvida sobre o curso, é só mandar."""


# Sufixo anexado a CADA resposta do RAG. Pede feedback explícito do
# aluno em vez de inferir intenção via LLM no próximo turno. Pedido
# expresso da coordenação no design do piloto: cada interação gera um
# datapoint claro de satisfação.
FEEDBACK_PROMPT_SUFFIX = """

━━━━━━━━━━━━━━━━━━━
*Conseguiu resolver sua dúvida?*
*1* — Sim, obrigado ✅
*2* — Não, vou reformular
*3* — Falar com o coordenador"""


# Resposta ao aluno que digitou "2" pedindo reformulação.
# Mantém a sessão aberta — a próxima mensagem dele é tratada como
# rephrase (incrementa attempt; na 3ª tentativa o sistema escala
# automaticamente pro coordenador).
REPHRASE_ACK = """Beleza! 💪

Manda a pergunta reformulada que eu tento responder de novo, agora com mais cuidado."""


# ============================================================================
# Live thread (conversa ao vivo aluno ↔ coordenador)
# ============================================================================

# Notifica o aluno que o coordenador entrou na conversa. Disparada quando
# o coordenador clica "Iniciar conversa" no painel.
THREAD_OPENED_NOTICE = """👨‍🏫 *Coordenação na linha.*

A partir de agora você está conversando direto com o coordenador. Pode mandar suas dúvidas e ele vai responder por aqui mesmo.

Quando quiser encerrar a conversa, envie */encerrar*."""


# Confirmação ao aluno que a thread foi encerrada pelo COORDENADOR.
THREAD_CLOSED_BY_COORDINATOR_NOTICE = """✅ O coordenador encerrou esta conversa.

Se tiver outra dúvida, é só mandar uma nova mensagem — vou tentar responder e, se precisar, escalo de novo."""


# Confirmação ao aluno após ele mandar /encerrar.
THREAD_CLOSED_BY_STUDENT_NOTICE = """✅ Conversa encerrada.

Quando quiser tirar outra dúvida, é só me mandar uma mensagem."""


# Aviso ao aluno quando o sistema fecha a thread por inatividade do coordenador.
THREAD_CLOSED_TIMEOUT_NOTICE = """⏱ A conversa com o coordenador foi encerrada por inatividade.

Se ainda precisar de ajuda, mande sua dúvida — vou tentar responder e, se precisar, te conecto de novo com a coordenação."""


# Prefixo aplicado em TODA mensagem do coordenador enviada pelo painel
# enquanto a thread está live. Deixa claro pro aluno que ainda é o
# humano respondendo (vs. bot, que não tem prefixo).
COORDINATOR_PREFIX = "👨‍🏫 *Coordenação:*\n\n"


# Sufixo anexado quando o RAG retorna fallback. Não muda a constante
# FALLBACK_MESSAGE (que o LLM precisa retornar literalmente pra ser
# detectada). É concatenado na hora de enviar pro WhatsApp.
FALLBACK_HINT_SUFFIX = (
    "\n\n💡 Tente reformular com mais detalhes ou envie */coordenador* "
    "pra falar diretamente com o coordenador."
)


def normalize_lid(jid: str) -> str:
    """Remove o sufixo de device. Ex: '240247703105761:6@lid' -> '240247703105761@lid'."""
    if not jid or "@lid" not in jid:
        return jid
    local, _, domain = jid.partition("@")
    local = local.split(":", 1)[0]
    return f"{local}@{domain}"


def normalize_phone(raw: str) -> str:
    """Remove espaços/parênteses/hífens e garante o DDI 55."""
    phone = (raw or "").strip()
    for ch in (" ", "-", "(", ")"):
        phone = phone.replace(ch, "")
    if phone and not phone.startswith("55"):
        phone = "55" + phone
    return phone


def build_welcome_text(full_name: str) -> str:
    first_name = (full_name or "").split()[0] if full_name else ""
    return WELCOME_MESSAGE.format(name=first_name)


# ============================================================================
# Token de cadastro + link wa.me
#
# Quando o aluno se cadastra no site, gera-se um token curto. O site mostra
# um link ``wa.me/<bot_number>?text=Oi%21%20...%20código:%20<token>`` pro
# aluno clicar — abrindo o WhatsApp já com a mensagem pré-formatada. Quando
# o bot recebe essa mensagem, extrai o token via regex e casa com o Student
# correspondente. Esse fluxo evita que o BOT envie mensagem proativa (que a
# Meta detecta como spam quando o destinatário demora a responder).
# ============================================================================

# Alfabeto sem caracteres ambíguos (0/O, 1/I/l) — facilita o aluno copiar
# se necessário e diminui chance de "código não funcionou" por erro de
# leitura.
_TOKEN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_TOKEN_LENGTH = 6

# Match estrito: precisa da palavra "código" (com ou sem acento) seguida
# de dois-pontos e o token. Evita falso positivo em mensagens aleatórias
# do aluno que por acaso tenham 6 letras maiúsculas.
_TOKEN_RE = re.compile(
    r"c[óo]digo\s*:?\s*([A-HJ-NP-Z2-9]{" + str(_TOKEN_LENGTH) + r"})",
    re.IGNORECASE,
)


def generate_registration_token() -> str:
    """Gera um token aleatório criptograficamente seguro pra cadastro.

    ~10⁹ combinações (32^6) — suficiente pro horizonte de cadastros do
    piloto. Colisões são tratadas pelo caller (basta gerar de novo se
    o INSERT falhar por UNIQUE constraint).
    """
    return "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(_TOKEN_LENGTH))


def extract_registration_token(text: str) -> str | None:
    """Extrai o token do corpo de uma mensagem, ou None se não houver.

    Aceita formatos tolerantes: ``código: ABC123``, ``codigo:ABC123``,
    ``Código ABC123`` (case-insensitive). Apenas o primeiro match conta.
    """
    if not text:
        return None
    m = _TOKEN_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def build_registration_link(bot_phone: str, token: str, full_name: str = "") -> str:
    """Monta o link ``https://wa.me/<bot_phone>?text=...`` que o aluno
    clica pra enviar a primeira mensagem ao bot.

    O ``bot_phone`` deve vir SEM ``+`` e COM DDI (ex: ``5531990899055``).
    O ``full_name`` é só pra deixar a mensagem mais natural pro aluno.
    """
    first_name = (full_name or "").split()[0] if full_name else ""
    saudacao = f"Oi! Sou {first_name}." if first_name else "Oi!"
    text = f"{saudacao} Quero começar a usar o bot da coordenação (código: {token})"
    encoded = urllib.parse.quote(text)
    return f"https://wa.me/{bot_phone}?text={encoded}"
