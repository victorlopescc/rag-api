"""Utilidades de WhatsApp: normalização de LID, número e textos fixos."""

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
# Não envia ao Ollama, devolve um nudge pro aluno mandar uma pergunta de fato.
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
