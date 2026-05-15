"""Triagem rápida (regex/lookup) da mensagem do aluno antes do pipeline RAG.

Antes de gastar uma chamada de 18–30s ao Ollama, identificamos mensagens que
claramente NÃO são perguntas factuais — saudações isoladas ("oi"),
trivialidades ("ok", "?", "kkk") e comandos explícitos (``/ajuda``,
``/cancelar``). Essas mensagens recebem respostas pré-formatadas
imediatamente.

Mensagens que **parecem** uma pergunta real (3+ palavras, OU "?", OU contêm
um termo do domínio acadêmico) seguem para o ``session_manager`` normal,
onde podem ser classificadas como ``yes`` / ``rephrase`` / etc.

Não duplica a lógica de ``/coordenador`` (que está em
``session_manager.is_escalation_sentinel``) nem de ``yes/sim/obrigado``
(que está em ``session_manager.classify_fast``).
"""
from __future__ import annotations

import unicodedata
from typing import Literal

MessageKind = Literal["greeting", "trivial", "help", "cancel", "question"]


# Saudações isoladas (match exato após normalizar). Cobertura razoável de
# formas comuns em PT-BR. Saudações dentro de uma frase maior NÃO casam
# (ex.: "oi, quando vai ser a ada?" segue como pergunta).
_GREETINGS: frozenset[str] = frozenset({
    "oi", "ola", "ole",
    "bom dia", "boa tarde", "boa noite",
    "tudo bem", "tudo bom", "td bem", "td bom",
    "e ai", "eai", "salve",
    "hey", "hi", "hello",
    "boa", "boas",
    "como vai", "como esta", "como você esta", "como voce esta",
    "tudo certo", "td certo",
    "blza", "beleza",
    "alo", "alô",
})


# Trivialidades isoladas — não são saudação nem pergunta nem confirmação.
# Curtas e sem conteúdo informativo.
_TRIVIAL_TOKENS: frozenset[str] = frozenset({
    "hmm", "uhm", "uhmm", "ehh", "hum", "humm", "hummm",
    "kkk", "kkkk", "kkkkk", "kkkkkk", "kk", "kkkkkkk",
    "rs", "rsrs", "rsrsrs", "haha", "hahaha",
    "ah", "ahn", "ahh", "aham",
    "nada", "nada nao", "nada não",
    "x", "xx", "xxx",
    "uai", "ué", "ue",
})


# Comandos slash. Comparados após normalizar, então acentos/case não
# importam mas a barra é obrigatória.
_COMMAND_HELP: frozenset[str] = frozenset({"/ajuda", "/help", "/comandos"})
_COMMAND_CANCEL: frozenset[str] = frozenset({"/cancelar", "/parar", "/sair"})
# Comando pra encerrar uma live thread com o coordenador. Diferente
# do /cancelar (que fecha a sessão de QA do bot), o /encerrar é
# especificamente pro aluno sair de uma conversa ao vivo.
_COMMAND_END_THREAD: frozenset[str] = frozenset({"/encerrar", "/fim"})


def is_end_thread_command(text: str) -> bool:
    """True se a mensagem é o comando explícito do aluno pra encerrar
    a live thread com o coordenador."""
    return _normalize(text) in _COMMAND_END_THREAD


# Termos do domínio acadêmico — se algum aparece na mensagem, mesmo que
# curta, tratamos como pergunta real (vai pro RAG).
_DOMAIN_TERMS: frozenset[str] = frozenset({
    # Documentos / siglas
    "ada", "ppc", "tcc", "nde", "nai", "puc", "sga", "coreu",
    # Provas / avaliação
    "prova", "provas", "avaliacao", "avaliacoes", "exame", "exames",
    "questao", "questoes", "alternativa", "alternativas",
    "calculadora", "recurso", "recursos",
    # Acadêmico
    "regulamento", "regulamentos", "estagio", "estagios",
    "calendario", "matricula", "horario", "horarios",
    "carga", "horaria", "disciplina", "disciplinas",
    "professor", "professores", "coordenador", "coordenacao",
    "curso", "periodo", "periodos", "semestre", "semestres",
    "aula", "aulas", "ementa", "tcc",
    # Documentos
    "documento", "documentos", "regimento", "diploma",
    # Tempo
    "data", "datas", "prazo", "prazos", "duracao",
})


def _normalize(text: str) -> str:
    """Lowercase + remove acentos + tira pontuação leve das pontas.

    Mantém ``?`` interno (relevante pra heurística de pergunta), mas
    tira ``?!.,;:`` das pontas pra match exato com saudações.
    """
    if not text:
        return ""
    norm = text.strip().lower()
    # Remove acentos
    norm = "".join(
        ch for ch in unicodedata.normalize("NFD", norm)
        if unicodedata.category(ch) != "Mn"
    )
    # Strip pontuação das pontas
    while norm and norm[0] in "?!.,;:":
        norm = norm[1:]
    while norm and norm[-1] in "?!.,;:":
        norm = norm[:-1]
    return norm.strip()


def classify(text: str) -> MessageKind:
    """Classifica a mensagem em uma das 5 categorias de triagem.

    Ordem de prioridade:
    1. Comandos slash (/ajuda, /cancelar)
    2. Saudações isoladas
    3. Confirmações ("obrigado", "sim") → ``question`` pra deixar
       passar pro ``plan_interaction``, que tem fast-path pra "yes".
    4. Trivialidades isoladas
    5. Heurística de pergunta curta SEM sinal de domínio → trivial
    6. Default → question (vai pro RAG)
    """
    if not text or not text.strip():
        return "trivial"

    norm = _normalize(text)
    if not norm:
        # Era só pontuação ("?", "??", "...") → trivial
        return "trivial"

    # Comandos slash têm prioridade
    if norm in _COMMAND_HELP:
        return "help"
    if norm in _COMMAND_CANCEL:
        return "cancel"

    # Saudações isoladas
    if norm in _GREETINGS:
        return "greeting"

    # Confirmações ("obrigado", "sim", "blz") devem passar pro
    # session_manager.classify_fast, que sabe lidar com elas (fecha a
    # sessão como resolved). Não interceptamos aqui.
    from services.session_manager import classify_fast, is_escalation_sentinel  # local para evitar import circular
    if classify_fast(text) == "yes":
        return "question"

    # Sentinel de escalação (/coordenador, coordenador, /coord, ...) é
    # tratado por session_manager.plan_interaction. Deixamos passar.
    if is_escalation_sentinel(text):
        return "question"

    # Trivialidades isoladas
    if norm in _TRIVIAL_TOKENS:
        return "trivial"

    words = norm.split()

    # Sinais de retentativa em mensagens curtas: "não entendi", "não foi",
    # "ainda não", "não respondeu" etc. Mesmo com 2 palavras e sem termo
    # de domínio, são MUITO comuns como reformulação implícita do aluno.
    # Deixar cair em "trivial" quebraria o fluxo de retentativa do RAG.
    # A primeira palavra ser "nao"/"ainda" é sinal forte; passa pro
    # session_manager classificar como rephrase.
    if words and words[0] in ("nao", "ainda"):
        return "question"

    # Heurística pra "vale chamar o RAG?":
    # - tem '?' explícito → provavelmente pergunta
    # - tem termo do domínio → pergunta
    # - tem 3+ palavras → pergunta (até "tem aula amanhã" cabe)
    # - 1-2 palavras sem nada disso → trivial
    has_question_mark = "?" in (text or "")
    has_domain = any(w in _DOMAIN_TERMS for w in words)

    if has_question_mark or has_domain or len(words) >= 3:
        return "question"
    return "trivial"
