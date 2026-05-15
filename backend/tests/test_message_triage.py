"""Testes da triagem de mensagens (services.message_triage)."""
import pytest

from services.message_triage import classify


@pytest.mark.parametrize("text", [
    "oi", "Oi", "OI", "ola", "Olá", "olá",
    "bom dia", "Boa tarde", "boa noite",
    "tudo bem", "tudo bem?", "Td bem?",
    "e ai", "eai", "salve!", "hey",
    "blza", "beleza",
])
def test_classifies_greetings(text):
    assert classify(text) == "greeting"


@pytest.mark.parametrize("text", [
    "?", "??", "...",
    "kkk", "kkkkk", "rs",
    "hmm", "ahn",
    "nada",
    "x", "xx",
])
def test_classifies_trivial(text):
    assert classify(text) == "trivial"


@pytest.mark.parametrize("text", ["ok", "obrigado", "valeu", "blz", "sim"])
def test_classifies_yes_words_as_question(text):
    """Confirmações passam por triage como 'question' pra deixarmos o
    fast-path do session_manager.classify_fast tratá-las como 'yes'."""
    assert classify(text) == "question"


@pytest.mark.parametrize("text", [
    "não entendi", "nao entendi",
    "não foi", "nao foi",
    "não respondeu", "nao respondeu",
    "não funcionou", "nao funcionou",
    "ainda não", "ainda nao",
    "ainda não entendi", "ainda nao entendi",
])
def test_short_negation_phrases_pass_as_question(text):
    """Mensagens curtas começando com 'não'/'nao'/'ainda' são sinais
    de retentativa — devem chegar ao session_manager pra serem
    classificadas como 'rephrase', não engolidas como trivial."""
    assert classify(text) == "question"


@pytest.mark.parametrize("text", ["/ajuda", "/AJUDA", "/help", "/comandos"])
def test_classifies_help_command(text):
    assert classify(text) == "help"


@pytest.mark.parametrize("text", ["/cancelar", "/parar", "/sair", "/SAIR"])
def test_classifies_cancel_command(text):
    assert classify(text) == "cancel"


@pytest.mark.parametrize("text", [
    "Quando vai ser a ADA?",
    "qual a duração do TCC?",
    "posso usar calculadora",  # tem termo de domínio
    "tem prova amanhã?",
    "ainda não respondeu",
    "alguma coisa sobre a grade do curso",
])
def test_classifies_real_questions(text):
    assert classify(text) == "question"


def test_empty_string_is_trivial():
    assert classify("") == "trivial"
    assert classify("   ") == "trivial"


def test_greeting_inside_sentence_is_question():
    """'oi, quando vai ser a ada?' deve passar como pergunta."""
    assert classify("oi, quando vai ser a ada?") == "question"


def test_punctuation_only_is_trivial():
    assert classify("???") == "trivial"
    assert classify("!!!") == "trivial"


def test_short_message_with_domain_term_is_question():
    """'ada amanha?' tem só 2 palavras mas tem termo de domínio."""
    assert classify("ada amanha?") == "question"


def test_short_message_without_signal_is_trivial():
    """'pode falar' — 2 palavras, sem '?', sem termo do domínio → trivial."""
    assert classify("pode falar") == "trivial"
