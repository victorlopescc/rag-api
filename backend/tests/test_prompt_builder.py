"""Testa o construtor de prompt do RAG."""
from pipeline.prompt_builder import FALLBACK_MESSAGE, SYSTEM_PROMPT, build_prompt


def _chunk(content, filename="doc.txt"):
    return {"content": content, "metadata": {"filename": filename}}


def test_prompt_contains_system_preamble():
    out = build_prompt("Qual a duração?", [_chunk("O curso dura 4 anos.")])
    assert SYSTEM_PROMPT in out


def test_prompt_includes_question_and_context():
    out = build_prompt(
        "Qual a duração?",
        [_chunk("O curso dura 4 anos.", "regulamento.pdf")],
    )
    assert "Qual a duração?" in out
    assert "O curso dura 4 anos." in out
    assert "regulamento.pdf" in out


def test_prompt_lists_multiple_chunks_numbered():
    out = build_prompt(
        "x",
        [_chunk("A", "f1.txt"), _chunk("B", "f2.txt")],
    )
    # Formato atual: [TRECHO N | DOC: ... | CATEGORIA: ...]
    assert "[TRECHO 1" in out
    assert "[TRECHO 2" in out
    assert "f1.txt" in out and "f2.txt" in out


def test_prompt_handles_missing_filename_metadata():
    out = build_prompt("x", [{"content": "Texto", "metadata": {}}])
    assert "documento" in out  # fallback default


def test_fallback_message_constant_is_non_empty():
    assert FALLBACK_MESSAGE.strip()
    assert "não encontrei" in FALLBACK_MESSAGE.lower()
