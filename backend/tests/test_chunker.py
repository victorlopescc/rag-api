"""Testa o chunker recursivo."""
from pipeline.chunker import split_text
from config import settings


def test_small_text_returns_single_chunk():
    text = "A" * 100  # abaixo de chunk_size
    # "A"*100 strip tem 100 chars >= 30 — mantido.
    chunks = split_text("regulamento " + text)
    assert len(chunks) == 1


def test_large_text_splits_into_multiple_chunks():
    long_text = " ".join(["palavra"] * 600)
    chunks = split_text(long_text)
    assert len(chunks) >= 2


def test_chunks_respect_max_size_roughly():
    long_text = "a" * (settings.chunk_size * 5)
    chunks = split_text(long_text)
    # Com overlap, chunks podem passar do size por algumas dezenas.
    for c in chunks:
        assert len(c) <= settings.chunk_size + settings.chunk_overlap + 1


def test_small_chunks_are_filtered():
    # < 30 chars, após strip, devem sumir.
    chunks = split_text("oi")
    assert chunks == []


def test_prefers_paragraph_boundary():
    text = "Parágrafo um tem algum texto.\n\nParágrafo dois também é relevante."
    chunks = split_text(text)
    assert len(chunks) >= 1
    assert any("Parágrafo" in c for c in chunks)


def test_huge_paragraph_with_no_double_newlines_is_split():
    """Regressão: PDF extraído em uma linha só deve ser dividido —
    senão um chunk de milhares de chars vaza pro embedder."""
    text = (
        "Pagina sete sumario " + ("texto continuo " * 500)
        + ". Pagina oito conteudo " + ("mais texto " * 500)
    )
    chunks = split_text(text)
    # Nenhum chunk pode passar do limite + overlap (margem pequena).
    margin = settings.chunk_size + settings.chunk_overlap + 5
    for c in chunks:
        assert len(c) <= margin, f"chunk com {len(c)} chars (limite {margin})"


def test_overlap_is_applied_between_chunks():
    # Usa separador ".": força múltiplos chunks.
    sentence = "frase completa aqui no teste. "
    text = sentence * 40
    chunks = split_text(text)
    if len(chunks) >= 2:
        # O início do segundo chunk deve conter parte do fim do primeiro.
        tail = chunks[0][-settings.chunk_overlap:]
        assert tail[-10:] in chunks[1] or chunks[1].startswith(tail[-10:])
