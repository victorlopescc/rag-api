"""Chunker do RAG — divide texto em pedaços de tamanho limitado.

Estratégia: split recursivo por separadores em ordem de preferência
(\\n\\n -> \\n -> "." -> " " -> raw). Aplica overlap entre chunks
consecutivos pra preservar continuidade através de fronteiras.

Histórico
---------
Foi tentado um chunker estrutura-aware que detectava marcadores como
``Art. N``, ``§ N``, ``CAPÍTULO X``, etc. e splitava nesses limites.
No eval (qwen 7b + mMiniLMv2 reranker, 32 perguntas), a versão
estrutural caiu de 21/32 pra 19/32: variância de tamanho de chunks
confundiu o reranker e o LLM ocasionalmente respondia só com a
citação em vez de extrair texto de chunks pequenos. Removido. Pode
voltar a fazer sentido com reranker maior (BGE-large) ou LLM maior
(qwen 14b+) — caso queira reativar, ver histórico do git.
"""
from __future__ import annotations

from config import settings


def split_text(text: str) -> list[str]:
    """Divide texto em chunks ≤ chunk_size + overlap (com pequena margem).
    Filtra chunks com < 30 chars após strip.
    """
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    separators = ["\n\n", "\n", ".", " ", ""]

    chunks = _split_recursive(text, separators, size)
    chunks = _apply_overlap(chunks, overlap, size)
    chunks = [c.strip() for c in chunks if len(c.strip()) >= 30]
    return chunks


def _split_recursive(text: str, separators: list[str], size: int) -> list[str]:
    """Recursivo: se um pedaço continua maior que ``size``, tenta
    o próximo separador; se nenhum funciona, corta no tamanho bruto."""
    if len(text) <= size:
        return [text]

    for idx, sep in enumerate(separators):
        if sep == "":
            return [text[i:i + size] for i in range(0, len(text), size)]
        if sep not in text:
            continue

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        rest = separators[idx + 1:]
        for part in parts:
            if len(part) > size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(_split_recursive(part, rest, size))
                continue
            candidate = current + sep + part if current else part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    return [text[i:i + size] for i in range(0, len(text), size)]


def _apply_overlap(chunks: list[str], overlap: int, size: int) -> list[str]:
    """Adiciona overlap entre chunks consecutivos, garantindo que o
    resultado não exceda ``size`` significativamente."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    safe_overlap = min(overlap, max(1, size // 4))
    result = [chunks[0]]
    for chunk in chunks[1:]:
        tail = result[-1][-safe_overlap:]
        merged = tail + " " + chunk
        if len(merged) > size:
            merged = merged[:size]
        result.append(merged)
    return result
