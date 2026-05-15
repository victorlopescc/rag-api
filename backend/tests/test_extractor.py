"""Testa extração de texto para TXT/PDF/DOCX."""
import io

import docx
import fitz
import pytest

from pipeline.extractor import extract_text


def test_extract_txt_decodes_utf8():
    text = extract_text("Olá mundo — acentos".encode("utf-8"), "a.txt")
    assert "Olá" in text
    assert "mundo" in text


def test_extract_txt_with_bad_bytes_does_not_raise():
    # Byte inválido de UTF-8 — errors="replace" mantém o resto legível.
    text = extract_text(b"Hello \xff world", "a.txt")
    assert "Hello" in text
    assert "world" in text


def test_extract_unknown_extension_raises():
    with pytest.raises(ValueError):
        extract_text(b"x", "file.xyz")


def test_extract_pdf_returns_page_markers():
    pdf_bytes = _build_pdf(["Página 1 — regulamento", "Página 2 — estágio"])
    text = extract_text(pdf_bytes, "doc.pdf")
    assert "Página 1" in text
    assert "regulamento" in text
    assert "estágio" in text


def test_extract_pdf_skips_empty_pages():
    pdf_bytes = _build_pdf(["conteúdo", ""])
    text = extract_text(pdf_bytes, "doc.pdf")
    assert "conteúdo" in text


def test_extract_docx_reads_paragraphs_and_tables():
    docx_bytes = _build_docx(
        paragraphs=["Parágrafo A", "Parágrafo B"],
        table_rows=[["c1", "c2"], ["c3", "c4"]],
    )
    text = extract_text(docx_bytes, "doc.docx")
    assert "Parágrafo A" in text
    assert "c1 | c2" in text
    assert "c3 | c4" in text


def test_clean_text_normalizes_whitespace():
    # Muitos espaços e quebras — extrator reduz.
    raw = "a   b\n\n\n\n\nc"
    pdf = _build_pdf([raw])
    text = extract_text(pdf, "x.pdf")
    assert "   " not in text
    assert "\n\n\n" not in text


# --- helpers ---------------------------------------------------------------

def _build_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for content in pages:
        page = doc.new_page()
        if content:
            page.insert_text((72, 72), content)
    buf = doc.tobytes()
    doc.close()
    return buf


def _build_docx(paragraphs: list[str], table_rows: list[list[str]]) -> bytes:
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table_rows:
        table = d.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for i, row in enumerate(table_rows):
            for j, val in enumerate(row):
                table.cell(i, j).text = val
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
