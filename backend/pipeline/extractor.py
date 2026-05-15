"""
Extração de texto de arquivos PDF e DOCX.
Retorna o texto limpo como string.
"""

import io
import re
import unicodedata

import fitz          # PyMuPDF
import docx          # python-docx


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Detecta o tipo pelo nome do arquivo e extrai o texto.
    Levanta ValueError para tipos não suportados.
    """
    name_lower = filename.lower()

    if name_lower.endswith(".pdf"):
        return _extract_pdf(file_bytes)
    elif name_lower.endswith(".docx"):
        return _extract_docx(file_bytes)
    elif name_lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Tipo de arquivo não suportado: {filename}")


def _extract_pdf(file_bytes: bytes) -> str:
    """
    Extrai texto de PDF usando PyMuPDF (fitz).
    Preserva a estrutura de parágrafos com quebras de linha.
    """
    text_parts = []

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"[Página {page_num}]\n{page_text}")

    full_text = "\n\n".join(text_parts)
    return _clean_text(full_text)


def _extract_docx(file_bytes: bytes) -> str:
    """
    Extrai texto de DOCX usando python-docx.
    Mantém a estrutura de parágrafos.
    """
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Extrai também texto de tabelas
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                paragraphs.append(row_text)

    return _clean_text("\n\n".join(paragraphs))


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normaliza para que acentos no texto e na query se equivalham
    text = unicodedata.normalize("NFC", text)
    return text.strip()