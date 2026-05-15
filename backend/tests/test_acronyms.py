"""Testes do expansor de siglas."""
from pipeline.acronyms import (
    detect_categories,
    expand_acronyms,
    query_variants,
    suggest_category,
)


def test_expands_known_acronym():
    assert expand_acronyms("quando vai ser a ADA?") == \
        "quando vai ser a ADA (Avaliação de Desempenho Acadêmico)?"


def test_expands_multiple_acronyms_in_same_query():
    out = expand_acronyms("o TCC tem PPC?")
    assert "TCC (Trabalho de Conclusão de Curso)" in out
    assert "PPC (Projeto Pedagógico do Curso)" in out


def test_does_not_expand_when_full_form_already_present():
    """Se o aluno já escreveu o nome completo, não duplica."""
    q = "quando vai ser a Avaliação de Desempenho Acadêmico?"
    assert expand_acronyms(q) == q


def test_does_not_expand_substring():
    """ADA está dentro de 'AGENDADAS' — não deve casar."""
    assert "AGENDADAS" in expand_acronyms("provas AGENDADAS")
    assert "Avaliação de Desempenho" not in expand_acronyms("provas AGENDADAS")


def test_case_insensitive_match():
    out = expand_acronyms("posso usar calculadora na ada?")
    # Sigla minúscula é detectada e expandida (normaliza pra forma do dict).
    assert "ADA (Avaliação de Desempenho Acadêmico)" in out


def test_unknown_acronym_passes_through():
    assert expand_acronyms("o XYZ existe?") == "o XYZ existe?"


def test_query_variants_returns_only_original_when_nothing_to_expand():
    assert query_variants("posso usar calculadora?") == ["posso usar calculadora?"]


def test_query_variants_includes_expansion_when_acronym_present():
    variants = query_variants("quando vai ser a ADA?")
    assert variants[0] == "quando vai ser a ADA?"
    assert any("Avaliação de Desempenho Acadêmico" in v for v in variants)


# ---------------------------------------------------------------------------
# detect_categories — checa que keywords hardcoded foram removidas
# ---------------------------------------------------------------------------

def test_detect_categories_via_acronym_in_text():
    assert detect_categories("Quanto vale a ADA?") == ["ADA"]
    assert detect_categories("orientador de TCC") == ["TCC"]


def test_detect_categories_does_NOT_use_keyword_heuristics():
    """As listas hardcoded (ada_keywords/tcc_keywords) foram removidas.
    Sem a sigla, retorna vazio — BM25 e denso cuidam do resto."""
    assert detect_categories("posso usar calculadora?") == []
    assert detect_categories("quem é meu orientador?") == []


# ---------------------------------------------------------------------------
# suggest_category
# ---------------------------------------------------------------------------

def test_suggest_category_by_filename_strong_signal():
    """Sigla no nome do arquivo é o sinal mais forte."""
    assert suggest_category("Resolucao_ADA_2026.pdf", "qualquer texto") == "ADA"
    assert suggest_category("Regulamento TCC.docx", "") == "TCC"
    assert suggest_category("PPC - Computacao.pdf", "") == "PPC"


def test_suggest_category_by_filename_substring_does_not_match():
    """Sigla precisa ser palavra inteira no filename, não substring."""
    # "ADAPTAR" contém "ADA" mas não como palavra isolada.
    assert suggest_category("ADAPTAR_curso.pdf", "") is None


def test_suggest_category_by_text_when_filename_neutral():
    """Filename sem sigla mas texto cita ADA várias vezes."""
    text = "A ADA será aplicada. Os alunos devem comparecer à ADA. " \
           "A ADA tem regras específicas. ADA. ADA."
    out = suggest_category("documento_qualquer.pdf", text)
    assert out == "ADA"


def test_suggest_category_threshold_avoids_passing_mention():
    """Texto cita a sigla 1-2 vezes só (referência cruzada) → não sugere."""
    text = (
        "Este é o regulamento de estágio. Conforme estabelecido pelo PPC, "
        "o estágio é obrigatório. " + ("texto sobre estágio. " * 100)
    )
    # PPC aparece 1x — abaixo do threshold (3).
    assert suggest_category("Estagio.pdf", text) is None


def test_suggest_category_returns_none_when_no_signal():
    assert suggest_category("documento.pdf", "texto sem nenhuma sigla aqui") is None
    assert suggest_category(None, None) is None
    assert suggest_category("", "") is None


def test_suggest_category_filename_takes_precedence_over_text():
    """Filename forte vence contagem no texto."""
    text = "PPC " * 20  # texto cita PPC muitas vezes
    # Mas filename tem ADA → ADA vence.
    assert suggest_category("Resolucao_ADA.pdf", text) == "ADA"
