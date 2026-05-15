"""Expansão de siglas comuns no contexto do curso.

A query do aluno costuma trazer só a sigla ("quando vai ser a ADA?"),
mas os chunks dos documentos misturam sigla + forma extensa. Embeddings
de uma sigla curta (3 letras) têm sinal fraco — o que faz o vetor da
query bater com tudo um pouco e nada certo.

Antes de embedar, expandimos a sigla pra carregar os dois sinais.
"""
from __future__ import annotations

import re
from typing import Iterable

# Sigla → forma extensa. Match é case-insensitive, mas a sigla precisa
# aparecer como palavra inteira (boundary). Edite com cuidado: cada
# entrada nova vira mais texto na query embedada.
ACRONYMS: dict[str, str] = {
    "ADA": "Avaliação de Desempenho Acadêmico",
    "PPC": "Projeto Pedagógico do Curso",
    "TCC": "Trabalho de Conclusão de Curso",
    "NDE": "Núcleo Docente Estruturante",
    "NAI": "Núcleo de Apoio à Inclusão",
    "SGA": "Sistema de Gestão Acadêmica",
    "CCD": "Colegiado de Coordenação Didática",
    "AED": "Algoritmos e Estrutura de Dados",
    "LDDM": "Linguagens Dinâmicas para Dispositivos Móveis",
    "LAC": "Laboratório de Aprendizagem Cooperativa",
    "ENADE": "Exame Nacional de Desempenho dos Estudantes",
    "POSCOMP": "Exame Nacional para Ingresso na Pós-Graduação em Computação",
    "COREU": "Coração Eucarístico",
    "PUC": "Pontifícia Universidade Católica",
}


def _word_re(acronym: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(acronym)}\b", flags=re.IGNORECASE)


def _filename_word_re(acronym: str) -> re.Pattern[str]:
    """Match de sigla em filename, considerando ``_``/``-``/``.``/espaço
    como separadores (regra ``\\b`` do regex trata ``_`` como letra).

    Sem isso, ``ADA`` em ``Resolucao_ADA_2026.pdf`` não casaria.
    """
    return re.compile(
        rf"(?:^|[^A-Za-z0-9]){re.escape(acronym)}(?=[^A-Za-z0-9]|$)",
        flags=re.IGNORECASE,
    )


def expand_acronyms(text: str, mapping: dict[str, str] | None = None) -> str:
    """Substitui cada sigla por "SIGLA (forma extensa)" — mantém os dois
    sinais no embedding. Se a forma extensa já estiver presente, não duplica.
    """
    if not text:
        return text
    m = mapping if mapping is not None else ACRONYMS
    out = text
    for acronym, full in m.items():
        # Se o usuário já escreveu a forma extensa, não mexe.
        if full.lower() in out.lower():
            continue
        pattern = _word_re(acronym)
        if pattern.search(out):
            out = pattern.sub(f"{acronym} ({full})", out)
    return out


def query_variants(question: str) -> list[str]:
    """Devolve uma lista de variantes da query a serem embedadas e mescladas
    no retrieval. Sempre inclui a query original; adiciona a versão expandida
    apenas quando há de fato uma sigla a expandir."""
    variants: list[str] = [question]
    expanded = expand_acronyms(question)
    if expanded != question:
        variants.append(expanded)
    return variants


# Sigla → categoria do documento. Quando a query menciona uma dessas siglas,
# fazemos uma busca extra restrita a `where={"category": ...}` pra garantir
# que os poucos chunks daquele documento não sejam soterrados pelos milhares
# de chunks dos outros docs (ex: ADA tem ~7 chunks, PPC tem ~970).
ACRONYM_TO_CATEGORY: dict[str, str] = {
    "ADA": "ADA",
    "PPC": "PPC",
    "TCC": "TCC",
}


def detect_categories(question: str) -> list[str]:
    """Retorna a lista de categorias possivelmente referenciadas na query.

    Casa por boundary de palavra, case-insensitive, sobre o dicionário
    ``ACRONYM_TO_CATEGORY``. Doc-agnóstico: pra um novo documento ser
    detectável, basta o coordenador registrar sua sigla nesse dicionário.

    Histórico: versões anteriores tinham listas hardcoded de keywords
    por categoria (``ada_keywords``, ``tcc_keywords``) pra detectar
    quando o aluno não usava a sigla. Foram removidas — viraram
    band-aids específicos por documento e quebravam a generalidade
    do sistema. O retrieval híbrido (BM25 + denso) cobre o mesmo
    sinal de forma geral: queries sem sigla agora dependem do BM25
    pra match literal e do denso pra paráfrase.
    """
    found: list[str] = []
    if not question:
        return found
    for acronym, category in ACRONYM_TO_CATEGORY.items():
        if _word_re(acronym).search(question):
            if category not in found:
                found.append(category)
    return found


def suggest_category(filename: str | None, text: str | None) -> str | None:
    """Sugere uma categoria pra um documento que está sendo subido.

    Doc-agnóstico: olha as siglas registradas em ``ACRONYM_TO_CATEGORY``
    e escolhe a que aparece mais no nome do arquivo + começo do texto.
    Sem heurísticas hardcoded por documento.

    Estratégia
    ----------
    1. **Filename match (forte)**: se o nome do arquivo contém uma sigla
       registrada (ex.: "Resolucao_ADA_2026.pdf"), retorna sua categoria
       imediatamente — sinal mais confiável.
    2. **Texto: contagem por sigla**: conta quantas vezes cada sigla
       registrada aparece nos primeiros 3000 chars (cabeçalho típico do
       doc onde a sigla principal aparece muito). Retorna a categoria
       com mais hits, desde que tenha ≥ 3 menções (evita falso positivo
       em docs que só citam a sigla de passagem).
    3. **Sem match suficiente**: devolve None — o coordenador escolhe
       manualmente.

    Devolve a string da categoria (ex.: "ADA") ou None.
    """
    # 1. Filename match (com boundary que respeita _ - . espaço)
    if filename:
        for acronym, category in ACRONYM_TO_CATEGORY.items():
            if _filename_word_re(acronym).search(filename):
                return category

    # 2. Texto: contagem das primeiras 3000 chars
    if text:
        snippet = text[:3000]
        best_cat: str | None = None
        best_count = 0
        for acronym, category in ACRONYM_TO_CATEGORY.items():
            count = len(_word_re(acronym).findall(snippet))
            if count > best_count:
                best_cat = category
                best_count = count
        # Threshold mínimo pra confiar — abaixo disso é só citação de passagem.
        if best_count >= 3:
            return best_cat

    return None

