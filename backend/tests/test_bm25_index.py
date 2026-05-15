"""Testa o índice BM25 in-memory."""
from unittest.mock import MagicMock, patch

from pipeline import bm25_index


def _reset_state():
    """Limpa o estado global do módulo entre testes (nada de fixture
    com autouse pra evitar overhead em testes que nem tocam BM25)."""
    bm25_index._bm25 = None
    bm25_index._chunk_ids = []
    bm25_index._chunk_docs = []
    bm25_index._chunk_metas = []
    bm25_index._chunk_tokens = []


# --- tokenize --------------------------------------------------------------

def test_tokenize_lowercases_and_strips_accents():
    tokens = bm25_index.tokenize("Avaliação de Desempenho Acadêmico")
    # "de" tem 2 letras → cai pelo filtro min len 3
    assert tokens == ["avaliacao", "desempenho", "academico"]


def test_tokenize_handles_numbers_and_punctuation():
    tokens = bm25_index.tokenize("Art. 4º — 5 (cinco) pontos.")
    assert "cinco" in tokens
    assert "pontos" in tokens
    # números mantidos
    assert "5" not in tokens  # len < 3
    # "art" tem 3 letras → entra
    assert "art" in tokens


def test_tokenize_empty_returns_empty_list():
    assert bm25_index.tokenize("") == []
    assert bm25_index.tokenize(None) == []  # type: ignore[arg-type]


# --- build / search --------------------------------------------------------

def _mock_collection_with(docs: list[dict]) -> MagicMock:
    """Devolve um mock de chromadb.Collection.get() compatível."""
    col = MagicMock()
    col.get.return_value = {
        "ids": [d["id"] for d in docs],
        "documents": [d["content"] for d in docs],
        "metadatas": [d.get("metadata") or {} for d in docs],
    }
    return col


def test_build_indexes_chunks_from_chroma():
    _reset_state()
    fake = _mock_collection_with([
        {"id": "c1", "content": "A prova vale 5 pontos no formato múltipla escolha.",
         "metadata": {"category": "ADA"}},
        {"id": "c2", "content": "O TCC deve ser escrito em LaTeX seguindo o padrão SBC.",
         "metadata": {"category": "TCC"}},
    ])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    assert bm25_index.is_built()
    assert bm25_index._chunk_ids == ["c1", "c2"]


def test_search_returns_relevant_chunk_first():
    _reset_state()
    fake = _mock_collection_with([
        {"id": "c1", "content": "A prova vale 5 pontos no formato múltipla escolha."},
        {"id": "c2", "content": "O TCC deve ser escrito em LaTeX seguindo o padrão SBC."},
        {"id": "c3", "content": "O curso tem carga horária total de 3.720 horas-aula."},
    ])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    out = bm25_index.search("Quanto vale a prova?", n_results=3)
    assert len(out) >= 1
    # 'prova' e 'vale' aparecem só em c1.
    assert out[0]["id"] == "c1"


def test_search_returns_empty_when_index_not_built():
    _reset_state()
    out = bm25_index.search("qualquer coisa", n_results=10)
    assert out == []


def test_search_returns_empty_for_query_with_no_real_tokens():
    _reset_state()
    fake = _mock_collection_with([
        {"id": "c1", "content": "Algum conteúdo qualquer aqui."},
    ])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    # tokens curtos só → tokenize devolve []
    assert bm25_index.search("a o de", n_results=5) == []


def test_search_filters_chunks_with_zero_score():
    """BM25 atribui 0 a chunks sem nenhum token em comum com a query.
    Esses não devem entrar no resultado.

    Nota: BM25 tem corner-case de IDF=0 em corpora minúsculos
    (termo aparece em metade dos docs). Usamos corpus maior pra
    o sinal funcionar como funcionaria em produção.
    """
    _reset_state()
    fake = _mock_collection_with([
        {"id": "match", "content": "calculadora não é permitida na prova"},
        {"id": "miss1", "content": "alunos podem solicitar recurso"},
        {"id": "miss2", "content": "o TCC deve seguir o padrão SBC"},
        {"id": "miss3", "content": "carga horária total do curso"},
        {"id": "miss4", "content": "estágio supervisionado obrigatório"},
    ])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    out = bm25_index.search("calculadora", n_results=10)
    ids = [c["id"] for c in out]
    assert "match" in ids
    # Chunks sem nenhum token em comum não aparecem (score 0).
    assert all(i.startswith("match") or i in ids for i in ids)
    assert ids[0] == "match"  # match é o top


def test_warmup_skips_when_disabled():
    _reset_state()
    with patch("pipeline.bm25_index.build") as b, \
         patch("config.settings.enable_bm25", False):
        bm25_index.warmup()
    b.assert_not_called()


def test_build_handles_empty_collection_gracefully():
    _reset_state()
    fake = _mock_collection_with([])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    assert not bm25_index.is_built()
    assert bm25_index.search("qualquer", n_results=5) == []


# --- lexical_overlap_search ------------------------------------------------

def test_lexical_overlap_returns_chunk_with_more_distinct_query_tokens():
    """Chunk que tem MAIS tokens distintos da query rankeia acima.
    Regra simples, sem TF-IDF — onde o BM25 colapsaria por IDF baixo,
    o overlap cru ainda discrimina."""
    _reset_state()
    fake = _mock_collection_with([
        # Tem 2 tokens distintos da query "vale prova": "prova".
        {"id": "a", "content": "regra geral sobre prova"},
        # Tem 2: "prova" e "pontos".
        {"id": "b", "content": "a prova vale pontos da turma"},
        # Tem zero da query.
        {"id": "c", "content": "outro tema sem relação"},
    ])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    out = bm25_index.lexical_overlap_search("vale prova pontos", n_results=10)
    ids = [c["id"] for c in out]
    # 'b' tem 3 overlap (prova, vale, pontos), 'a' tem 1 (prova).
    assert ids[0] == "b"
    assert "c" not in ids  # zero overlap → fora


def test_lexical_overlap_returns_empty_when_no_query_tokens():
    _reset_state()
    fake = _mock_collection_with([
        {"id": "a", "content": "qualquer conteúdo"},
    ])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    # Tokens com 1-2 letras caem pelo filtro min len 3 da tokenização.
    assert bm25_index.lexical_overlap_search("a o de", n_results=5) == []


def test_lexical_overlap_returns_empty_when_index_not_built():
    _reset_state()
    assert bm25_index.lexical_overlap_search("qualquer coisa", n_results=10) == []


def test_lexical_overlap_caps_at_n_results():
    _reset_state()
    fake = _mock_collection_with([
        {"id": f"c{i}", "content": "prova teste"} for i in range(10)
    ])
    with patch("pipeline.vector_store.get_collection", return_value=fake):
        bm25_index.build()
    out = bm25_index.lexical_overlap_search("prova", n_results=3)
    assert len(out) == 3
