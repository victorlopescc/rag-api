"""Testa o RAG engine com o retrieve/LLM mockados."""
from unittest.mock import patch

from rag_engine import ask
from pipeline.prompt_builder import FALLBACK_MESSAGE


def _chunk(id_, content, score, document_id="doc-1"):
    return {
        "id": id_,
        "content": content,
        "metadata": {"filename": "doc.txt", "document_id": document_id},
        "distance": 1 - score,
        "score": score,
    }


def test_ask_returns_fallback_when_no_relevant_chunks():
    # Score 0.01 fica abaixo de qualquer threshold (similarity_threshold=0.20
    # quando reranker OFF; reranker_min_score=0.05 quando ON).
    with patch("rag_engine.retrieve", return_value=[_chunk("c1", "irrelevante", 0.01)]), \
         patch("rag_engine.generate") as llm:
        resp = ask("Qual é o preço da cantina?")

    assert resp.was_fallback is True
    assert resp.answer == FALLBACK_MESSAGE
    assert resp.chunks_used == []
    llm.assert_not_called()


def test_ask_calls_llm_with_relevant_chunks():
    with patch("rag_engine.retrieve", return_value=[_chunk("c1", "O curso dura 4 anos.", 0.9)]), \
         patch("rag_engine.generate", return_value="O curso dura 4 anos.") as llm:
        resp = ask("Qual a duração do curso?")

    assert resp.was_fallback is False
    assert resp.chunks_used == [
        {"id": "c1", "document_id": "doc-1", "score": 0.9}
    ]
    llm.assert_called_once()
    assert "O curso dura 4 anos." in llm.call_args.args[0]


def test_ask_marks_fallback_when_llm_repeats_fallback_phrase():
    with patch("rag_engine.retrieve", return_value=[_chunk("c1", "conteúdo", 0.9)]), \
         patch("rag_engine.generate", return_value=FALLBACK_MESSAGE):
        resp = ask("pergunta")

    assert resp.was_fallback is True


def test_ask_forwards_strategy_to_retrieve():
    with patch("rag_engine.retrieve", return_value=[]) as retr, \
         patch("rag_engine.generate"):
        resp = ask("x", strategy="query_rewrite")

    retr.assert_called_once()
    assert retr.call_args.kwargs["strategy"] == "query_rewrite"
    assert resp.strategy == "query_rewrite"


def test_ask_passes_category_to_retrieve():
    with patch("rag_engine.retrieve", return_value=[]) as retr, \
         patch("rag_engine.generate"):
        ask("x", category="regulamento")

    assert retr.call_args.kwargs["category"] == "regulamento"


def test_ask_measures_latency():
    with patch("rag_engine.retrieve", return_value=[]), \
         patch("rag_engine.generate"):
        resp = ask("x")

    assert resp.latency_ms >= 0


def test_ask_prepends_prior_question_into_retrieval_only():
    """prior_question deve entrar no retrieve mas NÃO no prompt do LLM."""
    captured = {}
    def fake_retrieve(question, **_):
        captured["retrieve_q"] = question
        return [_chunk("c1", "30 questões.", 0.9)]

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "30 questões."

    with patch("rag_engine.retrieve", side_effect=fake_retrieve), \
         patch("rag_engine.generate", side_effect=fake_generate):
        ask("quantas questões?", prior_question="quando vai ser a ADA?")

    # Retrieval recebeu as duas perguntas concatenadas.
    assert "ADA" in captured["retrieve_q"]
    assert "quantas questões?" in captured["retrieve_q"]
    # Prompt do LLM viu APENAS a pergunta atual.
    assert "quantas questões?" in captured["prompt"]
    assert "quando vai ser a ADA?" not in captured["prompt"]


def test_ask_default_strategy_is_default():
    with patch("rag_engine.retrieve", return_value=[]) as retr, \
         patch("rag_engine.generate"):
        resp = ask("x")

    assert retr.call_args.kwargs["strategy"] == "default"
    assert resp.strategy == "default"
