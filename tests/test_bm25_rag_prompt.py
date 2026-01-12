import pytest
from investigator.bm25_rag import BM25RAG


def test_build_prompt_trims_to_fit_limit():
    rag = BM25RAG()
    chunks = ["x" * 1500, "y" * 1500, "z" * 1500]
    prompt = rag.build_prompt("Who did it?", chunks, limit=3000)
    assert len(prompt) <= 3000


def test_build_prompt_raises_if_single_chunk_too_large():
    rag = BM25RAG()
    chunks = ["x" * 4000]
    with pytest.raises(Exception):
        rag.build_prompt("Q", chunks, limit=3000)
