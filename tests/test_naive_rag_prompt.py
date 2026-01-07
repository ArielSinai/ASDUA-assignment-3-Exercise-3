import pytest
from investigator.naive_rag import NaiveRAG

def test_build_prompt_trims_to_fit_limit():
    rag = NaiveRAG()
    # don't need indexing for build_prompt

    # Make huge fake chunks
    chunks = ["x" * 1500, "y" * 1500, "z" * 1500]  # total would exceed 3000 with header
    prompt = rag.build_prompt("Who did it?", chunks, limit=3000)
    assert len(prompt) <= 3000

def test_build_prompt_raises_if_single_chunk_too_large():
    rag = NaiveRAG()
    chunks = ["x" * 4000]
    with pytest.raises(Exception):
        rag.build_prompt("Q", chunks, limit=3000)
