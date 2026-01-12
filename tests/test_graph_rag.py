import numpy as np
import pytest

import investigator.graph_rag as gr
from investigator.story_loader import StoryMessage


class FakeEmbedder:
    """
    Deterministic tiny embedder to avoid downloading real sentence-transformers models.
    Produces normalized vectors so cosine similarity works.
    """

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        def vec(t: str) -> np.ndarray:
            tl = (t or "").lower()
            v = np.array(
                [
                    1.0 if "usb" in tl else 0.0,
                    1.0 if "marcus" in tl else 0.0,
                    min(len(tl) / 200.0, 1.0), 
                ],
                dtype=np.float32,
            )
            n = np.linalg.norm(v) + 1e-12
            return v / n

        mat = np.stack([vec(t) for t in texts], axis=0)
        return mat


class DummyLLM:
    def __init__(self):
        self.last_prompt = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "ANSWER: dummy\nEVIDENCE:\n<message/>\nREASON: dummy"


def _m(i: str, ts: str, sender: str, receiver: str, body: str) -> StoryMessage:
    return StoryMessage(
        id=i,
        ts=ts,
        sender_id=sender,
        receiver_id=receiver,
        sender_name=None,
        receiver_name=None,
        body=body,
    )


@pytest.fixture
def fake_graph(monkeypatch):
    """
    Create GraphRAG with patched SentenceTransformer => FakeEmbedder.
    """
    class FakeST:
        def __new__(cls, *args, **kwargs):
            return FakeEmbedder()

    monkeypatch.setattr(gr, "SentenceTransformer", FakeST)
    return gr.GraphRAG(gr.GraphRAGConfig(seed_top_k=3, hop_radius=1, max_messages=10))


def test_index_builds_temporal_edges(fake_graph):
    msgs = [
        _m("m1", "t1", "a", "b", "hi"),
        _m("m2", "t2", "c", "d", "yo"),
        _m("m3", "t3", "e", "f", "sup"),
    ]
    fake_graph.index(msgs)

    assert 1 in fake_graph.adj[0]
    assert 0 in fake_graph.adj[1]
    assert 2 in fake_graph.adj[1]
    assert 1 in fake_graph.adj[2]


def test_index_builds_sender_receiver_edges(fake_graph):
    msgs = [
        _m("m1", "t1", "marcus", "alex", "first"),
        _m("m2", "t2", "zoe", "six", "other"),
        _m("m3", "t3", "marcus", "someone", "same sender as m1"),
        _m("m4", "t4", "x", "alex", "same receiver as m1"),
    ]
    fake_graph.index(msgs)

    assert 2 in fake_graph.adj[0]
    assert 0 in fake_graph.adj[2]

    assert 3 in fake_graph.adj[0]
    assert 0 in fake_graph.adj[3]


def test_retrieve_seeds_prefers_usb_message(fake_graph):
    msgs = [
        _m("m1", "t1", "zoe", "six", "nothing here"),
        _m("m2", "t2", "marcus", "alex", "Bring that USB you borrowed."),
        _m("m3", "t3", "alex", "marcus", "random talk"),
    ]
    fake_graph.index(msgs)

    seeds = fake_graph._retrieve_seeds("Who requested to bring the USB?", top_k=2)
    assert 1 in seeds  


def test_expand_collects_neighbors(fake_graph):
    msgs = [
        _m("m1", "t1", "a", "b", "msg1"),
        _m("m2", "t2", "a", "b", "msg2"),
        _m("m3", "t3", "a", "b", "msg3"),
    ]
    fake_graph.index(msgs)

    expanded = fake_graph._expand([1], radius=1)
    assert expanded == {0, 1, 2}


def test_answer_never_exceeds_prompt_limit(monkeypatch, fake_graph):
    msgs = [
        _m(f"m{i}", f"t{i}", "marcus", "alex", "USB " + ("x" * 400))
        for i in range(30)
    ]
    fake_graph.index(msgs)

    LIMIT = 600  
    def fake_enforce(prompt: str, limit: int = 3000):
        if len(prompt) > LIMIT:
            raise ValueError(f"Prompt too long: {len(prompt)}")
        return None

    monkeypatch.setattr(gr, "enforce_prompt_limit", fake_enforce)

    llm = DummyLLM()
    out = fake_graph.answer("Who requested to bring the USB?", llm)
    assert out.startswith("ANSWER:")
    assert llm.last_prompt is not None
    assert len(llm.last_prompt) <= LIMIT
