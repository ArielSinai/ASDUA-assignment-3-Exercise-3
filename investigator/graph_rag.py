from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from investigator.prompt_limit import enforce_prompt_limit
from investigator.story_loader import StoryMessage


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class GraphRAGConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed_top_k: int = 6          # כמה הודעות “seed” לפי similarity
    hop_radius: int = 2          # כמה צעדים בגרף להרחיב
    max_messages: int = 18       # כמה הודעות מקסימום לשלוח ל-LLM (מגבלה רכה)


class GraphRAG:
    """
    Graph-based RAG (no external graph libs):
    - Nodes: messages
    - Edges: temporal adjacency + same sender/receiver adjacency
    - Retrieve: embedding similarity to get seed nodes
    - Expand: BFS hop_radius to collect multi-message evidence
    - Prompt budgeting: never exceed 3000 chars (hard requirement)
    """

    def __init__(self, config: GraphRAGConfig | None = None):
        self.config = config or GraphRAGConfig()
        self.model = SentenceTransformer(self.config.model_name)

        self.messages: list[StoryMessage] = []
        self.msg_text: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.adj: list[set[int]] = []

    def index(self, messages: list[StoryMessage]) -> None:
        self.messages = [m for m in messages if (m.body or "").strip()]
        self.msg_text = [self._render(m) for m in self.messages]

        # Build adjacency graph
        n = len(self.messages)
        self.adj = [set() for _ in range(n)]

        # 1) temporal edges (i <-> i+1)
        for i in range(n - 1):
            self.adj[i].add(i + 1)
            self.adj[i + 1].add(i)

        # 2) same sender/receiver “nearby” edges:
        # connect each message to the previous message from same sender and same receiver
        last_by_sender: dict[str, int] = {}
        last_by_receiver: dict[str, int] = {}

        for i, m in enumerate(self.messages):
            s = m.sender_id or ""
            r = m.receiver_id or ""
            if s and s in last_by_sender:
                j = last_by_sender[s]
                self.adj[i].add(j)
                self.adj[j].add(i)
            if r and r in last_by_receiver:
                j = last_by_receiver[r]
                self.adj[i].add(j)
                self.adj[j].add(i)
            if s:
                last_by_sender[s] = i
            if r:
                last_by_receiver[r] = i

        # embeddings
        embs = self.model.encode(
            self.msg_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = embs.astype(np.float32)

    def answer(self, question: str, llm: LLMClient) -> str:
        if self.embeddings is None:
            raise RuntimeError("Must call index() before answer().")

        seed_idx = self._retrieve_seeds(question, top_k=self.config.seed_top_k)
        expanded_idx = self._expand(seed_idx, radius=self.config.hop_radius)

        # Deterministic ordering by time (index)
        expanded_idx_sorted = sorted(expanded_idx)

        # Soft cap on how many candidates we even consider
        candidates = expanded_idx_sorted[: self.config.max_messages]

        # ---- Prompt budgeting: add blocks incrementally until reaching 3000 chars ----
        selected_blocks: list[str] = []

        for i in candidates:
            block = self._render_xml(self.messages[i])

            tentative_blocks = selected_blocks + [block]
            tentative_context = "\n\n".join(tentative_blocks)

            try:
                # _build_prompt enforces the 3000 char limit
                _ = self._build_prompt(question, tentative_context, limit=3000)
                selected_blocks.append(block)
            except ValueError:
                # Adding this block exceeds limit -> stop
                break

        # If nothing fit (rare), fall back to the single best seed message
        if not selected_blocks and seed_idx:
            selected_blocks = [self._render_xml(self.messages[seed_idx[0]])]

        context = "\n\n".join(selected_blocks)
        prompt = self._build_prompt(question, context, limit=3000)
        return llm.complete(prompt)

    def _retrieve_seeds(self, question: str, top_k: int) -> list[int]:
        q_vec = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)

        sims = self.embeddings @ q_vec
        idx = np.argsort(-sims)[:top_k]
        return [int(i) for i in idx]

    def _expand(self, seeds: list[int], radius: int) -> set[int]:
        visited: set[int] = set(seeds)
        frontier: set[int] = set(seeds)

        for _ in range(radius):
            nxt: set[int] = set()
            for u in frontier:
                for v in self.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break

        return visited

    def _build_prompt(self, question: str, context: str, limit: int = 3000) -> str:
        base = (
            "You are AI Investigator 1.0.\n"
            "Answer the user's question using ONLY the provided story messages.\n"
            "If you don't know, explain why (not in story / not conclusive).\n\n"
            "Return exactly in this format:\n"
            "ANSWER: <short answer>\n"
            "EVIDENCE:\n"
            "<paste 1-5 exact message blocks that prove it>\n"
            "REASON: <why this evidence supports the answer>\n\n"
            f"QUESTION: {question}\n\n"
            "STORY MESSAGES:\n"
            f"{context}\n"
        )
        enforce_prompt_limit(base, limit=limit)
        return base

    def _render(self, m: StoryMessage) -> str:
        s = m.sender_name or m.sender_id or "UNKNOWN"
        r = m.receiver_name or m.receiver_id or "UNKNOWN"
        body = (m.body or "").strip().replace("\n", " ")
        return f"[{m.ts}] {s} -> {r}: {body}"

    def _render_xml(self, m: StoryMessage) -> str:
        body = (m.body or "").strip().replace("\n", " ")
        return (
            f'<message ts="{m.ts}" id="{m.id}">\n'
            f'  <sender ref="{m.sender_id}"/>\n'
            f'  <receiver ref="{m.receiver_id}"/>\n'
            f"  <body>{body}</body>\n"
            f"</message>"
        )
