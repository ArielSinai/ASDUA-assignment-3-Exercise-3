from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import re

from investigator.prompt_limit import enforce_prompt_limit
from investigator.story_loader import StoryMessage


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", (text or "").lower())


def _render_message_xmlish(m: StoryMessage) -> str:
    sid = m.sender_id or "unknown"
    rid = m.receiver_id or "unknown"
    body = (m.body or "").strip()
    ts = m.ts or ""
    return (
        f'<message id="{m.id}" ts="{ts}">\n'
        f'  <sender ref="{sid}"/>\n'
        f'  <receiver ref="{rid}"/>\n'
        f"  <body>{body}</body>\n"
        f"</message>"
    )


@dataclass
class BM25RAGConfig:
    top_k: int = 8
    neighbor_radius: int = 1


class BM25RAG:
    """
    Pure lexical RAG using BM25.
    - index(): builds BM25 over rendered message blocks
    - answer(): retrieve top_k + expand neighbors, then build prompt <= 3000 chars
    """

    def __init__(self, config: BM25RAGConfig | None = None):
        self.config = config or BM25RAGConfig()
        self._bm25 = None

        self.messages: list[StoryMessage] = []
        self.docs: list[str] = []
        self.docs_tokens: list[list[str]] = []

        self._bm25_import_error: Exception | None = None
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
            self._BM25Okapi = BM25Okapi
        except Exception as e:
            self._BM25Okapi = None
            self._bm25_import_error = e

    def index(self, messages: list[StoryMessage]) -> None:
        if self._BM25Okapi is None:
            raise RuntimeError(
                "Missing dependency 'rank-bm25'. Install it (pip install rank-bm25)."
            ) from self._bm25_import_error

        self.messages = [m for m in messages if (m.body or "").strip()]
        self.docs = [_render_message_xmlish(m) for m in self.messages]
        self.docs_tokens = [_tokenize(d) for d in self.docs]

        self._bm25 = self._BM25Okapi(self.docs_tokens)

    def _expand_with_neighbors(self, idxs: list[int], radius: int) -> list[int]:
        n = len(self.docs)
        chosen: set[int] = set()
        for i in idxs:
            for j in range(max(0, i - radius), min(n, i + radius + 1)):
                chosen.add(j)
        return sorted(chosen)

    def build_prompt(self, question: str, evidence_chunks: list[str], limit: int = 3000) -> str:
        chunks = list(evidence_chunks)

        while True:
            context = "\n\n".join(chunks)
            prompt = (
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

            try:
                enforce_prompt_limit(prompt, limit=limit)
                return prompt
            except Exception:
                if len(chunks) <= 1:
                    raise
                chunks.pop()

    def answer(self, question: str, llm: LLMClient) -> str:
        if self._bm25 is None:
            raise RuntimeError("Must call index() before answer().")

        q_tokens = _tokenize(question)
        scores = self._bm25.get_scores(q_tokens)

        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        seeds = ranked[: self.config.top_k]

        expanded = self._expand_with_neighbors(seeds, radius=self.config.neighbor_radius)
        evidence = [self.docs[i] for i in expanded]

        prompt = self.build_prompt(question, evidence, limit=3000)
        return llm.complete(prompt)
