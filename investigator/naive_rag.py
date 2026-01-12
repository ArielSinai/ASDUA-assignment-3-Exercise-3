from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol
import re

import numpy as np

from investigator.prompt_limit import enforce_prompt_limit
from investigator.story_loader import StoryMessage


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


# -----------------------------
# Rendering / tokenization
# -----------------------------


def render_message_xmlish(m: StoryMessage) -> str:
    """Render a single message in an XML-ish format similar to the exercise spec."""
    sid = m.sender_id or "unknown"
    rid = m.receiver_id or "unknown"
    body = (m.body or "").strip()
    ts = m.ts or ""
    return (
        f'<message id="{m.id}" ts="{ts}">\n'
        f'  <sender ref="{sid}"/>'
        + (f"  <!-- {m.sender_name} -->" if m.sender_name else "")
        + "\n"
        f'  <receiver ref="{rid}"/>'
        + (f"  <!-- {m.receiver_name} -->" if m.receiver_name else "")
        + "\n"
        f"  <body>{body}</body>\n"
        f"</message>"
    )


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "how", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "so", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
}


def extract_keywords(query: str, *, min_len: int = 3, max_terms: int = 8) -> list[str]:
    toks = [t for t in tokenize(query) if len(t) >= min_len and t not in _STOPWORDS]
    if not toks:
        return []

    seen: set[str] = set()
    uniq: list[str] = []
    for t in toks:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    return uniq[:max_terms]


@dataclass
class RetrievalResult:
    chunks: list[str]
    scores: list[float]


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _lexical_score(query_kws: list[str], doc_text: str) -> float:
    if not query_kws:
        return 0.0
    doc_low = (doc_text or "").lower()
    hits = sum(1 for kw in query_kws if kw in doc_low)
    return hits / float(len(query_kws))


class NaiveRAG:
    """
    Naive RAG:
    - Each message is a "doc" (XML-ish block)
    - Retrieve top_k by embedding similarity + small lexical overlap bonus
    - Expand with temporal neighbors (radius=1)
    - Build a prompt that MUST obey the prompt limit by progressively packing evidence
    - Output format matches the exercise spec (ANSWER / EVIDENCE blocks / REASON)
    """

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self._embed_model_name = embed_model
        self._embedder_import_error: Exception | None = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(embed_model)
        except Exception as e:
            self.model = None
            self._embedder_import_error = e

        self.messages: list[StoryMessage] = []
        self.docs: list[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, messages: list[StoryMessage], chunk_chars: int = 900) -> None:  # kept for API compat
        if self.model is None:
            raise RuntimeError(
                "Missing dependency 'sentence-transformers'. Install it with: pip install sentence-transformers"
            ) from self._embedder_import_error

        self.messages = list(messages)
        self.docs = [render_message_xmlish(m) for m in self.messages]
        embs = self.model.encode(self.docs, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = embs.astype(np.float32)

    def _hybrid_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("RAG not indexed. Call index() first.")

        q_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        q_vec = _normalize_vec(q_vec)
        cos = self.embeddings @ q_vec

        kws = extract_keywords(query)
        if not kws:
            return cos

        lex = np.array([_lexical_score(kws, d) for d in self.docs], dtype=np.float32)
        return (0.80 * cos) + (0.20 * lex)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        scores = self._hybrid_scores(query)
        idx = np.argsort(-scores)[:top_k]
        return RetrievalResult(
            chunks=[self.docs[i] for i in idx],
            scores=[float(scores[i]) for i in idx],
        )

    def _expand_with_neighbors(self, hit_indices: Iterable[int], *, radius: int = 1) -> list[str]:
        if not self.docs:
            return []

        wanted: set[int] = set()
        n = len(self.docs)
        for i in hit_indices:
            for j in range(max(0, i - radius), min(n, i + radius + 1)):
                wanted.add(j)

        # Keep chronological order
        return [self.docs[i] for i in range(n) if i in wanted]

    def _build_prompt(self, question: str, context: str, *, limit: int = 3000) -> str:
        prompt = (
            "You are AI Investigator 1.0.\n"
            "Answer the user's question using ONLY the provided story messages.\n"
            "If you don't know, explain why (not in story / ambiguous / conflicting / not conclusive).\n\n"
            "Return EXACTLY in this format:\n"
            "ANSWER: <short answer>\n"
            "EVIDENCE:\n"
            "<paste 1-5 exact <message ...> blocks from STORY MESSAGES>\n"
            "REASON: <why this evidence supports the answer OR why not conclusive>\n\n"
            "Rules:\n"
            "1) Evidence must be copied verbatim from STORY MESSAGES.\n"
            "2) If the answer is not conclusive, write: ANSWER: Not conclusive.\n"
            "   and EVIDENCE: NO_RELEVANT_QUOTES (if nothing supports an answer).\n"
            "3) Do NOT invent message ids or content.\n\n"
            f"QUESTION: {question}\n\n"
            "STORY MESSAGES:\n"
            f"{context}\n"
        )
        enforce_prompt_limit(prompt, limit=limit)
        return prompt

    def answer(self, question: str, llm: LLMClient, top_k: int = 5) -> str:
        if self.embeddings is None:
            raise RuntimeError("RAG not indexed. Call index() first.")

        scores = self._hybrid_scores(question)
        hit_idx = np.argsort(-scores)[:top_k]

        expanded_docs = self._expand_with_neighbors(hit_idx.tolist(), radius=1)

        # Progressive packing: add message blocks until the prompt hits the limit.
        selected_blocks: list[str] = []
        for block in expanded_docs:
            tentative = selected_blocks + [block]
            context = "\n\n".join(tentative)
            try:
                _ = self._build_prompt(question, context, limit=3000)
                selected_blocks.append(block)
            except Exception:
                break

        context = "\n\n".join(selected_blocks)
        try:
            prompt = self._build_prompt(question, context, limit=3000)
        except Exception:
            prompt = self._build_prompt(question, "", limit=3000)

        return llm.complete(prompt)
