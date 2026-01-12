from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol
import re

import numpy as np

from investigator.prompt_limit import enforce_prompt_limit
from investigator.story_loader import StoryMessage


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...



def render_message_xmlish(m: StoryMessage) -> str:
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
    Naive hybrid RAG:
    - sentence-transformers embeddings
    - simple lexical overlap
    - neighbor expansion
    """

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self._embedder_import_error: Exception | None = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embed_model)
        except Exception as e:
            self.model = None
            self._embedder_import_error = e

        self.messages: list[StoryMessage] = []
        self.docs: list[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, messages: list[StoryMessage]) -> None:
        if self.model is None:
            raise RuntimeError(
                "Missing dependency 'sentence-transformers'"
            ) from self._embedder_import_error

        self.messages = list(messages)
        self.docs = [render_message_xmlish(m) for m in self.messages]
        embs = self.model.encode(self.docs, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = embs.astype(np.float32)

    def _hybrid_scores(self, query: str) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Call index() first")

        q_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_vec = _normalize_vec(q_vec.astype(np.float32))

        cos = self.embeddings @ q_vec
        kws = extract_keywords(query)
        if not kws:
            return cos

        lex = np.array([_lexical_score(kws, d) for d in self.docs], dtype=np.float32)
        return 0.8 * cos + 0.2 * lex

    def _expand_with_neighbors(self, hit_indices: Iterable[int], radius: int = 1) -> list[str]:
        wanted: set[int] = set()
        n = len(self.docs)
        for i in hit_indices:
            for j in range(max(0, i - radius), min(n, i + radius + 1)):
                wanted.add(j)
        return [self.docs[i] for i in range(n) if i in wanted]

    def _pack_excerpts(self, docs: list[str], max_chunk_chars: int = 800) -> list[str]:
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0

        for d in docs:
            add_len = len(d) + 2
            if cur and cur_len + add_len > max_chunk_chars:
                chunks.append("\n\n".join(cur))
                cur = [d]
                cur_len = add_len
            else:
                cur.append(d)
                cur_len += add_len

        if cur:
            chunks.append("\n\n".join(cur))
        return chunks

    def build_prompt(self, question: str, evidence_chunks: list[str], limit: int = 3000) -> str:
        header = (
            "You are AI Investigator 1.0.\n"
            "Answer the user's question about the story.\n\n"
            "Output format (exactly):\n"
            "ANSWER: <short answer or 'Not conclusive.'>\n"
            "EVIDENCE:\n"
            "- <direct quote 1>\n"
            "- <direct quote 2>\n"
            "REASON: <why not conclusive, if applicable>\n\n"
        )

        chunks = list(evidence_chunks)
        while True:
            prompt = (
                header
                + f"QUESTION:\n{question}\n\n"
                + "STORY EXCERPTS:\n"
                + "\n\n---\n\n".join(chunks)
            )
            try:
                enforce_prompt_limit(prompt, limit=limit)
                return prompt
            except Exception:
                if len(chunks) <= 1:
                    raise
                chunks.pop()

    def answer(self, question: str, llm: LLMClient, top_k: int = 5) -> str:
        scores = self._hybrid_scores(question)
        hit_idx = np.argsort(-scores)[:top_k]

        expanded = self._expand_with_neighbors(hit_idx.tolist(), radius=1)
        packed = self._pack_excerpts(expanded)

        prompt = self.build_prompt(question, packed, limit=3000)
        return llm.complete(prompt)
