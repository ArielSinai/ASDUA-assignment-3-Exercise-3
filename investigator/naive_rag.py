from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from investigator.prompt_limit import enforce_prompt_limit
from investigator.story_loader import StoryMessage


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def chunk_messages(messages: list[StoryMessage], max_chars: int = 900) -> list[str]:
    """
    Creates chunks of story text. Each chunk is a concatenation of message lines.
    We keep chunks relatively small so we can fit multiple chunks in a 3000-char prompt.
    """
    chunks: list[str] = []
    current: list[str] = []
    cur_len = 0

    for m in messages:
        line = f"[{m.ts}] {m.sender_name or m.sender_id}: {m.body}".strip()
        if not line:
            continue

        if cur_len + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            cur_len = len(line) + 1
        else:
            current.append(line)
            cur_len += len(line) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


def _cosine_sim_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return m @ q


@dataclass
class RetrievalResult:
    chunks: list[str]
    scores: list[float]


class NaiveRAG:
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, messages: list[StoryMessage], chunk_chars: int = 900) -> None:
        self.chunks = chunk_messages(messages, max_chars=chunk_chars)
        embs = self.model.encode(self.chunks, convert_to_numpy=True, normalize_embeddings=False)
        self.embeddings = embs.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if self.embeddings is None:
            raise RuntimeError("RAG not indexed. Call index() first.")

        q_vec = self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        sims = _cosine_sim_matrix(q_vec, self.embeddings)  
        idx = np.argsort(-sims)[:top_k]

        return RetrievalResult(
            chunks=[self.chunks[i] for i in idx],
            scores=[float(sims[i]) for i in idx],
        )

    def build_prompt(self, question: str, evidence_chunks: list[str], limit: int = 3000) -> str:
        """
        Build a prompt that fits within the character limit.
        If too long, we progressively drop the least relevant chunks (end of list).
        """
        header = (
            "You are AI Investigator 1.0.\n"
            "Answer the user's question about the story.\n"
            "Rules:\n"
            "1) If you know the answer, respond with: <ANSWER> and then EVIDENCE lines quoted from the story.\n"
            "2) Evidence must be direct quotes.\n"
            "3) If not conclusive, say why (not in story / ambiguous / conflicting).\n\n"
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
        retrieved = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, retrieved.chunks, limit=3000)
        return llm.complete(prompt)
