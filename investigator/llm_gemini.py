from __future__ import annotations

import os
import time
from typing import Optional

from google import genai

from investigator.prompt_limit import enforce_prompt_limit


class GeminiLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        max_retries: int = 3,
        base_sleep_seconds: float = 1.0,
        graceful_fallback: bool = True,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY"
            )

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=self.api_key)

        self.max_retries = max_retries
        self.base_sleep_seconds = base_sleep_seconds
        self.graceful_fallback = graceful_fallback

    def complete(self, prompt: str) -> str:
        enforce_prompt_limit(prompt, limit=3000)

        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return (resp.text or "").strip()

            except Exception as e:
                last_err = e
                msg = str(e)
                low = msg.lower()

                retryable = (
                    ("503" in msg)
                    or ("unavailable" in low)
                    or ("overloaded" in low)
                    or ("resource_exhausted" in low)  
                    or ("429" in msg)
                    or ("rate" in low and "limit" in low)
                    or ("timeout" in low)
                    or ("temporar" in low)  
                )

                if retryable and attempt < self.max_retries - 1:
                    sleep_s = self.base_sleep_seconds * (2**attempt)
                    time.sleep(sleep_s)
                    continue

                if not self.graceful_fallback:
                    raise

                return (
                    "ANSWER: Not conclusive.\n"
                    "EVIDENCE:\n"
                    "NO_RELEVANT_QUOTES\n"
                    "REASON: The LLM service is temporarily unavailable (rate limit / overload). "
                    f"Please try again. Details: {msg}\n"
                )

        if not self.graceful_fallback:
            raise last_err or RuntimeError("Gemini failed with unknown error")

        return (
            "ANSWER: Not conclusive.\n"
            "EVIDENCE:\n"
            "NO_RELEVANT_QUOTES\n"
            "REASON: The LLM service is temporarily unavailable. Please try again.\n"
        )
