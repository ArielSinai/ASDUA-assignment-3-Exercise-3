from __future__ import annotations

import os

from google import genai

from investigator.prompt_limit import enforce_prompt_limit


class GeminiLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Put it in your .env file or set it as an environment variable."
            )

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        self.client = genai.Client(api_key=self.api_key)

    def complete(self, prompt: str) -> str:
        enforce_prompt_limit(prompt, limit=3000)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return (resp.text or "").strip()
