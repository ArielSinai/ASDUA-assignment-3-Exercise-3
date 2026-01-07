class PromptTooLongError(ValueError):
    pass


def enforce_prompt_limit(prompt: str, limit: int = 3000) -> None:
    if prompt is None:
        raise TypeError("prompt must not be None")
    if len(prompt) > limit:
        raise PromptTooLongError(
            f"Prompt too long: {len(prompt)} characters (limit={limit})."
        )
