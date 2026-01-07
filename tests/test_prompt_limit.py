import pytest
from investigator.prompt_limit import enforce_prompt_limit, PromptTooLongError

def test_allows_prompt_under_limit():
    s = "a" * 3000
    enforce_prompt_limit(s, limit=3000)  

def test_raises_when_prompt_over_limit():
    s = "a" * 3001
    with pytest.raises(PromptTooLongError):
        enforce_prompt_limit(s, limit=3000)

def test_error_message_includes_lengths():
    s = "a" * 3001
    with pytest.raises(PromptTooLongError) as e:
        enforce_prompt_limit(s, limit=3000)
    msg = str(e.value)
    assert "3001" in msg
    assert "3000" in msg
