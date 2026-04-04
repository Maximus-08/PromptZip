"""Additional tests to improve coverage."""

import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.prompt_zip_environment import PromptZipEnvironment, _segment, _GroqClient
from models import PromptZipAction

@pytest.fixture
def env():
    e = PromptZipEnvironment()
    e.reset()
    return e


# ── 10. all spans locked terminates episode ────────────────────────────────────

def test_all_spans_locked_terminates(env):
    obs = env._last_obs
    for span_id in obs.spans:
        obs = env.step(PromptZipAction(action_type="preserve", span_id=span_id))
    assert obs.done is True
    assert obs.reward == 0.0

# ── 11. seed is respected ───────────────────────────────────────────────────

def test_seed_modifies_dataset_idx():
    # With difficulty="medium" there are 8 prompts in the pool (4 summarization + 4 code_gen)
    # seed=5 → idx = 5 % 8 = 5, then +1 after load = 6
    env1 = PromptZipEnvironment()
    env1.reset(seed=5, difficulty="medium")
    assert env1._dataset_idx == (5 % 8) + 1
    # seed=10 → idx = 10 % 8 = 2, then +1 = 3
    env2 = PromptZipEnvironment()
    env2.reset(seed=10, difficulty="medium")
    assert env2._dataset_idx == (10 % 8) + 1

# ── 12. _segment edge cases ─────────────────────────────────────────────────

def test_segment_empty_strings():
    # Test empty or whitespace
    spans, seps = _segment("   ")
    assert len(spans) == 1
    assert list(spans.values())[0] == ""
    
    # Test consecutive newlines without content
    spans, seps = _segment("\n\n\n\n")
    assert len(spans) == 1
    assert list(spans.values())[0] == ""

# ── 13. Groq client judge and mock behavior ─────────────────────────────────

def test_groq_client_judge():
    client = _GroqClient()
    # Force mock
    client._client = None
    
    # Empty chat result returns fallback score 10.0
    score = client.judge("orig", "comp", "orig_out", "comp_out", "summarization")
    assert score == 10.0

def test_groq_chat_exception(monkeypatch):
    import groq
    client = _GroqClient()
    
    # Mock a real client that raises an exception on chat.completions.create
    class MockChat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                raise Exception("Simulated network failure")
    
    class MockClient:
        chat = MockChat()
        
    client._client = MockClient()
    
    # Rewrite should handle exception and return empty string
    res = client._chat("test-model", [])
    assert res == ""
    
    # This empty string causes rewrite to use identity mock
    res_rewrite = client.rewrite("hello world")
    assert res_rewrite == "hello"

# ── 14. final_reward branch ─────────────────────────────────────────────────
def test_final_reward_negative():
    env = PromptZipEnvironment()
    env._groq._client = None
    # Patch judge to return a low score to trigger final -= 5.0
    env._groq.judge = lambda *args, **kwargs: 2.0
    env.reset()
    
    # Setup some dummy compression to make tokens_saved > 0
    env._original_token_count = 10
    span_id = list(env._spans.keys())[0]
    # Delete a span to reduce token count
    from models import PromptZipAction
    env.step(PromptZipAction(action_type="elide", span_id=span_id))
    
    # Force termination
    env._token_budget = 1000
    obs = env.step(PromptZipAction(action_type="elide", span_id=list(env._spans.keys())[0]))
    
    # Final reward should be negative due to the cliff
    assert obs.done is True
    assert obs.reward < 0
