"""
Direct environment tests — no HTTP server required.
Run: (from prompt_zip_env/ directory)
    source ../.venv/bin/activate
    python -m pytest server/test_prompt_zip_env.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from server.prompt_zip_environment import PromptZipEnvironment
from models import PromptZipAction
from openenv.core.env_server.types import State


@pytest.fixture
def env():
    e = PromptZipEnvironment()
    e.reset()
    return e


# ── 1. reset() returns valid observation ─────────────────────────────────────

def test_reset_returns_valid_obs():
    env = PromptZipEnvironment()
    obs = env.reset()
    assert isinstance(obs.prompt_text, str) and obs.prompt_text
    assert isinstance(obs.spans, dict) and len(obs.spans) >= 1
    assert isinstance(obs.token_count, int) and obs.token_count > 0
    assert isinstance(obs.task_type, str)
    assert isinstance(obs.token_budget, int) and obs.token_budget > 0
    assert isinstance(obs.action_history, list) and len(obs.action_history) == 0
    assert isinstance(obs.locked_spans, list) and len(obs.locked_spans) == 0
    assert obs.done is False
    assert obs.reward is not None


# ── 2. elide decreases token_count and returns positive reward ───────────────

def test_elide_decreases_tokens():
    env = PromptZipEnvironment()
    env._groq._client = None  # force mock — hermetic regardless of GROQ_API_KEY
    obs = env.reset()

    # Need at least 2 spans to avoid empty-prompt guard
    if len(obs.spans) < 2:
        pytest.skip("fewer than 2 spans — prompt not splittable")

    span_id = list(obs.spans.keys())[0]
    before = obs.token_count
    obs2 = env.step(PromptZipAction(action_type="elide", span_id=span_id))
    assert obs2.token_count < before
    # With mock judge (score=7.0), final_reward is always non-negative
    assert obs2.reward is not None and obs2.reward >= 0


# ── 3. rephrase decreases token_count even with mock fallback ────────────────

def test_rephrase_mock_decreases_tokens():
    env = PromptZipEnvironment()
    # Force mock by clearing client
    env._groq._client = None
    obs = env.reset()

    # Find a span long enough for mock to shorten
    long_id = None
    for sid, text in obs.spans.items():
        if len(text.split()) >= 3:
            long_id = sid
            break
    if long_id is None:
        pytest.skip("no span with >= 3 words found")

    before = obs.token_count
    obs2 = env.step(PromptZipAction(action_type="rephrase", span_id=long_id))
    # Mock returns first 60% of words — token_count should not increase
    assert obs2.token_count <= before
    assert obs2.reward is not None


# ── 4. preserve adds span to locked_spans, prompt unchanged ──────────────────

def test_preserve_locks_span(env):
    obs = env._last_obs
    span_id = list(obs.spans.keys())[0]
    original_text = obs.prompt_text

    obs2 = env.step(PromptZipAction(action_type="preserve", span_id=span_id))
    assert span_id in obs2.locked_spans
    assert obs2.prompt_text == original_text


# ── 5. step on locked span returns negative reward, no mutation ───────────────

def test_locked_span_penalised(env):
    obs = env._last_obs
    span_id = list(obs.spans.keys())[0]
    env.step(PromptZipAction(action_type="preserve", span_id=span_id))
    obs_before = env._last_obs

    obs2 = env.step(PromptZipAction(action_type="elide", span_id=span_id))
    assert obs2.reward is not None and obs2.reward <= 0
    assert obs2.prompt_text == obs_before.prompt_text


# ── 6. step after done=True returns done=True, no crash ──────────────────────

def test_step_after_done():
    env = PromptZipEnvironment()
    env.reset()
    env._done = True  # force termination
    obs = env.step(PromptZipAction(action_type="elide", span_id="nonexistent"))
    assert obs.done is True


# ── 7. elide all spans triggers empty-prompt guard ────────────────────────────

def test_empty_prompt_guard():
    env = PromptZipEnvironment()
    # Force a single-span prompt
    import uuid as _uuid
    env._done = False
    env._spans = {str(_uuid.uuid4()): "Only span."}
    env._locked_spans = []
    env._action_history = []
    env._original_token_count = 2
    env._initial_span_count = 1
    from openenv.core.env_server.types import State as _State
    env._state = _State(episode_id="test", step_count=0)

    span_id = list(env._spans.keys())[0]
    obs = env.step(PromptZipAction(action_type="elide", span_id=span_id))
    assert obs.done is True
    assert obs.reward is not None and obs.reward <= 0


# ── 8. full episode reaches done=True with numeric final reward ───────────────

def test_full_episode_terminates():
    env = PromptZipEnvironment()
    obs = env.reset()
    max_iters = 50
    for _ in range(max_iters):
        if obs.done:
            break
        available = [sid for sid in obs.spans if sid not in obs.locked_spans]
        if not available:
            break
        obs = env.step(PromptZipAction(action_type="elide", span_id=available[0]))
    assert obs.done is True
    assert isinstance(obs.reward, (int, float))


# ── 9. state property returns valid State ─────────────────────────────────────

def test_state_property(env):
    s = env.state
    assert isinstance(s, State)
    assert s.episode_id is not None
    assert isinstance(s.step_count, int) and s.step_count >= 0
