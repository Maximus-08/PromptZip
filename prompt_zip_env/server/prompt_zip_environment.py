"""
Core environment logic for PromptZip RL.

reset() loads a bloated prompt, segments it into UUID-keyed spans,
pre-generates original_output via Target LLM (or mock), and returns
the initial PromptZipObservation.

step(action) applies one of: elide, rephrase, preserve to the targeted
span, computes intermediate reward, and checks termination conditions.
At the end of an episode (budget met or max_steps hit) calls the Judge
for final reward. Short-circuits all-locked or empty-prompt states.
"""

import os
import re
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import PromptZipAction, PromptZipObservation

# ──────────────────────────────────────────────
# Hardcoded dataset: 4 per task type = 16 total
# ──────────────────────────────────────────────
DATASET: list[dict] = [
    # ── summarization (4) ──────────────────────
    {
        "task_type": "summarization",
        "token_budget": 15,
        "prompt": (
            "I would like you to please provide me with a very detailed and comprehensive "
            "summary of the main points covered in the following text. "
            "Please make sure to be thorough and include all key ideas. "
            "Summarize the following passage:\n\n"
            "{content}"
        ),
    },
    {
        "task_type": "summarization",
        "token_budget": 12,
        "prompt": (
            "Could you kindly take a moment to summarize, in as much detail as possible, "
            "the content of the following document? "
            "I would appreciate a comprehensive and well-structured response. "
            "Here is the document:\n\n"
            "{content}"
        ),
    },
    {
        "task_type": "summarization",
        "token_budget": 14,
        "prompt": (
            "For the purposes of this task, I need you to summarize the following text "
            "as thoroughly as possible, making sure not to leave out any crucial details. "
            "Please be as exhaustive as you reasonably can. "
            "The text to summarize is:\n\n"
            "{content}"
        ),
    },
    {
        "task_type": "summarization",
        "token_budget": 10,
        "prompt": (
            "Please help me by summarizing the passage below. "
            "I would really appreciate if you could be thorough and cover everything. "
            "Do not skip over minor details. "
            "Summarize:\n\n"
            "{content}"
        ),
    },
    # ── code_gen (4) ──────────────────────────
    {
        "task_type": "code_gen",
        "token_budget": 20,
        "prompt": (
            "I am hoping that you could help me with a Python programming task. "
            "I would really appreciate it if you could write a function that "
            "takes a list of integers as input and returns the sum of all even numbers. "
            "Please make sure your code is clean and well-commented. "
            "Also, if it is not too much trouble, please include a brief docstring."
        ),
    },
    {
        "task_type": "code_gen",
        "token_budget": 18,
        "prompt": (
            "Could you be so kind as to write a Python script for me? "
            "I need a function called `flatten` that takes a nested list "
            "and returns a flat list. "
            "If possible, please handle arbitrarily deep nesting. "
            "Feel free to add type hints if you think they would be helpful."
        ),
    },
    {
        "task_type": "code_gen",
        "token_budget": 22,
        "prompt": (
            "I am working on a project and I would be extremely grateful "
            "if you could help me write a Python class for a simple stack data structure. "
            "The class should support push, pop, and peek operations. "
            "Please make sure to handle edge cases like popping from an empty stack. "
            "Any additional comments or docstrings would also be very welcome."
        ),
    },
    {
        "task_type": "code_gen",
        "token_budget": 16,
        "prompt": (
            "I was wondering if perhaps you might be able to assist me with writing "
            "a Python function that checks whether a given string is a palindrome. "
            "It would be great if you could handle both uppercase and lowercase letters. "
            "Please include any edge case handling you think might be relevant."
        ),
    },
    # ── reasoning (4) ─────────────────────────
    {
        "task_type": "reasoning",
        "token_budget": 18,
        "prompt": (
            "I would like you to please think through the following problem "
            "step by step, being careful to show all of your reasoning. "
            "Do not skip any steps. "
            "Here is the problem: "
            "If a train leaves station A at 9 AM travelling at 60 mph, "
            "and another train leaves station B at 10 AM travelling at 80 mph toward station A, "
            "and the stations are 210 miles apart, when do they meet?"
        ),
    },
    {
        "task_type": "reasoning",
        "token_budget": 15,
        "prompt": (
            "Could you kindly reason through the following logical puzzle step by step? "
            "Please show every inference so your reasoning is easy to follow. "
            "Puzzle: "
            "All plants need water. "
            "Cacti are plants. "
            "Does a cactus need water? "
            "Explain your reasoning thoroughly."
        ),
    },
    {
        "task_type": "reasoning",
        "token_budget": 20,
        "prompt": (
            "I am presenting you with a multi-step arithmetic problem "
            "and I would be grateful if you could walk me through each step carefully. "
            "Please be explicit and do not skip intermediate calculations. "
            "Problem: "
            "A store sells apples for $0.50 each and oranges for $0.75 each. "
            "If a customer buys 4 apples and 3 oranges, how much do they spend in total?"
        ),
    },
    {
        "task_type": "reasoning",
        "token_budget": 14,
        "prompt": (
            "Please take the time to carefully and thoroughly reason through the following question. "
            "Show your work and explain each logical step. "
            "Question: "
            "Is it possible for a triangle to have two right angles? "
            "Please justify your answer with geometric reasoning."
        ),
    },
    # ── qa (4) ────────────────────────────────
    {
        "task_type": "qa",
        "token_budget": 10,
        "prompt": (
            "I was hoping you might be able to help me answer a question. "
            "I would greatly appreciate a concise and accurate response. "
            "Here is my question: "
            "What is the capital city of France?"
        ),
    },
    {
        "task_type": "qa",
        "token_budget": 12,
        "prompt": (
            "Could you please be so kind as to tell me the answer to the following question? "
            "I understand this may be a simple question but I want to make sure I get it right. "
            "Question: "
            "How many planets are in our solar system?"
        ),
    },
    {
        "task_type": "qa",
        "token_budget": 11,
        "prompt": (
            "I would really appreciate it if you could help me out with a trivia question. "
            "Please give me a direct and accurate answer. "
            "Here is the question I have for you: "
            "Who wrote the play Romeo and Juliet?"
        ),
    },
    {
        "task_type": "qa",
        "token_budget": 13,
        "prompt": (
            "Thank you so much for taking the time to assist me with this. "
            "I have a factual question and I am hoping you can provide a clear answer. "
            "My question is: "
            "What is the speed of light in a vacuum, approximately?"
        ),
    },
]

# ──────────────────────────────────────────────
# Token counting (no tiktoken — no C-extension)
# ──────────────────────────────────────────────
def _count_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ──────────────────────────────────────────────
# Sentence splitter
# ──────────────────────────────────────────────
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n\n+")

def _segment(text: str) -> dict[str, str]:
    """Split text into sentences and return a {uuid: sentence} dict."""
    raw = _SPLIT_RE.split(text.strip())
    parts = [s.strip() for s in raw if s.strip()]
    if not parts:
        parts = [text.strip()]
    return {str(uuid.uuid4()): part for part in parts}


# ──────────────────────────────────────────────
# Groq client wrapper with timeout + fallback
# ──────────────────────────────────────────────
class _GroqClient:
    def __init__(self) -> None:
        self._client: Any = None
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            try:
                import groq  # type: ignore
                self._client = groq.Groq(api_key=api_key)
            except Exception:
                self._client = None

    def _chat(self, model: str, messages: list[dict], timeout: float = 10.0) -> str:
        if self._client is None:
            return ""  # triggers fallback at call site
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    def rewrite(self, span: str) -> str:
        result = self._chat(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following to convey the same meaning in fewer words. "
                        "Return only the rewritten text, no explanation.\n\n"
                        + span
                    ),
                }
            ],
        )
        if not result:
            # Mock: first 60% of words, minimum 1
            words = span.split()
            keep = max(1, int(len(words) * 0.6))
            return " ".join(words[:keep])
        return result

    def generate(self, prompt: str) -> str:
        result = self._chat(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return result if result else "[mock output]"

    def judge(
        self,
        original_prompt: str,
        compressed_prompt: str,
        original_output: str,
        compressed_output: str,
        task_type: str,
    ) -> float:
        judge_prompt = (
            f"You are evaluating prompt compression quality.\n\n"
            f"Task type: {task_type}\n\n"
            f"Original prompt:\n{original_prompt}\n\n"
            f"Compressed prompt:\n{compressed_prompt}\n\n"
            f"Original output:\n{original_output}\n\n"
            f"Compressed output:\n{compressed_output}\n\n"
            "Score the compressed output on a scale from 0 to 10 based on:\n"
            "1. Semantic preservation (0-2.5)\n"
            "2. Factual accuracy (0-2.5)\n"
            "3. Completeness (0-2.5)\n"
            "4. Coherence (0-2.5)\n\n"
            "Reply with ONLY a single number between 0 and 10."
        )
        result = self._chat(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": judge_prompt}],
        )
        if not result:
            return 7.0  # mock fallback
        try:
            score = float(result.strip().split()[0])
            return max(0.0, min(10.0, score))
        except (ValueError, IndexError):
            return 7.0


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
class PromptZipEnvironment(Environment):  # type: ignore[type-arg]
    """RL environment for compressing LLM prompts via span-level actions."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._groq = _GroqClient()
        self._dataset = DATASET
        self._dataset_idx: int = 0
        self._state = State(episode_id=None, step_count=0)

        # Episode state
        self._spans: dict[str, str] = {}
        self._locked_spans: list[str] = []
        self._action_history: list[dict] = []
        self._task_type: str = "summarization"
        self._token_budget: int = 20
        self._original_token_count: int = 0
        self._initial_span_count: int = 0
        self._original_output: str = ""
        self._done: bool = False
        self._last_obs: Optional[PromptZipObservation] = None

    # ── internal helpers ──────────────────────

    def _prompt_text(self) -> str:
        return " ".join(self._spans.values())

    def _token_count(self) -> int:
        return _count_tokens(self._prompt_text())

    def _build_obs(self, reward: float, done: bool, metadata: Optional[dict] = None) -> PromptZipObservation:
        obs = PromptZipObservation(
            done=done,
            reward=reward,
            metadata=metadata or {},
            prompt_text=self._prompt_text(),
            spans=dict(self._spans),
            token_count=self._token_count(),
            task_type=self._task_type,
            token_budget=self._token_budget,
            action_history=list(self._action_history),
            locked_spans=list(self._locked_spans),
        )
        self._last_obs = obs
        return obs

    def _is_terminated(self) -> bool:
        tc = self._token_count()
        step_limit = self._state.step_count >= 2 * self._initial_span_count
        return tc <= self._token_budget or step_limit

    def _run_judge_flow(self) -> float:
        compressed_prompt = self._prompt_text()
        compressed_output = self._groq.generate(compressed_prompt)
        tokens_saved = self._original_token_count - self._token_count()
        quality = self._groq.judge(
            original_prompt=self._dataset[self._dataset_idx]["prompt"],
            compressed_prompt=compressed_prompt,
            original_output=self._original_output,
            compressed_output=compressed_output,
            task_type=self._task_type,
        )
        final = quality * (tokens_saved / self._original_token_count) if self._original_token_count else 0.0
        if quality < 6.0:
            final -= 5.0
        return final

    # ── OpenEnv API ───────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PromptZipObservation:
        """Load the next prompt, segment into spans, pre-generate original_output."""
        self._done = False
        self._action_history = []
        self._locked_spans = []

        entry = self._dataset[self._dataset_idx % len(self._dataset)]
        self._dataset_idx += 1
        self._task_type = entry["task_type"]
        self._token_budget = entry["token_budget"]

        self._spans = _segment(entry["prompt"])
        self._original_token_count = _count_tokens(entry["prompt"])
        self._initial_span_count = len(self._spans) or 1

        self._original_output = self._groq.generate(entry["prompt"])

        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )

        return self._build_obs(reward=0.0, done=False)

    def step(
        self,
        action: PromptZipAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PromptZipObservation:
        """Apply one compression action and return the updated observation."""
        # Post-done guard
        if self._done:
            return self._build_obs(reward=0.0, done=True, metadata={"info": "episode already done"})

        self._state.step_count += 1

        # Invalid or locked span guard
        if action.span_id not in self._spans or action.span_id in self._locked_spans:
            return self._build_obs(
                reward=-0.1,
                done=False,
                metadata={"info": f"invalid or locked span_id: {action.span_id}"},
            )

        prev_tokens = self._token_count()

        # ── Execute action ────────────────────
        if action.action_type == "elide":
            del self._spans[action.span_id]

            # Empty-prompt guard: all spans gone → penalise, terminate
            if not self._spans:
                self._done = True
                return self._build_obs(
                    reward=-5.0,
                    done=True,
                    metadata={"info": "all spans elided — destroying meaning"},
                )

        elif action.action_type == "rephrase":
            original_span = self._spans[action.span_id]
            rewritten = self._groq.rewrite(original_span)
            self._spans[action.span_id] = rewritten

        elif action.action_type == "preserve":
            self._locked_spans.append(action.span_id)

        self._action_history.append({"action_type": action.action_type, "span_id": action.span_id})

        new_tokens = self._token_count()
        step_reward = (prev_tokens - new_tokens) / self._original_token_count * 0.5 if self._original_token_count else 0.0

        # ── Termination checks ────────────────
        all_locked = set(self._locked_spans) >= set(self._spans.keys())
        if all_locked:
            # Short-circuit: no compression achieved → 0 reward, skip judge
            self._done = True
            return self._build_obs(
                reward=0.0,
                done=True,
                metadata={"info": "all spans locked — no compression"},
            )

        if self._is_terminated():
            self._done = True
            final_reward = self._run_judge_flow()
            return self._build_obs(
                reward=step_reward + final_reward,
                done=True,
                metadata={"info": "episode terminated", "final_reward": final_reward},
            )

        return self._build_obs(reward=step_reward, done=False)

    @property
    def state(self) -> State:
        return self._state
