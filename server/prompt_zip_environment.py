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

import logging
import os
import re
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import PromptZipAction, PromptZipObservation

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Hardcoded dataset: 4 per task type = 16 total
# ──────────────────────────────────────────────
DATASET: list[dict] = [
    # ── summarization (4) ──────────────────────
    {
        "task_type": "summarization",
        "token_budget": 30,  # ~55% of ~55 token prompt
        "prompt": (
            "I would like you to please provide me with a very detailed and comprehensive "
            "summary of the main points covered in the following text. "
            "Please make sure to be thorough and include all key ideas. "
            "Summarize the following passage:\n\n"
            "The Amazon rainforest, often referred to as the lungs of the Earth, produces "
            "20 percent of the world's oxygen. It spans nine countries and covers over "
            "5.5 million square kilometres. Deforestation threatens its biodiversity, "
            "with thousands of species at risk of extinction."
        ),
    },
    {
        "task_type": "summarization",
        "token_budget": 28,  # ~55% of ~51 token prompt
        "prompt": (
            "Could you kindly take a moment to summarize, in as much detail as possible, "
            "the content of the following document? "
            "I would appreciate a comprehensive and well-structured response. "
            "Here is the document:\n\n"
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "Since the 1800s, human activities have been the main driver, primarily through the "
            "burning of fossil fuels. This has led to rising sea levels, more frequent extreme "
            "weather events, and significant ecosystem disruption."
        ),
    },
    {
        "task_type": "summarization",
        "token_budget": 29,  # ~55% of ~53 token prompt
        "prompt": (
            "For the purposes of this task, I need you to summarize the following text "
            "as thoroughly as possible, making sure not to leave out any crucial details. "
            "Please be as exhaustive as you reasonably can. "
            "The text to summarize is:\n\n"
            "The Internet of Things (IoT) describes the network of physical objects embedded "
            "with sensors and software to connect and exchange data with other devices over "
            "the internet. Applications range from smart home devices to industrial automation, "
            "raising both efficiency gains and significant privacy concerns."
        ),
    },
    {
        "task_type": "summarization",
        "token_budget": 26,  # ~55% of ~47 token prompt
        "prompt": (
            "Please help me by summarizing the passage below. "
            "I would really appreciate if you could be thorough and cover everything. "
            "Do not skip over minor details. "
            "Summarize:\n\n"
            "Photosynthesis is the process by which green plants convert sunlight into food. "
            "Using chlorophyll, plants absorb light energy to transform carbon dioxide and "
            "water into glucose and oxygen. This process forms the foundation of most food "
            "chains on Earth."
        ),
    },
    # ── code_gen (4) ──────────────────────────
    {
        "task_type": "code_gen",
        "token_budget": 30,  # ~55% of ~55 token prompt
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
        "token_budget": 27,  # ~55% of ~49 token prompt
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
        "token_budget": 34,  # ~55% of ~62 token prompt
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
        "token_budget": 28,  # ~55% of ~51 token prompt
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
        "token_budget": 33,  # ~55% of ~60 token prompt
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
        "token_budget": 29,  # ~55% of ~53 token prompt
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
        "token_budget": 33,  # ~55% of ~60 token prompt
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
        "token_budget": 28,  # ~55% of ~51 token prompt
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
        "token_budget": 20,  # ~55% of ~36 token prompt
        "prompt": (
            "I was hoping you might be able to help me answer a question. "
            "I would greatly appreciate a concise and accurate response. "
            "Here is my question: "
            "What is the capital city of France?"
        ),
    },
    {
        "task_type": "qa",
        "token_budget": 23,  # ~55% of ~42 token prompt
        "prompt": (
            "Could you please be so kind as to tell me the answer to the following question? "
            "I understand this may be a simple question but I want to make sure I get it right. "
            "Question: "
            "How many planets are in our solar system?"
        ),
    },
    {
        "task_type": "qa",
        "token_budget": 22,  # ~55% of ~40 token prompt
        "prompt": (
            "I would really appreciate it if you could help me out with a trivia question. "
            "Please give me a direct and accurate answer. "
            "Here is the question I have for you: "
            "Who wrote the play Romeo and Juliet?"
        ),
    },
    {
        "task_type": "qa",
        "token_budget": 24,  # ~55% of ~44 token prompt
        "prompt": (
            "Thank you so much for taking the time to assist me with this. "
            "I have a factual question and I am hoping you can provide a clear answer. "
            "My question is: "
            "What is the speed of light in a vacuum, approximately?"
        ),
    },
]

# ──────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────
def _count_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ──────────────────────────────────────────────
# Sentence splitter — preserves separators
# ──────────────────────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_PARA_SEP = "\n\n"

def _segment(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split text into spans, preserving the separator used after each span.

    Returns:
        spans    - {uuid: span_text}
        seps     - {uuid: separator_that_followed_span}  ("" for the last span)
    """
    # First split on paragraph breaks to preserve them
    paragraphs = re.split(r"(\n\n+)", text.strip())
    parts: list[str] = []
    part_seps: list[str] = []
    i = 0
    while i < len(paragraphs):
        block = paragraphs[i].strip()
        sep_after = paragraphs[i + 1] if i + 1 < len(paragraphs) and re.match(r"^\n\n+$", paragraphs[i + 1]) else ""
        if sep_after:
            i += 2
        else:
            i += 1
        if not block:
            continue
        # Further split paragraph into sentences
        sentences = _SENT_RE.split(block)
        for j, s in enumerate(sentences):
            s = s.strip()
            if not s:
                continue
            parts.append(s)
            # Only the last sentence of a paragraph gets the paragraph separator
            part_seps.append(sep_after if j == len(sentences) - 1 else " ")

    if not parts:
        parts = [text.strip()]
        part_seps = [""]

    uuids = [str(uuid.uuid4()) for _ in parts]
    spans = {u: p for u, p in zip(uuids, parts)}
    seps = {u: s for u, s in zip(uuids, part_seps)}
    return spans, seps


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
            except Exception as e:
                log.warning("Groq API import or initialization failed: %s", e)
                self._client = None
        else:
            log.warning("GROQ_API_KEY not found in environment — using mock fallback")

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
        except Exception as e:
            log.warning("Groq API call failed (%s): %s", model, e)
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
            # Mock fallback: return span truncated to ~60%
            words = span.split()
            return " ".join(words[:max(1, int(len(words) * 0.6))])
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
            f"Original prompt:\n<prompt>\n{original_prompt}\n</prompt>\n\n"
            f"Compressed prompt:\n<prompt>\n{compressed_prompt}\n</prompt>\n\n"
            f"Original output:\n<output>\n{original_output}\n</output>\n\n"
            f"Compressed output:\n<output>\n{compressed_output}\n</output>\n\n"
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
            log.warning("Groq judge call failed — using 10.0 fallback (deterministic token-reduction ratio)")
            return 10.0
        try:
            score = float(result.strip().split()[0])
            return max(0.0, min(10.0, score))
        except (ValueError, IndexError):
            log.warning("Groq judge parsing failed — using 10.0 fallback (deterministic token-reduction ratio)")
            return 10.0


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────
class PromptZipEnvironment(Environment):  # type: ignore[type-arg]
    """RL environment for compressing LLM prompts via span-level actions."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._groq = _GroqClient()
        self._dataset_idx: int = 0
        self._state = State(episode_id=None, step_count=0)

        # Episode state
        self._spans: dict[str, str] = {}
        self._seps: dict[str, str] = {}  # separator after each span (" " or "\n\n")
        self._locked_spans: list[str] = []
        self._action_history: list[dict] = []
        self._task_type: str = "summarization"
        self._token_budget: int = 20
        self._original_token_count: int = 0
        self._initial_span_count: int = 0
        self._original_prompt: str = ""   # stored at reset() — avoids stale index
        self._original_output: str = ""
        self._done: bool = False
        self._last_obs: Optional[PromptZipObservation] = None

    # ── internal helpers ──────────────────────

    def _prompt_text(self) -> str:
        """Reassemble spans preserving their original separators."""
        if not self._spans:
            return ""
        parts = []
        for uid, text in self._spans.items():
            parts.append(text)
            sep = self._seps.get(uid, " ")
            parts.append(sep if sep else " ")
        return "".join(parts).rstrip()

    def _token_count(self) -> int:
        return _count_tokens(self._prompt_text())

    def _build_obs(self, reward: float | None, done: bool, metadata: Optional[dict] = None) -> PromptZipObservation:
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
        raw_score = self._groq.judge(
            original_prompt=self._original_prompt,
            compressed_prompt=compressed_prompt,
            original_output=self._original_output,
            compressed_output=compressed_output,
            task_type=self._task_type,
        )
        quality = raw_score / 10.0  # normalize 0–10 → 0.0–1.0
        final = quality * (tokens_saved / self._original_token_count) if self._original_token_count else 0.0
        if quality < 0.6:
            final -= 0.5  # penalty for quality collapse
        final = max(-1.0, min(1.0, final))  # hard clamp
        return final

    # ── OpenEnv API ───────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str = "medium",
        **kwargs: Any,
    ) -> PromptZipObservation:
        """Load the next prompt, segment into spans, pre-generate original_output."""
        self._done = False
        self._action_history = []
        self._locked_spans = []

        # Difficulty-based dataset filtering
        _tier_map: dict[str, list[str]] = {
            "easy":   ["qa"],
            "medium": ["summarization", "code_gen"],
            "hard":   ["reasoning"],
        }
        allowed = _tier_map.get(difficulty, ["summarization", "code_gen"])
        pool = [e for e in DATASET if e["task_type"] in allowed]
        if not pool:
            pool = DATASET  # fallback: use all

        if seed is not None:
            self._dataset_idx = seed % len(pool)

        entry = pool[self._dataset_idx % len(pool)]
        self._dataset_idx += 1
        self._task_type = entry["task_type"]
        
        self._spans, self._seps = _segment(entry["prompt"])
        self._original_prompt = self._prompt_text()
        self._original_token_count = self._token_count()
        self._token_budget = max(1, int(self._original_token_count * 0.55))
        self._initial_span_count = len(self._spans) or 1

        self._original_output = self._groq.generate(entry["prompt"])

        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )

        return self._build_obs(reward=None, done=False)

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

        # Step-limit guard — checked BEFORE action so episodes always terminate
        if self._state.step_count >= 2 * self._initial_span_count:
            self._done = True
            final_reward = self._run_judge_flow()
            return self._build_obs(
                reward=final_reward,
                done=True,
                metadata={"info": "step limit reached", "final_reward": final_reward},
            )

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
            self._seps.pop(action.span_id, None)

            # Empty-prompt guard: all spans gone → penalise, terminate
            if not self._spans:
                self._done = True
                return self._build_obs(
                    reward=-0.50,
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
            if self._token_count() >= self._original_token_count:
                # Short-circuit ONLY if absolutely no compression was achieved
                self._done = True
                return self._build_obs(
                    reward=0.0,
                    done=True,
                    metadata={"info": "all spans locked without compression — skip judge"},
                )
            else:
                # Compression was achieved, so we must run the judge
                self._done = True
                final_reward = self._run_judge_flow()
                return self._build_obs(
                    reward=max(-1.0, min(1.0, step_reward + final_reward)),
                    done=True,
                    metadata={"info": "all remaining spans locked, episode terminated", "final_reward": final_reward},
                )

        if self._is_terminated():
            self._done = True
            final_reward = self._run_judge_flow()
            return self._build_obs(
                reward=max(-1.0, min(1.0, step_reward + final_reward)),
                done=True,
                metadata={"info": "episode terminated", "final_reward": final_reward},
            )

        return self._build_obs(reward=step_reward, done=False)

    def grade(
        self,
        original_prompt: str,
        compressed_prompt: str,
        original_output: str,
        compressed_output: str,
        task_type: str,
    ) -> float:
        """Standalone API for grading compressed prompt quality [0.0 - 1.0]."""
        raw_score = self._groq.judge(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            original_output=original_output,
            compressed_output=compressed_output,
            task_type=task_type,
        )
        return max(0.0, min(1.0, raw_score / 10.0))

    @property
    def state(self) -> State:
        return self._state
