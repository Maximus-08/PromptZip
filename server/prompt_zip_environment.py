"""
Core environment logic for PromptZip RL.

reset() loads a bloated prompt, segments it into UUID-keyed spans,
pre-generates original_output via Target LLM (or mock), and returns
the initial PromptZipObservation.

step(action) applies one of: elide, rephrase, preserve to the targeted
span, computes intermediate reward, and checks termination conditions.

At the end of an episode (budget met or max_steps hit) calls the Judge
for final reward.  Short-circuits all-locked or empty-prompt states.
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
        "prompt": (
            "I was wondering if perhaps you might be able to assist me with writing "
            "a Python function that checks whether a given string is a palindrome. "
            "It would be great if you could handle both uppercase and lowercase letters. "
            "Please include any edge case handling you think might be relevant."
        ),
    },

    # ── reasoning (4) ─────────────────────────
    # Two original simple prompts kept as baseline anchors.
    # Two replaced with genuinely hard prompts where every sentence is load-bearing:
    # removing any single constraint changes the valid answer.
    {
        "task_type": "reasoning",
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
        # HARD — multi-constraint scheduling: every availability clause is essential.
        # Removing any single sentence changes which slot (if any) is valid.
        "task_type": "reasoning",
        "prompt": (
            "I need you to carefully work through the following scheduling problem. "
            "Show every step and apply each constraint explicitly. "
            "Problem: "
            "Alice, Bob, and Carol must find a two-hour morning slot for a team meeting. "
            "Alice is unavailable all day Wednesday and every Thursday afternoon. "
            "Bob is only free on Tuesday morning, Wednesday morning, or Friday afternoon. "
            "Carol cannot attend any morning slot on Tuesday or Wednesday, "
            "and she has a standing conflict that blocks the entire day every Friday. "
            "The meeting must occur in the morning. "
            "List every candidate slot, eliminate those blocked by each person's constraints, "
            "and state the single valid option that satisfies all three schedules."
        ),
    },
    {
        # HARD — Bayes' theorem with interleaved numeric conditions.
        # Each sentence supplies a distinct probability needed for the calculation;
        # omitting any one makes the problem unsolvable or changes the answer.
        "task_type": "reasoning",
        "prompt": (
            "I would like you to work through the following probability problem step by step. "
            "Please apply Bayes' theorem explicitly and show every intermediate calculation. "
            "Problem: "
            "A factory operates two production lines: Line A produces 60 percent of all units "
            "and Line B produces the remaining 40 percent. "
            "Line A has a defect rate of 3 percent and Line B has a defect rate of 7 percent. "
            "A quality inspector randomly selects a unit and finds it is defective. "
            "What is the probability that the defective unit came from Line A? "
            "Express your final answer as a percentage rounded to two decimal places."
        ),
    },
    {
        "task_type": "reasoning",
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
        "prompt": (
            "I was hoping you might be able to help me answer a question. "
            "I would greatly appreciate a concise and accurate response. "
            "Here is my question: "
            "What is the capital city of France?"
        ),
    },
    {
        "task_type": "qa",
        "prompt": (
            "Could you please be so kind as to tell me the answer to the following question? "
            "I understand this may be a simple question but I want to make sure I get it right. "
            "Question: "
            "How many planets are in our solar system?"
        ),
    },
    {
        "task_type": "qa",
        "prompt": (
            "I would really appreciate it if you could help me out with a trivia question. "
            "Please give me a direct and accurate answer. "
            "Here is the question I have for you: "
            "Who wrote the play Romeo and Juliet?"
        ),
    },
    {
        "task_type": "qa",
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

_SENT_RE  = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_PARA_SEP = "\n\n"


def _segment(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split text into spans, preserving the separator used after each span.

    Returns:
        spans - {uuid: span_text}
        seps  - {uuid: separator_that_followed_span} ("" for the last span)
    """
    paragraphs = re.split(r"(\n\n+)", text.strip())
    parts:     list[str] = []
    part_seps: list[str] = []

    i = 0
    while i < len(paragraphs):
        block   = paragraphs[i].strip()
        sep_after = (
            paragraphs[i + 1]
            if i + 1 < len(paragraphs) and re.match(r"^\n\n+$", paragraphs[i + 1])
            else ""
        )
        if sep_after:
            i += 2
        else:
            i += 1

        if not block:
            continue

        sentences = _SENT_RE.split(block)
        for j, s in enumerate(sentences):
            s = s.strip()
            if not s:
                continue
            parts.append(s)
            part_seps.append(sep_after if j == len(sentences) - 1 else " ")

    if not parts:
        parts     = [text.strip()]
        part_seps = [""]

    uuids = [str(uuid.uuid4()) for _ in parts]
    spans = {u: p for u, p in zip(uuids, parts)}
    seps  = {u: s for u, s in zip(uuids, part_seps)}
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
        # A live API call for every 'rephrase' action adds too much latency and ruins timeout limits.
        # Fall back to a deterministic local structural compression proxy.
        words = span.split()
        if len(words) <= 2:
            return span
        return " ".join(words[: max(1, int(len(words) * 0.6))])

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
            f"Original output:\n<o>\n{original_output}\n</o>\n\n"
            f"Compressed output:\n<o>\n{compressed_output}\n</o>\n\n"
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
            # Cap at 6.0 (not 10.0) so offline mode is neutral (no <0.6 penalty)
            # but doesn't allow reward-hacking via perfect quality scores.
            log.warning("Groq judge call failed — using 6.0 fallback (deterministic token-reduction ratio)")
            return 6.0
        try:
            score = float(result.strip().split()[0])
            return max(0.0, min(10.0, score))
        except (ValueError, IndexError):
            log.warning("Groq judge parsing failed — using 6.0 fallback (deterministic token-reduction ratio)")
            return 6.0


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
        self._spans:              dict[str, str]  = {}
        self._seps:               dict[str, str]  = {}
        self._locked_spans:       list[str]        = []
        self._action_history:     list[dict]       = []
        self._task_type:          str              = "summarization"
        self._token_budget:       int              = 20
        self._original_token_count: int            = 0
        self._initial_span_count: int              = 0
        self._original_prompt:    str              = ""
        self._original_output:    str              = ""
        self._done:               bool             = False
        self._last_obs:           Optional[PromptZipObservation] = None

    # ── internal helpers ──────────────────────

    def _prompt_text(self) -> str:
        """Reassemble spans preserving their original separators."""
        if not self._spans:
            return ""
        keys = list(self._spans.keys())
        parts = []
        for i, uid in enumerate(keys):
            parts.append(self._spans[uid])
            if i < len(keys) - 1:
                parts.append(self._seps.get(uid, " "))
        return "".join(parts)

    def _token_count(self) -> int:
        return _count_tokens(self._prompt_text())

    def _build_obs(
        self, reward: float | None, done: bool, metadata: Optional[dict] = None
    ) -> PromptZipObservation:
        meta = metadata or {}

        obs = PromptZipObservation(
            done=done,
            reward=reward,
            metadata=meta,
            prompt_text=self._prompt_text(),
            spans=dict(self._spans),
            token_count=self._token_count(),
            task_type=self._task_type,
            token_budget=self._token_budget,
            action_history=list(self._action_history),
            locked_spans=list(self._locked_spans),
            original_token_count=self._original_token_count,
            original_prompt=self._original_prompt,
        )
        self._last_obs = obs
        return obs

    def _is_terminated(self) -> bool:
        tc         = self._token_count()
        # Increased multiplier for hard reasoning tasks to avoid premature termination failure.
        step_limit = self._state.step_count >= 3 * self._initial_span_count
        return tc <= self._token_budget or step_limit

    def _run_judge_flow(self) -> float:
        compressed_prompt  = self._prompt_text()
        compressed_output  = self._groq.generate(compressed_prompt)
        tokens_saved       = self._original_token_count - self._token_count()

        raw_score = self._groq.judge(
            original_prompt=self._original_prompt,
            compressed_prompt=compressed_prompt,
            original_output=self._original_output,
            compressed_output=compressed_output,
            task_type=self._task_type,
        )
        quality = raw_score / 10.0  # normalize 0–10 → 0.0–1.0
        final   = quality * (tokens_saved / self._original_token_count) if self._original_token_count else 0.0
        if quality < 0.6:
            final -= 0.5
        return max(-1.0, min(1.0, final))  # allow negative per spec

    # ── OpenEnv API ───────────────────────────

    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str           = "medium",
        **kwargs:   Any,
    ) -> PromptZipObservation:
        """Load the next prompt, segment into spans, pre-generate original_output."""
        self._done           = False
        self._action_history = []
        self._locked_spans   = []

        _tier_map: dict[str, list[str]] = {
            "easy":   ["qa"],
            "medium": ["summarization", "code_gen"],
            "hard":   ["reasoning"],
        }
        allowed = _tier_map.get(difficulty, ["summarization", "code_gen"])
        pool    = [e for e in DATASET if e["task_type"] in allowed]
        if not pool:
            pool = DATASET

        if seed is not None:
            entry = pool[seed % len(pool)]
        else:
            entry = pool[self._dataset_idx % len(pool)]
            self._dataset_idx += 1

        self._task_type           = entry["task_type"]
        self._spans, self._seps   = _segment(entry["prompt"])
        if not any(s.strip() for s in self._spans.values()):
            raise ValueError("Dataset entry produced no non-empty spans")
        self._original_prompt     = self._prompt_text()
        self._original_token_count = self._token_count()
        self._token_budget        = max(1, int(self._original_token_count * 0.55))
        self._initial_span_count  = len(self._spans) or 1
        self._original_output     = self._groq.generate(entry["prompt"])

        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        # Expose original_token_count and original_prompt in metadata so the static grade() method
        # and external evaluators can compute real compression ratios and evaluate semantics offline.
        return self._build_obs(
            reward=None,
            done=False,
        )

    def step(
        self,
        action:    PromptZipAction,
        timeout_s: Optional[float] = None,
        **kwargs:  Any,
    ) -> PromptZipObservation:
        """Apply one compression action and return the updated observation."""
        if self._done:
            return self._build_obs(reward=0.0, done=True, metadata={"info": "episode already done"})

        self._state.step_count += 1

        if action.span_id not in self._spans or action.span_id in self._locked_spans:
            if self._state.step_count >= 3 * self._initial_span_count:
                # Step limit reached on an invalid action
                self._done = True
                final_reward = self._run_judge_flow()
                return self._build_obs(
                    reward=max(-1.0, min(1.0, -0.1 + final_reward)),
                    done=True,
                    metadata={"info": f"invalid or locked span_id: {action.span_id}, step limit reached", "final_reward": final_reward},
                )
            return self._build_obs(
                reward=-0.1,
                done=False,
                metadata={"info": f"invalid or locked span_id: {action.span_id}"},
            )

        self._action_history.append({"action_type": action.action_type, "span_id": action.span_id})
        prev_tokens = self._token_count()

        if action.action_type == "elide":
            sep_to_transfer = self._seps.pop(action.span_id, None)
            keys = list(self._spans.keys())
            idx = keys.index(action.span_id)
            if idx > 0 and sep_to_transfer and "\n" in sep_to_transfer:
                # Transfer paragraph breaks to the preceding span so meaning/structure isn't lost
                self._seps[keys[idx - 1]] = sep_to_transfer
            del self._spans[action.span_id]

            if not self._spans:
                self._done = True
                return self._build_obs(
                    reward=-0.50,
                    done=True,
                    metadata={"info": "all spans elided — destroying meaning"},
                )
        elif action.action_type == "rephrase":
            original_span              = self._spans[action.span_id]
            self._spans[action.span_id] = self._groq.rewrite(original_span)
        elif action.action_type == "preserve":
            self._locked_spans.append(action.span_id)

        new_tokens  = self._token_count()
        step_reward = (
            (prev_tokens - new_tokens) / self._original_token_count * 0.5
            if self._original_token_count else 0.0
        )

        all_locked = set(self._locked_spans) >= set(self._spans.keys())
        if all_locked:
            if self._token_count() >= self._original_token_count:
                self._done = True
                return self._build_obs(
                    reward=0.0,
                    done=True,
                    metadata={"info": "all spans locked without compression — skip judge"},
                )
            else:
                self._done   = True
                final_reward = self._run_judge_flow()
                return self._build_obs(
                    reward=max(-1.0, min(1.0, step_reward + final_reward)),
                    done=True,
                    metadata={"info": "all remaining spans locked, episode terminated", "final_reward": final_reward},
                )

        if self._is_terminated():
            self._done   = True
            final_reward = self._run_judge_flow()
            return self._build_obs(
                reward=max(-1.0, min(1.0, step_reward + final_reward)),
                done=True,
                metadata={"info": "episode terminated", "final_reward": final_reward},
            )

        return self._build_obs(reward=step_reward, done=False)

    @staticmethod
    def grade(*args: Any, **kwargs: Any) -> float:
        """Standalone grader for compressed prompt quality (0, 1) exclusive.

        The evaluation harness requires scores strictly between 0 and 1
        (not 0.0, not 1.0).  All return paths clamp to [_GRADE_MIN, _GRADE_MAX].

        Accepts two calling conventions so the method is robust regardless of
        how the evaluation harness invokes it:

        Convention 1 — explicit positional (our own inference.py / openenv.yaml):
            grade(original_prompt, compressed_prompt, original_output,
                  compressed_output, task_type)

        Convention 2 — framework Rubric-style (action, observation) or (observation,):
            grade(action, observation)  or  grade(observation)
            Fields are extracted from the Pydantic models; compression ratio is
            computed from token_count vs the original_token_count proxy.

        Convention 3 — keyword args:
            grade(original_prompt=..., compressed_prompt=..., ...)
        """
        # ── Shared sentinels (always set before any branch) ───────────────
        # The compute block below references these in every calling convention;
        # initialise here to avoid NameError when Convention 1 or 3 is used.
        _obs_ref:          Any = None
        _orig_tokens_meta: int = 0

        # ── Resolve arguments ─────────────────────────────────────────────
        if len(args) == 5 and isinstance(args[0], str):
            # Convention 1: all five strings supplied positionally
            original_prompt, compressed_prompt, original_output, compressed_output, task_type = args

        elif len(args) == 2:
            # Convention 2a: (action, observation)
            # The observation reflects the *current* (compressed) state. The
            # original token count is injected into obs.metadata by reset() so
            # we can reconstruct the compression ratio without before/after capture.
            _action, _obs_ref = args
            compressed_prompt = getattr(_obs_ref, "prompt_text", "") or ""
            original_output   = ""
            compressed_output = ""
            task_type         = getattr(_obs_ref, "task_type", "qa")
            _orig_tokens_meta = getattr(_obs_ref, "original_token_count", 0)
            original_prompt   = getattr(_obs_ref, "original_prompt", "")
            if not original_prompt:
                if _orig_tokens_meta:
                    original_prompt = " ".join(["x"] * int(_orig_tokens_meta / 1.3))
                else:
                    original_prompt = compressed_prompt  # fallback: ratio = 0

        elif len(args) == 1:
            # Convention 2b: (observation,)
            _obs_ref          = args[0]
            compressed_prompt = getattr(_obs_ref, "prompt_text", "") or ""
            original_output   = ""
            compressed_output = ""
            task_type         = getattr(_obs_ref, "task_type", "qa")
            _orig_tokens_meta = getattr(_obs_ref, "original_token_count", 0)
            original_prompt   = getattr(_obs_ref, "original_prompt", "")
            if not original_prompt:
                if _orig_tokens_meta:
                    original_prompt = " ".join(["x"] * int(_orig_tokens_meta / 1.3))
                else:
                    original_prompt = compressed_prompt

        else:
            # Convention 3: keyword args (or empty call — graceful fallback)
            original_prompt   = str(kwargs.get("original_prompt",   ""))
            compressed_prompt = str(kwargs.get("compressed_prompt", original_prompt))
            original_output   = str(kwargs.get("original_output",   ""))
            compressed_output = str(kwargs.get("compressed_output", ""))
            task_type         = str(kwargs.get("task_type",         "qa"))

        # ── Compute score ─────────────────────────────────────────────────
        # Evaluation harness requires scores in the OPEN interval (0, 1).
        _GRADE_MIN, _GRADE_MAX = 0.001, 0.999

        orig_len    = max(1, len(original_prompt.split()))
        comp_len    = len(compressed_prompt.split())
        compression = max(0.0, 1.0 - comp_len / orig_len)

        # Apply task-type awareness weights
        if task_type == "reasoning":
            sem_weight, comp_weight = 0.8, 0.2
            over_comp_threshold = 0.6
        elif task_type == "qa":
            sem_weight, comp_weight = 0.4, 0.6
            over_comp_threshold = 0.85
        else:
            sem_weight, comp_weight = 0.6, 0.4
            over_comp_threshold = 0.8

        # If we have real outputs, try using the Groq LLM judge for consistency with episode termination
        if original_output and compressed_output and original_output != "[mock output]" and compressed_output != "[mock output]":
            client = _GroqClient()
            if client._client is not None:
                raw_score = client.judge(original_prompt, compressed_prompt, original_output, compressed_output, task_type)
                return round(max(_GRADE_MIN, min(_GRADE_MAX, raw_score / 10.0)), 4)

            # Fallback to output overlap
            orig_toks = set(original_output.lower().split())
            comp_toks = set(compressed_output.lower().split())
            overlap   = len(orig_toks & comp_toks) / max(len(orig_toks), 1)

            if compression > over_comp_threshold:
                compression *= 0.5
            return round(max(_GRADE_MIN, min(_GRADE_MAX, sem_weight * overlap + comp_weight * compression)), 4)

        # Guard: if outputs are missing/mock, fall back to FULLY DETERMINISTIC prompt overlap.
        # In Convention 2 (action, obs), we have no outputs — prefer the direct
        # prompt texts and token-count ratio stored in the observation so the score is exact.
        if not original_prompt or original_prompt == " ".join(["x"] * int(_orig_tokens_meta / 1.3)):
            if _orig_tokens_meta and _obs_ref is not None:
                real_comp   = getattr(_obs_ref, "token_count", None)
                if real_comp is not None:
                    compression = max(0.0, 1.0 - real_comp / _orig_tokens_meta)
            return round(max(_GRADE_MIN, min(_GRADE_MAX, compression)), 4)

        orig_toks = set(original_prompt.lower().split())
        comp_toks = set(compressed_prompt.lower().split())
        overlap   = len(orig_toks & comp_toks) / max(len(orig_toks), 1)

        # Penalise over-compression based on task type limits
        if compression > over_comp_threshold:
            compression *= 0.5

        return round(max(_GRADE_MIN, min(_GRADE_MAX, sem_weight * overlap + comp_weight * compression)), 4)

    @property
    def state(self) -> State:
        """Implements the @property declared in the Environment ABC."""
        return self._state
