"""
PromptZip RL — inference.py

Baseline agent that uses an OpenAI-compatible client to play episodes in
the PromptZip environment. The LLM is shown the current observation and
must output one action per step.

Required environment variables:
  API_BASE_URL  — OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — API key for the inference endpoint

Optional:
  GROQ_API_KEY  — Groq key for environment's internal Judge (mock if unset)
  DEBUG         — Set to "1" for verbose logging
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from models import PromptZipAction, PromptZipObservation
import requests

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# Competition spec mandates OPENAI_API_KEY; HF_TOKEN is the fallback alias
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or ""
DEBUG        = os.environ.get("DEBUG", "0") == "1"
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")
USE_DIRECT   = os.environ.get("USE_DIRECT", "0") == "1"

MAX_STEPS             = 20
TEMPERATURE           = 0.0
MAX_TOKENS            = 256
# Normaliser for episode score: sum(rewards) / MAX_TOTAL_REWARD, clamped to [0, 1].
# Set to 1.2 so a good episode (step rewards ~0.2 + final ~0.8) lands near 0.8.
MAX_TOTAL_REWARD      = 1.2
SUCCESS_SCORE_THRESHOLD = 0.25
FALLBACK_ACTION       = '{"action_type": "preserve", "span_id": "__fallback__"}'

SYSTEM_PROMPT = """\
You are a prompt compression agent. You receive a bloated LLM prompt split into
sentence-level spans, each identified by a UUID. Your goal is to reduce the token
count below the token_budget while preserving the semantic meaning of the prompt.

At each step you must output exactly ONE JSON action:
{"action_type": "<elide|rephrase|preserve>", "span_id": "<uuid>"}

Rules:
- elide:    delete the span entirely — use for filler, polite preambles, redundant text
- rephrase: keep the span but flag it for rewriting to fewer words — use for verbose but necessary content
- preserve: lock the span unchanged — use for task-critical instructions or factual data
- Never target a span_id that is in locked_spans
- Never target a span_id not in spans

Output ONLY the JSON object. No explanation, no markdown, no extra text.
"""


def make_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "none")


def parse_action(text: str) -> dict:
    """Extract JSON action from model output, with fallback."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return json.loads(FALLBACK_ACTION)


def obs_to_user_message(obs) -> str:
    """Format the observation into a clear prompt for the agent."""
    unlocked = {k: v for k, v in obs.spans.items() if k not in obs.locked_spans}
    lines = [
        f"task_type: {obs.task_type}",
        f"token_count: {obs.token_count} (budget: {obs.token_budget})",
        f"tokens_to_remove: {max(0, obs.token_count - obs.token_budget)}",
        "",
        "Available spans (uuid → text):",
    ]
    for uid, text in unlocked.items():
        lines.append(f"  {uid}: {text!r}")
    if obs.locked_spans:
        lines.append(f"\nLocked (do NOT target): {obs.locked_spans}")
    lines.append(f"\nPrevious step reward: {obs.reward if obs.reward is not None else 0.0}")
    return "\n".join(lines)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards}", flush=True)


def log(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)


class HttpEnv:
    """Thin HTTP wrapper so run_episode() works identically in server mode."""

    def reset(self, difficulty: str = "medium", seed: int | None = None, **kwargs) -> PromptZipObservation:
        payload: dict = {"difficulty": difficulty}
        if seed is not None:
            payload["seed"] = seed
        resp = requests.post(f"{ENV_URL}/reset", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Handle both flat observation and wrapped {"observation": {...}, "reward": ...}
        if "observation" in data and isinstance(data["observation"], dict):
            obs = data["observation"]
            obs.setdefault("reward",   data.get("reward"))
            obs.setdefault("done",     data.get("done", False))
            obs.setdefault("metadata", data.get("info", {}))
            data = obs
        return PromptZipObservation.model_validate(data)

    def step(self, action: PromptZipAction, **kwargs) -> PromptZipObservation:
        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": action.model_dump()},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if "observation" in data and isinstance(data["observation"], dict):
            obs = data["observation"]
            obs.setdefault("reward",   data.get("reward"))
            obs.setdefault("done",     data.get("done", False))
            obs.setdefault("metadata", data.get("info", {}))
            data = obs
        return PromptZipObservation.model_validate(data)


def run_episode(env, client: OpenAI, difficulty: str, seed: int = 0) -> tuple[float, bool, int, list[float]]:
    """Run one episode and return (score, success, steps_taken, rewards).

    Emits one [START]/[END] log pair per episode (9 total across all tiers),
    using the actual task_type from the first observation so the logged task
    is always accurate (medium can be summarization OR code_gen).
    """
    try:
        obs = env.reset(difficulty=difficulty, seed=seed)
    except Exception as exc:
        if DEBUG:
            log(f"  [DEBUG] Reset failed: {exc}")
        return 0.0, False, 0, []

    log_start(task=obs.task_type, env="prompt_zip_env", model=MODEL_NAME)

    rewards: list[float] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        user_msg = obs_to_user_message(obs)
        messages.append({"role": "user", "content": user_msg})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            if DEBUG:
                log(f"  [DEBUG] Model call failed: {exc}")
            response_text = FALLBACK_ACTION

        action_dict = parse_action(response_text)
        messages.append({"role": "assistant", "content": response_text})

        # Validate span — fallback to first unlocked span if model hallucinated
        span_id = action_dict.get("span_id", "")
        unlocked = [s for s in obs.spans if s not in obs.locked_spans]
        if span_id not in obs.spans or span_id in obs.locked_spans:
            if unlocked:
                action_dict["span_id"]     = unlocked[0]
                action_dict["action_type"] = "preserve"
            else:
                # Should be unreachable (since episodes terminate when fully locked),
                # but if reached, force exit rather than stepping on a locked span
                # which would cause a non-terminal penalty step.
                obs.done = True
                break

        action = PromptZipAction(
            action_type=action_dict.get("action_type", "preserve"),
            span_id=action_dict["span_id"],
        )
        obs = env.step(action)

        step_reward = obs.reward if obs.reward is not None else 0.0
        rewards.append(step_reward)
        steps_taken = step

        log_step(step=step, action=response_text, reward=step_reward, done=obs.done, error=None)

        if obs.done:
            break

    score   = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0) if MAX_TOTAL_REWARD > 0 else 0.0
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success, steps_taken, rewards


def main() -> None:
    if not API_KEY:
        print("WARNING: API_KEY not set — inference calls may fail", flush=True)

    client = make_client()

    if USE_DIRECT:
        from server.prompt_zip_environment import PromptZipEnvironment
        env = PromptZipEnvironment()
        print("[DEBUG] Running in direct-import mode (USE_DIRECT=1)", flush=True)
    else:
        env = HttpEnv()
        print(f"[DEBUG] Running in HTTP mode (ENV_URL={ENV_URL})", flush=True)

    difficulties            = ["easy", "medium", "hard"]
    episodes_per_difficulty = 3  # seeds 0/1/2 sample different prompts per tier

    results: dict[str, list[float]] = {d: [] for d in difficulties}
    total_steps = 0
    all_rewards = []

    for difficulty in difficulties:
        for seed in range(episodes_per_difficulty):
            score, success, steps, rewards = run_episode(env, client, difficulty, seed=seed)
            results[difficulty].append(score)
            total_steps += steps
            all_rewards.extend(rewards)

    print("\n" + "=" * 60, flush=True)
    print("BASELINE SCORES", flush=True)
    print("=" * 60, flush=True)
    grand_total = 0.0
    for difficulty in difficulties:
        scores = results[difficulty]
        avg    = sum(scores) / len(scores) if scores else 0.0
        grand_total += avg
        print(f"  {difficulty:8s}: avg={avg:+.4f} ({scores})", flush=True)
    
    final_score = grand_total / len(difficulties) if difficulties else 0.0
    print(f"  {'TOTAL':8s}: {grand_total:+.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
