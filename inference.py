"""
PromptZip RL — inference.py

Baseline agent that uses an OpenAI-compatible client to play episodes in
the PromptZip environment. The LLM is shown the current observation and
must output one action per step.

Required environment variables:
    API_BASE_URL   — OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — Model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       — API key for the inference endpoint

Optional:
    GROQ_API_KEY   — Groq key for environment's internal Judge (mock if unset)
    DEBUG          — Set to "1" for verbose logging
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from models import PromptZipAction

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
DEBUG        = os.environ.get("DEBUG", "0") == "1"

MAX_STEPS    = 20
TEMPERATURE  = 0.0
MAX_TOKENS   = 256
FALLBACK_ACTION = '{"action_type": "preserve", "span_id": "__fallback__"}'

SYSTEM_PROMPT = """\
You are a prompt compression agent. You receive a bloated LLM prompt split into
sentence-level spans, each identified by a UUID. Your goal is to reduce the token
count below the token_budget while preserving the semantic meaning of the prompt.

At each step you must output exactly ONE JSON action:

  {"action_type": "<elide|rephrase|preserve>", "span_id": "<uuid>"}

Rules:
- elide: delete the span entirely — use for filler, polite preambles, redundant text
- rephrase: keep the span but flag it for rewriting to fewer words — use for verbose but necessary content
- preserve: lock the span unchanged — use for task-critical instructions or factual data
- Never target a span_id that is in locked_spans
- Never target a span_id not in spans

Output ONLY the JSON object. No explanation, no markdown, no extra text.
"""


def make_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "none")


def parse_action(text: str) -> dict:
    """Extract JSON action from model output, with fallback."""
    text = text.strip()
    # Try to find first {...} block
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
        f"token_count: {obs.token_count}  (budget: {obs.token_budget})",
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


def log(msg: str) -> None:
    print(msg, flush=True)


def run_episode(env, client: OpenAI, difficulty: str, episode_num: int) -> float:
    """Run one episode and return total cumulative reward."""
    obs = env.reset(difficulty=difficulty)
    log(f"\n{'='*60}")
    log(f"Episode {episode_num} | difficulty={difficulty} | task={obs.task_type}")
    log(f"Initial tokens: {obs.token_count} → budget: {obs.token_budget}")
    log(f"{'='*60}")

    total_reward = 0.0
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

        if DEBUG:
            log(f"  Step {step}: {action_dict}")

        # Validate span — fallback to first unlocked span if model hallucinated
        span_id = action_dict.get("span_id", "")
        unlocked = [s for s in obs.spans if s not in obs.locked_spans]
        if span_id not in obs.spans or span_id in obs.locked_spans:
            if unlocked:
                action_dict["span_id"] = unlocked[0]
                action_dict["action_type"] = "preserve"
            else:
                obs = env.step(PromptZipAction(action_type="preserve", span_id=list(obs.spans.keys())[0]))
                break

        action = PromptZipAction(
            action_type=action_dict.get("action_type", "preserve"),
            span_id=action_dict["span_id"],
        )
        obs = env.step(action)

        step_reward = obs.reward or 0.0
        total_reward += step_reward
        log(f"  Step {step:2d}: {action.action_type:8s} | tokens={obs.token_count:3d} | reward={step_reward:+.4f}")

        if obs.done:
            final = obs.metadata.get("final_reward", 0.0)
            log(f"  → DONE | final_reward={final:+.4f} | total={total_reward:+.4f}")
            break

    return total_reward


def main() -> None:
    from server.prompt_zip_environment import PromptZipEnvironment

    if not HF_TOKEN:
        log("WARNING: HF_TOKEN not set — inference calls may fail")

    client = make_client()
    env = PromptZipEnvironment()

    difficulties = ["easy", "medium", "hard"]
    episodes_per_difficulty = 2   # 6 total, well within 20-minute runtime

    results: dict[str, list[float]] = {d: [] for d in difficulties}

    episode_num = 1
    for difficulty in difficulties:
        for _ in range(episodes_per_difficulty):
            reward = run_episode(env, client, difficulty, episode_num)
            results[difficulty].append(reward)
            episode_num += 1

    log("\n" + "=" * 60)
    log("BASELINE SCORES")
    log("=" * 60)
    grand_total = 0.0
    for difficulty in difficulties:
        scores = results[difficulty]
        avg = sum(scores) / len(scores) if scores else 0.0
        grand_total += avg
        log(f"  {difficulty:8s}: avg={avg:+.4f}  ({scores})")
    log(f"  {'TOTAL':8s}: {grand_total:+.4f}")
    log("=" * 60)


if __name__ == "__main__":
    main()
