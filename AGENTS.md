# PromptZip RL — Agents

## Overview

PromptZip has **two agents** that operate within each episode: the **Compression Agent**
(the policy being trained or evaluated) and the **LLM Judge** (the frozen quality
evaluator that produces the final reward signal). They operate in a closed loop — one
compresses span-by-span, the other scores the result at episode end.

---

## Agent 1: Compression Agent (RL Policy)

### Role

The primary agent. It observes a bloated LLM prompt segmented into UUID-keyed sentence
spans, and iteratively applies compression actions to reduce token count while
preserving meaning. At inference time, this is the LLM driven by `inference.py` via
the OpenAI-compatible client.

### Observation Space

```python
PromptZipObservation(
    prompt_text: str,           # Current full prompt text (spans joined)
    spans: dict[str, str],      # {uuid: span_text} — stable UUIDs across steps
    token_count: int,           # Approximate current token count
    task_type: str,             # "summarization" | "code_gen" | "reasoning" | "qa"
    token_budget: int,          # Target: episode ends when token_count ≤ this
    action_history: list[dict], # [{"action_type": ..., "span_id": ...}, ...]
    locked_spans: list[str],    # UUIDs that cannot be targeted again
    done: bool,                 # True when episode has ended
    reward: float | None,       # Reward from last step (None at reset)
    metadata: dict,             # Info dict — termination reason, final_reward, etc.
)
```

### Action Space

```python
PromptZipAction(
    action_type: Literal["rephrase", "elide", "preserve"],
    span_id: str,   # UUID of the target span (must be in obs.spans and not locked)
)
```

| Action | Effect on Prompt | Step Reward | Use When |
|---|---|---|---|
| `elide` | Span deleted permanently | `(prev_tokens - new_tokens) / original × 0.5` | Polite preambles, filler |
| `rephrase` | Span rewritten in fewer words by Groq | `(prev_tokens - new_tokens) / original × 0.5` | Verbose but necessary content |
| `preserve` | Span added to `locked_spans`, no text change | `0.0` | Task-critical instructions, factual data |

**Invalid actions** (targeting a non-existent or already-locked span) return
`reward = -0.1` and leave the prompt unchanged.

### Decision Loop

At each step the agent must:

1. Inspect `obs.spans` to see available span texts and their UUIDs
2. Inspect `obs.locked_spans` to filter out already-committed spans
3. Inspect `obs.token_count` vs `obs.token_budget` to gauge urgency
4. Choose one `(action_type, span_id)` pair

The episode ends when `done=True` in the returned observation. The agent should
**always check `obs.done` before calling `step()` again**.

### Difficulty Tiers

The environment serves three task tiers. `reset(difficulty=...)` targets a specific
tier for reproducible grader runs:

| Difficulty | Task Type | Token Budget | Challenge |
|---|---|---|---|
| `"easy"` | QA | 10–13 | Short prompt, obvious filler — single elide often wins |
| `"medium"` | Summarization, Code Gen | 12–22 | Longer prompts, some spans are load-bearing |
| `"hard"` | Reasoning | 14–20 | Multi-step structure — wrong compression breaks the answer chain |

### Learned Strategies (Emergent from Reward Signal)

A trained agent develops task-conditioned heuristics:

| Task Type | What to Elide First | What to Preserve |
|---|---|---|
| QA | "I was hoping you might help...", "I would really appreciate..." | The actual question |
| Summarization | "Please be thorough", "Do not skip minor details" | Source text content |
| Code Gen | "If it is not too much trouble...", "Feel free to add..." | Function spec, data types |
| Reasoning | "Could you be so kind as to...", padding phrases | All numbers, logical structure |

---

## Agent 2: LLM Judge (Quality Evaluator)

### Role

A frozen Groq-hosted LLM that acts as the environment's **grader**. It is called once
per episode at termination (when `token_count ≤ budget` or step limit reached). It
compares the output produced by the compressed prompt against the output produced by
the original prompt and returns a quality score.

### Model

`llama-3.3-70b-versatile` via Groq API (10-second timeout).
Fallback if Groq unavailable: returns `0.0` (safe no-reward fallback).

### Input

```python
judge_input = {
    "original_prompt":    str,  # The uncompressed prompt from reset()
    "compressed_prompt":  str,  # The agent's final prompt text
    "original_output":    str,  # Output generated from original prompt at reset()
    "compressed_output":  str,  # Output generated from compressed prompt now
    "task_type":          str,  # Used to apply the appropriate rubric
}
```

### Output

```
raw_score: float  # 0.0 – 10.0  (normalized to 0.0–1.0 before reward computation)
```

### Scoring Rubric

The judge prompt instructs the model to score on four equally-weighted dimensions:

| Dimension | Max | What It Checks |
|---|---|---|
| Semantic preservation | 2.5 | Do both outputs cover the same key points? |
| Factual accuracy | 2.5 | Are all facts from the original preserved? |
| Completeness | 2.5 | Is anything meaningful omitted? |
| Coherence | 2.5 | Is the compressed output well-structured? |

### Reward Computation

```python
quality    = raw_score / 10.0                              # → 0.0–1.0
tokens_saved = original_token_count - current_token_count
final      = quality * (tokens_saved / original_token_count)
if quality < 0.6:
    final -= 0.5                                           # penalty for quality collapse
final      = max(-1.0, min(1.0, final))                    # hard clamp
```

### Design Decisions

The judge is **frozen** — it does not learn and its weights are never updated. This
prevents the compression agent from reward-hacking by co-adapting with the judge.

The judge is only called when the episode terminates via budget or step limit. Short-
circuit terminations (all spans locked without compression, empty prompt) skip the
judge entirely and return fixed rewards (`0.0` and `−0.50` respectively), saving API
calls on trivially bad episodes.

---

## Agent Interaction Loop

```
Dataset
  │
  │  reset(difficulty="medium")
  ▼
Environment
  │  PromptZipObservation
  │  {spans, token_count, token_budget, task_type, ...}
  ▼
Compression Agent  ─────────────────────────────────────────────┐
  │                                                              │
  │  step(PromptZipAction(action_type, span_id))                 │
  ▼                                                              │
Environment                                                      │
  │  applies action to spans                                     │
  │  computes step_reward                                        │
  │  checks termination                                          │
  │                                                              │
  ├──[not done]──► Updated PromptZipObservation ────────────────►┘
  │
  └──[done: budget or step limit]
       │
       ├─► Target LLM generates compressed_output
       │
       ├─► LLM Judge scores (compressed vs original)
       │     quality_score → 0.0–1.0
       │
       └─► final_reward = quality × (tokens_saved / total)
             (clamped, penalised if quality < 0.6)
```

---

## What the Compression Agent Receives at Each Step

```
After reset():
  obs.done          = False
  obs.reward        = None          ← no reward yet
  obs.token_count   = 42            ← current token estimate
  obs.token_budget  = 20            ← must get below this
  obs.spans         = {             ← UUID → text mapping
    "a1b2...": "I was hoping you might be able to help me answer a question.",
    "c3d4...": "I would greatly appreciate a concise and accurate response.",
    "e5f6...": "Here is my question:",
    "g7h8...": "What is the capital city of France?"
  }
  obs.locked_spans  = []
  obs.action_history = []

After step(elide, "a1b2..."):
  obs.done          = False
  obs.reward        = +0.143        ← step_reward for tokens removed
  obs.token_count   = 30
  obs.spans         = {             ← span a1b2 gone
    "c3d4...": "I would greatly appreciate a concise and accurate response.",
    "e5f6...": "Here is my question:",
    "g7h8...": "What is the capital city of France?"
  }

After step(elide, "c3d4..."):
  obs.done          = True          ← token_count=18 ≤ budget=20
  obs.reward        = +0.640        ← step_reward + final_reward
  obs.metadata      = {"info": "episode terminated", "final_reward": 0.497}
```

---

## Agent Constraints Summary

| Property | Compression Agent | LLM Judge |
|---|---|---|
| Trainable | ✅ Weights updated by RL loop | ❌ Frozen throughout |
| Runs per episode | Multiple `step()` calls | Once at terminal step |
| External API | `API_BASE_URL` (inference.py) | Groq (`GROQ_API_KEY`) |
| Offline fallback | Deterministic mock obs | Returns `0.0` quality |
| GPU required | ❌ No | ❌ No |

---

## Running the Baseline Agent

```bash
export API_BASE_URL="https://api.openai.com/v1"   # or any OpenAI-compatible endpoint
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export GROQ_API_KEY="your-groq-key"               # optional — mock fallback if unset

python inference.py
```

The script runs all 16 episodes (easy → medium → hard) sequentially, logs each step's
action and reward, and prints a final score summary. Runtime is under 5 minutes on
2 vCPU / 8 GB with the Groq backend active.
