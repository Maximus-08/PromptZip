---
title: PromptZip RL Environment
emoji: 🗜️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# PromptZip RL Environment

PromptZip is a real-world OpenEnv-compatible reinforcement learning environment where an agent learns to perform **prompt compression**. As Large Language Models (LLMs) scale, inference costs heavily correlate with input token length. PromptZip simulates the genuine engineering task of reducing prompt token count to save money while retaining output quality and task semantic meaning.

The agent observes a bloated prompt broken down into sentence-level spans and iteratively decides whether to delete filler text, request a compressed rewrite of verbose language, or lock essential instructions.

## Task Definitions and Difficulty Tiers

The environment provides 3 distinct difficulty tiers, scaling up in nuance:

- **Easy (QA)**: Budgets of 20-24 tokens. Simple questions padded with obvious filler like "I was wondering if you could...". A single elision usually hits the target budget.
- **Medium (Summarization, Code Generation)**: Budgets of 26-34 tokens. Prompts include context to summarize or function requirements. Some filler exists, but preserving structural requirements is essential.
- **Hard (Reasoning)**: Budgets of 28-33 tokens. Multi-step reasoning questions (e.g. arithmetic or logic puzzles). Deleting the wrong semantic span will catastrophically break the Target LLM’s reasoning chain. Challenges frontier models to discern what information is load-bearing.

## Action & Observation Space

### Action Space (`PromptZipAction`)
At each step, the agent targets one span via its UUID and applies one of three actions:

| Action | Description | Reward Effect |
|--------|-------------|---------------|
| `elide` | Permanently deletes the span. | `(prev - new) / original_tokens × 0.5` |
| `rephrase` | Rewrites the span in fewer words. | `(prev - new) / original_tokens × 0.5` |
| `preserve` | Locks the span unchanged and protects it from future actions. | `0.0` |

### Observation Space (`PromptZipObservation`)
| Field | Type | Description |
|-------|------|-------------|
| `prompt_text` | `str` | Full current prompt text. |
| `spans` | `dict[str, str]` | stable UUIDs mapping to text spans. |
| `token_count` | `int` | Approximate current token count. |
| `task_type` | `str` | The task category (qa, summarization, etc.). |
| `token_budget` | `int` | Target token threshold to end the episode. |
| `action_history` | `list[dict]` | Previous actions taken. |
| `locked_spans` | `list[str]` | UUIDs that cannot be targeted again. |
| `done`, `reward`, `metadata` | ... | Standard RL interaction elements. |

## Reward Function

The reward function provides dense intermediate signals per step plus a terminal judge score:
- **Intermediate**: `(prev_tokens - new_tokens) / original_tokens × 0.5` for successful compression steps.
- **Penalties**: `-0.1` for invalid actions, `-0.5` for eliding all spans (empty prompt).
- **Final Reward**: `quality_score × (tokens_saved / tokens_original)`. Collapsed quality (<0.6) yields a `-0.5` penalty. Scores are strictly clamped between `[-1.0, 1.0]`.

## Setup and Docker

To run the environment in Docker locally:
```bash
docker build -t prompt_zip_env .
docker run -p 8000:8000 prompt_zip_env
```
The application will serve from `http://localhost:8000`.

## Inference Baseline

The baseline client script uses the standard `openai` library compatible package. Set your environment variables (use `.env.example` as a template):
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"
```

To run baseline execution:
```bash
python inference.py
```

### Reproducible Baseline Scores

Running `gpt-4o-mini` on the environment yields the following scores (aggregated across seeds):
- **easy**: ~ +0.5520 
- **medium**: ~ +0.4851
- **hard**: ~ +0.3210
*(Exact numbers vary per sample but consistently drop according to difficulty.)*
