---
title: PromptZip RL Environment
emoji: 🗜️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# PromptZip RL Environment

An OpenEnv-compatible reinforcement learning environment where an agent learns to compress LLM prompts via **elide**, **rephrase**, and **preserve** actions on sentence-level spans — preserving output quality while reducing token count.

## Install

```bash
pip install "prompt_zip_env @ git+https://huggingface.co/spaces/<your-username>/prompt_zip_env"
```

## Quick Start

```python
from prompt_zip_env import PromptZipEnv, PromptZipAction

async with PromptZipEnv(base_url="https://<your-username>-prompt-zip-env.hf.space") as client:
    obs = await client.reset()
    print(obs.spans)        # {uuid: span_text, ...}
    print(obs.token_budget) # target token count

    span_id = list(obs.spans.keys())[0]
    obs = await client.step(PromptZipAction(action_type="elide", span_id=span_id))
    print(obs.reward)       # intermediate reward
    print(obs.done)         # False until budget hit or max_steps
```

Sync usage:

```python
with PromptZipEnv(base_url="...").sync() as client:
    obs = client.reset()
    obs = client.step(PromptZipAction(action_type="preserve", span_id=list(obs.spans.keys())[0]))
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your Groq API key:

```
GROQ_API_KEY=your_key_here
```

If `GROQ_API_KEY` is not set, the environment runs with deterministic mock responses (rewrite = first 60% of words, target/judge return fixed fallbacks). The environment never hangs or crashes without an API key.

## MDP

| Component | Detail |
|---|---|
| **Action** | `PromptZipAction(action_type, span_id)` |
| **Observation** | `PromptZipObservation` — spans dict, token_count, done, reward |
| **Intermediate reward** | `(prev_tokens - new_tokens) / original_tokens × 0.5` |
| **Final reward** | `quality_score × (tokens_saved / tokens_original)` |
| **Termination** | `token_count ≤ budget`, `max_steps` hit, all locked, or empty prompt |
