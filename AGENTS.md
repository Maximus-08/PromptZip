# PromptZip RL — Agents

## Overview

PromptZip RL has **four components** that interact within the environment:

| Component | Role | Trainable |
| --- | --- | --- |
| **Compression Agent** | RL policy being trained; selects actions on prompt spans | ✅ Yes |
| **Rewrite LLM** | Execution primitive; rephrases spans on demand | ❌ No |
| **Target LLM** | Generates outputs from both original and compressed prompts for comparison | ❌ No |
| **LLM Judge** | Scores quality of compressed output vs. original output | ❌ No |

All three use the Groq API at runtime; a deterministic mock fallback is used if GROQ_API_KEY is unavailable. All run inside the Docker container.

---

## Agent 1: Compression Agent (RL Policy)

### Role

The primary agent being trained. It observes a bloated prompt (pre-segmented into spans) and iteratively selects an action and a target span to reduce token count while preserving meaning.

### Observation Space

The agent receives a `PromptZipObservation` object (subclasses OpenEnv's `Observation`):

```python
class PromptZipObservation(Observation):
    done: bool = False
    reward: float | int | None = None
    metadata: dict = {}
    prompt_text: str           # Current full prompt text
    spans: dict[str, str]      # {uuid: span_text} — stable IDs
    token_count: int           # Current token count (approx)
    task_type: str             # "summarization" | "code_gen" | "reasoning" | "qa"
    token_budget: int          # Target token count to stay under
    action_history: list       # [(action_type, span_id), ...] — previous steps
    locked_spans: list[str]    # UUIDs of preserved spans
```

### Action Space

The agent outputs a single `PromptZipAction` per step (subclasses OpenEnv's `Action`):

```python
class PromptZipAction(Action):
    action_type: Literal["rephrase", "elide", "preserve"]
    span_id: str  # UUID key into spans dict
```

| action_type | Description | Execution mechanism | When the agent learns to use it |
| --- | --- | --- | --- |
| `rephrase` | Rewrite the selected span in fewer tokens | Environment calls Rewrite LLM with the span | Verbose instructions, formal register |
| `elide` | Delete the selected span entirely | Environment removes span from `spans[]` and rebuilds `prompt_text` | Polite preambles, redundant boilerplate |
| `preserve` | Lock the selected span against future edits | Environment adds span index to `locked_spans`; future steps skip it | Factual content, task-critical instructions |

**Why three actions, not four:** The original design included both `rephrase` and `compress` as separate actions. These are functionally identical — both rewrite text to fewer tokens — so they are merged into `rephrase`. The `chunk` action is excluded from Round 1; it requires a fundamentally different episode structure and will be added in v2.

### Behavior

* Takes **multiple steps per episode** — one `PromptZipAction` per `step()` call
* Continues until: `token_count ≤ token_budget`, `quality_score < 6.0` (early termination), `max_steps = 2 × len(spans)` is reached, or short-circuit conditions met (all spans locked or empty prompt)
* `step()` returns `PromptZipObservation` (Observation base provides done, reward, metadata)
* Policy and span selector weights are updated jointly via **GRPO** through TRL/Torchforge

### Reward Signal

**Intermediate reward** (in `StepResult.reward` after each `step()`):

```
step_reward = (prev_token_count - new_token_count) / original_token_count × 0.5
```

Positive when tokens are reduced. Negative when a bad rephrase increases tokens.

**Final reward** (in `StepResult.reward` at episode termination):

```
final_reward = quality_score × (tokens_saved / tokens_original)
```

If `quality_score < 6.0`: additional penalty of `−5.0` applied.

### Learned Strategies (Emergent)

| Task Type | Learned Priority |
| --- | --- |
| Summarization | Elide request framing first; preserve source content spans |
| Code generation | Preserve system prompt spans; rephrase user instruction spans |
| Multi-step reasoning | Preserve chain-of-thought structure; elide padding and meta-instructions |
| Q&A | Elide polite preamble; preserve the question core |

---

## Rewrite LLM (Execution Primitive)

### Role

A small, frozen language model called by the **environment** to execute `rephrase` actions. It is not an RL agent and does not learn.

### Spec

| Property | Value |
| --- | --- |
| **Model** | Llama-3.3-70b-versatile (via Groq) |
| **Runtime** | Groq API (fallback: deterministic mock returns first 60% of words) |
| **Temperature** | 0 (deterministic output) |
| **Prompt template** | `"Rewrite the following to convey the same meaning in fewer tokens. Return only the rewritten text, no explanation.\n\n{span}"` |
| **Trainable** | No — frozen throughout |

Called only for `rephrase` steps. `elide` and `preserve` require no LLM call.

---

## Target LLM (Output Generator)

### Role

A small, frozen language model that generates text outputs from both the **original prompt** and the **compressed prompt**. These outputs are what the LLM Judge compares to produce the quality score. Without this component, there is nothing to evaluate.

### When It Runs

* **At `reset()`**: Generates `original_output` from the bloated prompt and **caches it** for the episode. This avoids re-generating the baseline on every step.
* **At episode termination**: Generates `compressed_output` from the final compressed prompt.

### Spec

| Property | Value |
| --- | --- |
| **Model** | Llama-3.1-8b-instant (via Groq) |
| **Runtime** | Groq API (fallback: returns "[mock output]") |
| **Temperature** | 0 (deterministic — same prompt always produces same output) |
| **Trainable** | No — frozen throughout |

> **Why a small model?** The Target LLM is called twice per episode (once at reset, once at termination). A lighter model keeps environment latency low. The judge evaluates semantic equivalence, not output quality, so the baseline doesn't need to be "good" — only consistent.

---

## Agent 2: LLM Judge (Quality Evaluator)

### Role

A frozen LLM that runs **once per episode, at termination**. It receives the original and compressed prompt outputs from the Target LLM and returns a quality score.

### Input

```python
{
  "original_prompt": str,       # The uncompressed prompt
  "compressed_prompt": str,     # The agent's final compressed version
  "original_output": str,       # Target LLM output from original prompt
  "compressed_output": str,     # Target LLM output from compressed prompt
  "task_type": str              # Context for evaluation rubric
}
```

### Output

```python
quality_score: float  # 0.0 – 10.0
```

### Scoring Rubric

The judge evaluates along four dimensions, each scored 0–2.5:

| Dimension | What it checks |
| --- | --- |
| **Semantic preservation** | Does the compressed output cover the same key points? |
| **Factual accuracy** | Are all facts from the baseline present and correct? |
| **Completeness** | Is anything meaningful omitted? |
| **Coherence** | Is the compressed output well-structured and unambiguous? |

`quality_score = sum of four dimension scores` (0–10).

### Design Decisions

**Frozen judge:** The judge does not learn or update during training, preventing reward hacking.

**Temperature=0, single call:** The judge runs once at `temperature=0`. Running 3 calls at `temperature=0` would produce 3 identical scores (deterministic model) with no noise-reduction benefit and 3× latency cost. If non-determinism is needed in future, raise temperature and use a mean of 3 samples.

**Runs once per episode, at termination:** Calling the judge at every step would be cost-prohibitive. Intermediate shaping comes from the token-delta step reward; the judge provides the terminal quality signal only.

---

## Agent Interaction Loop

```
sequenceDiagram
    participant D as Dataset
    participant E as Environment
    participant TGT as Target LLM (frozen)
    participant RW as Rewrite LLM (frozen)
    participant C as Compression Agent
    participant J as LLM Judge (frozen)

    D->>E: reset() — load bloated prompt, segment into spans
    E->>TGT: original prompt → generate original_output (cached)
    E->>C: PromptZipObservation {prompt, spans[], tokens, budget}
    loop Until done=True
        C->>E: step(PromptZipAction(action_type, span_id))
        alt action_type == rephrase
            E->>RW: span text
            RW->>E: rewritten span (temp=0)
        else action_type == elide
            E->>E: remove span from spans dict
        else action_type == preserve
            E->>E: add span_id to locked_spans
        end
        E->>C: Updated PromptZipObservation (with step_reward, done)
    end
    E->>TGT: compressed prompt → generate compressed_output
    TGT->>E: compressed_output
    E->>J: {original_prompt, compressed_prompt, original_output, compressed_output, task_type}
    J->>E: quality_score (0–10)
    E->>C: StepResult(observation, final_reward=[quality × (saved/total) ± penalty], done=True)
    Note over C: Policy + span selector updated via GRPO
```

---

## Reward Signal Flow

```
                    Compression Agent
                          │
                  step(PromptZipAction) × N
                          │
                          ▼
              ┌─────────────────────────────┐
              │   Environment               │
              │   executes action on span   │
              │   StepResult.reward =       │
              │     token_delta × 0.5       │
              └────────────┬────────────────┘
                           │ (at termination)
                           ▼
              ┌─────────────────────────────┐
              │   Target LLM (frozen)       │
              │   → compressed_output       │
              └────────────┬────────────────┘
                           │
              ┌─────────────────────────────┐
              │   LLM Judge (frozen)        │
              │   temp=0, single call       │
              │   → quality_score (0–10)    │
              └────────────┬────────────────┘
                           │
       final_reward = quality × (saved / total) [± penalty]
                           │
                           ▼
              ┌─────────────────────────────┐
              │   GRPO / TRL                │
              │   policy + span selector    │
              │   weight update             │
              └─────────────────────────────┘
```

---

## Key Constraints

| Property | Compression Agent | Rewrite LLM | Target LLM | LLM Judge |
| --- | --- | --- | --- | --- |
| **Trainable** | ✅ Yes — GRPO each episode | ❌ No — frozen | ❌ No — frozen | ❌ No — frozen |
| **Runs per episode** | Multiple `step()` calls | Once per `rephrase` step | Twice (reset + termination) | Once at episode end |
| **Temperature** | N/A | 0 (deterministic) | 0 (deterministic) | 0 (deterministic) |
| **GPU required** | Yes (for GRPO training) | No | No | No |
| **Deployed as** | Trained separately; weights loaded into container | Groq API / Mock | Groq API / Mock | Groq API / Mock |
