from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


class PromptZipAction(Action):
    """Single compression action targeting one span by its UUID."""

    action_type: Literal["rephrase", "elide", "preserve"]
    span_id: str  # UUID key into PromptZipObservation.spans


class PromptZipObservation(Observation):
    """Full episode state returned by reset() and step()."""

    # Explicitly redefined from base for safety across framework versions.
    # reward is float | None (stricter than base class bool|int|float|None —
    # we only ever emit floats or None so this is safe to narrow).
    done:     bool              = False
    reward:   float | None      = None
    metadata: dict[str, Any]    = Field(default_factory=dict)

    prompt_text:    str          # full reassembled prompt text
    spans:          dict[str, str]  # {uuid: span_text} — stable across elides
    token_count:    int          # approx: len(words) * 1.3
    task_type:      str          # summarization | code_gen | reasoning | qa
    token_budget:   int
    action_history: list[dict]   # [{"action_type": ..., "span_id": ...}, ...]
    locked_spans:   list[str]    # UUIDs of preserved spans


class PromptZipReward(BaseModel):
    """Typed reward breakdown returned at each step and episode termination."""

    step_reward:        float           # Intermediate token-reduction reward for this action
    final_reward:       Optional[float] = None   # Terminal judge reward (None until episode ends)
    total_reward:       float           = 0.0    # step_reward + final_reward (if terminal)
    termination_reason: Optional[str]  = None   # Why the episode ended (None if mid-episode)
    quality_score:      Optional[float] = None   # Normalised 0–1 quality from LLM judge
