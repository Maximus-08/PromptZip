from typing import Any, Literal
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class PromptZipAction(Action):
    """Single compression action targeting one span by its UUID."""

    action_type: Literal["rephrase", "elide", "preserve"]
    span_id: str  # UUID key into PromptZipObservation.spans


class PromptZipObservation(Observation):
    """Full episode state returned by reset() and step()."""

    # Explicitly redefined from base for safety across framework versions
    done: bool = False
    reward: float | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    prompt_text: str
    spans: dict[str, str]          # {uuid: span_text} — stable across elides
    token_count: int               # approx: len(words) * 1.3
    task_type: str                 # summarization | code_gen | reasoning | qa
    token_budget: int
    action_history: list[dict]     # [{"action_type": ..., "span_id": ...}, ...]
    locked_spans: list[str]        # UUIDs of preserved spans
