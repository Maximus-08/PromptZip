from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


class PromptZipAction(Action):
    """Single compression action targeting one span by its UUID."""

    action_type: Literal["rephrase", "elide", "preserve"]
    span_id: str  # UUID key into PromptZipObservation.spans


class PromptZipObservation(Observation):
    """Full episode state returned by reset() and step()."""

    prompt_text:    str          # full reassembled prompt text
    spans:          dict[str, str]  # {uuid: span_text} — stable across elides
    token_count:    int          # approx: len(words) * 1.3
    task_type:      str          # summarization | code_gen | reasoning | qa
    token_budget:   int
    action_history: list[dict]   # [{"action_type": ..., "span_id": ...}, ...]
    locked_spans:   list[str]    # UUIDs of preserved spans

    # Included directly so they survive transparently over HTTP serialisation
    original_token_count: int = 0
    original_prompt: str = ""


class PromptZipReward(BaseModel):
    """Typed reward signal returned at episode termination."""

    value: float = Field(..., ge=-1.0, le=1.0, description="Clamped final reward [-1, 1]")
    compression_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Fraction of tokens removed")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="LLM judge quality score [0, 1]")
