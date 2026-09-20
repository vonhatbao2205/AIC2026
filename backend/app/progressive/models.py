from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..models import ImageModelSelection, SearchScope
from ..types import ChannelHit


@dataclass
class RetrievalTrace:
    """Request-local, untruncated pre-fusion observations. No credentials."""

    channel_hits: dict[str, list[ChannelHit]] = field(default_factory=dict)
    # Kept outside the ranking pool, including RRF and video grouping.
    localization_hits: dict[str, list[ChannelHit]] = field(default_factory=dict)
    channel_status: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    queries: dict[str, list[str]] = field(default_factory=dict)
    query_config: dict[str, Any] = field(default_factory=dict)
    work: dict[str, int] = field(default_factory=dict)
    categories: tuple[str, ...] = ()
    latency: dict[str, Any] = field(default_factory=dict)
    rrf_k: int = 60


class HintInput(BaseModel):
    hint_id: str = Field(min_length=1, max_length=128)
    raw_text: str = Field(min_length=1, max_length=8000)
    input_mode: Literal["delta", "cumulative"] = "delta"
    source: str = Field(default="operator", max_length=128)
    # Offset from the start of the task, in milliseconds. Never wall-clock time.
    revealed_at: int | None = Field(default=None, ge=0)
    enabled: bool = True


class ProgressiveConfig(ImageModelSelection):
    task_id: str = Field(default="manual", max_length=256)
    dataset_revision: str | None = Field(default=None, max_length=256)
    # Operator-supplied registry versions; null/empty means not attested.
    model_revisions: dict[str, str] = Field(default_factory=dict)
    scope: SearchScope = Field(default_factory=SearchScope)
    top_k: int = Field(default=200, ge=1, le=1000)
    max_videos: int = Field(default=50, ge=1, le=250)
    translate: bool = True
    hybrid: bool = False
    method: Literal[
        "phm", "phm_no_rescue", "phm_arithmetic", "cumulative", "latest", "hint_rrf", "dual_view"
    ] = "phm"
    epsilon: float = Field(default=0.10, gt=0, lt=1)
    memory_weight: float = Field(default=0.70, ge=0, le=1)
    memory_limit: int = Field(default=250, ge=40, le=250)
    survivor_limit: int = Field(default=8, ge=0, le=8)
    newcomer_limit: int = Field(default=4, ge=0, le=4)
    pair_budget: int = Field(default=16, ge=0, le=16)
    local_k: int = Field(default=3, ge=1, le=3)

    @model_validator(mode="after")
    def fixed_scope(self):
        if self.scope.mode == "auto":
            raise ValueError("Progressive sessions require all or a fixed manual scope")
        return self


class HintUpdate(BaseModel):
    expected_revision: int = Field(ge=0)
    client_request_id: str = Field(min_length=1, max_length=128)
    hints: list[HintInput] = Field(default_factory=list, max_length=32)
    elapsed_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [h.hint_id for h in self.hints]
        if len(ids) != len(set(ids)):
            raise ValueError("hint_id must be unique within a ledger")
        return self
