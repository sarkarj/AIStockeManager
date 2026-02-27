from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ArtifactMeta(BaseModel):
    ticker: str
    created_at: str
    now_iso: str
    policy_id: str
    policy_version: str
    policy_hash: str
    app_version: str = "0.1.0"
    notes: list[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not ticker:
            raise ValueError("ticker cannot be empty")
        return ticker


class ReplayArtifact(BaseModel):
    meta: ArtifactMeta
    context_pack: dict
    drl_result: dict
    drl_trace: dict
    hub_card: dict | None = None
