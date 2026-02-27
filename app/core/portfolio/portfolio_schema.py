from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Holding(BaseModel):
    ticker: str
    avg_cost: float = Field(ge=0.0)
    quantity: float = Field(ge=0.0)
    tags: list[str] | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not ticker:
            raise ValueError("ticker cannot be empty")
        return ticker


class Portfolio(BaseModel):
    holdings: list[Holding] = Field(default_factory=list)
