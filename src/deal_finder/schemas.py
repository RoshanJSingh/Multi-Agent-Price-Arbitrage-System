"""Shared schemas for products, deals, and pricing outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class Deal(BaseModel):
    """A product deal extracted from an external feed."""

    product_description: str = Field(min_length=10)
    price: float = Field(gt=0)
    url: str | HttpUrl


class DealSelection(BaseModel):
    """Structured JSON output returned by the scanner agent."""

    deals: list[Deal]


class Opportunity(BaseModel):
    """A surfaced arbitrage opportunity."""

    deal: Deal
    estimate: float = Field(ge=0)
    discount: float

    @property
    def is_profitable(self) -> bool:
        return self.discount > 0


@dataclass(slots=True)
class ProductRecord:
    """Normalized product row used for model training and RAG indexing."""

    title: str
    description: str
    category: str
    price: float
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_document(self) -> str:
        return f"{self.title}\n{self.description}".strip()


@dataclass(frozen=True, slots=True)
class PriceEstimate:
    """A single model estimate with a named source."""

    source: str
    price: float
    confidence: float = 1.0

