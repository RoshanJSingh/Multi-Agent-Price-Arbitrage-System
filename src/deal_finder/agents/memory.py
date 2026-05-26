"""Persistent deal memory."""

from __future__ import annotations

import json
from pathlib import Path

from deal_finder.schemas import Opportunity


class DealMemory:
    """Tracks opportunities that have already been surfaced."""

    def __init__(self, path: str | Path = "memory.json") -> None:
        self.path = Path(path)

    def read(self) -> list[Opportunity]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [Opportunity.model_validate(item) for item in payload]

    def write(self, opportunities: list[Opportunity]) -> None:
        payload = [opportunity.model_dump(mode="json") for opportunity in opportunities]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def known_urls(self) -> set[str]:
        return {str(opportunity.deal.url) for opportunity in self.read()}

