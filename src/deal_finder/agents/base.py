"""Shared agent primitives."""

from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(slots=True)
class AgentEvent:
    """Structured event emitted by agents for dashboards and logs."""

    agent: str
    message: str


class Agent:
    """Base class for agents with consistent logging."""

    name = "Agent"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("deal_finder")

    def log(self, message: str) -> AgentEvent:
        event = AgentEvent(agent=self.name, message=message)
        self.logger.info("[%s] %s", event.agent, event.message)
        return event


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a default console logger for local runs."""

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

