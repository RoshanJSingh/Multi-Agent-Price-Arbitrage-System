"""Specialist pricing agent backed by a fine-tuned LoRA model."""

from __future__ import annotations

from deal_finder.agents.base import Agent
from deal_finder.config import get_settings


class SpecialistAgent(Agent):
    """Call the remote fine-tuned LLM deployed through Modal."""

    name = "Specialist Agent"

    def __init__(self, modal_app_name: str | None = None, modal_class_name: str | None = None) -> None:
        super().__init__()
        settings = get_settings()
        self.modal_app_name = modal_app_name or settings.modal_app_name
        self.modal_class_name = modal_class_name or settings.modal_class_name
        self._pricer = None

    @property
    def pricer(self):
        if self._pricer is None:
            self.log("connecting to remote LoRA pricing service")
            import modal

            pricer_cls = modal.Cls.lookup(self.modal_app_name, self.modal_class_name)
            self._pricer = pricer_cls()
        return self._pricer

    def price(self, description: str) -> float:
        self.log("calling fine-tuned specialist model")
        result = self.pricer.price.remote(description)
        return max(0.0, float(result))

