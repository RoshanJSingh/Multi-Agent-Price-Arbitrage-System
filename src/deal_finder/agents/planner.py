"""Planner agent that coordinates the deal workflow."""

from __future__ import annotations

from deal_finder.agents.base import Agent
from deal_finder.agents.ensemble import EnsembleAgent
from deal_finder.agents.memory import DealMemory
from deal_finder.agents.messaging import MessagingAgent
from deal_finder.agents.scanner import ScannerAgent
from deal_finder.config import get_settings
from deal_finder.schemas import Deal, Opportunity


class PlannerAgent(Agent):
    """Coordinate scanning, pricing, ranking, and persistence."""

    name = "Planner Agent"

    def __init__(
        self,
        scanner: ScannerAgent,
        ensemble: EnsembleAgent,
        messenger: MessagingAgent | None = None,
        memory: DealMemory | None = None,
        deal_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.scanner = scanner
        self.ensemble = ensemble
        self.messenger = messenger or MessagingAgent()
        self.memory = memory or DealMemory()
        self.deal_threshold = get_settings().deal_threshold if deal_threshold is None else deal_threshold

    def price_deal(self, deal: Deal) -> Opportunity:
        self.log("pricing candidate deal")
        estimate = self.ensemble.price(deal.product_description)
        return Opportunity(deal=deal, estimate=estimate, discount=estimate - deal.price)

    def plan(self) -> Opportunity | None:
        """Run one scan/price/rank cycle and persist the best opportunity."""

        existing = self.memory.read()
        selection = self.scanner.scan(known_urls={str(item.deal.url) for item in existing})
        if selection is None or not selection.deals:
            self.log("no deals selected by scanner")
            return None

        opportunities = [self.price_deal(deal) for deal in selection.deals[:5]]
        opportunities.sort(key=lambda opportunity: opportunity.discount, reverse=True)
        best = opportunities[0]
        self.log(f"best discount is ${best.discount:.2f}")

        if best.discount <= self.deal_threshold:
            return None

        self.memory.write(existing + [best])
        self.messenger.alert(best)
        return best

