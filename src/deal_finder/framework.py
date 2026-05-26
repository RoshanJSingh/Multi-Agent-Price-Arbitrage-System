"""Factory for the complete seven-agent deal-finding workflow."""

from __future__ import annotations

from deal_finder.agents.ensemble import EnsembleAgent
from deal_finder.agents.frontier import FrontierAgent
from deal_finder.agents.memory import DealMemory
from deal_finder.agents.messaging import MessagingAgent
from deal_finder.agents.planner import PlannerAgent
from deal_finder.agents.random_forest import RandomForestAgent
from deal_finder.agents.scanner import ScannerAgent
from deal_finder.agents.specialist import SpecialistAgent
from deal_finder.config import get_settings
from deal_finder.rag.vector_store import ProductVectorStore


def build_planner(memory_path: str = "memory.json") -> PlannerAgent:
    """Build the scanner, frontier, specialist, RF, ensemble, planner, and messenger agents."""

    settings = get_settings()
    vector_store = ProductVectorStore(settings.vectorstore_path)
    frontier = FrontierAgent(vector_store=vector_store)
    specialist = SpecialistAgent()
    random_forest = RandomForestAgent()
    ensemble = EnsembleAgent(
        specialist=specialist,
        frontier=frontier,
        random_forest=random_forest,
    )
    return PlannerAgent(
        scanner=ScannerAgent(),
        ensemble=ensemble,
        messenger=MessagingAgent(),
        memory=DealMemory(memory_path),
        deal_threshold=settings.deal_threshold,
    )

