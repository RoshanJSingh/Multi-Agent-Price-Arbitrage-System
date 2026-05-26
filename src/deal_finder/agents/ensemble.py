"""Hybrid AI/ML ensemble pricing agent."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from deal_finder.agents.base import Agent
from deal_finder.agents.frontier import FrontierAgent
from deal_finder.agents.random_forest import RandomForestAgent
from deal_finder.agents.specialist import SpecialistAgent
from deal_finder.schemas import PriceEstimate


class EnsembleAgent(Agent):
    """Blend specialist, frontier, and Random Forest price estimates."""

    name = "Ensemble Agent"

    def __init__(
        self,
        specialist: SpecialistAgent,
        frontier: FrontierAgent,
        random_forest: RandomForestAgent,
        blender_path: str | Path = "models/ensemble_model.joblib",
    ) -> None:
        super().__init__()
        self.specialist = specialist
        self.frontier = frontier
        self.random_forest = random_forest
        self.blender_path = Path(blender_path)
        self._blender = None

    @property
    def blender(self):
        if self._blender is None and self.blender_path.exists():
            import joblib

            self._blender = joblib.load(self.blender_path)
        return self._blender

    def collect_estimates(self, description: str) -> list[PriceEstimate]:
        self.log("collecting specialist, frontier, and Random Forest estimates")
        return [
            PriceEstimate("specialist", self.specialist.price(description)),
            PriceEstimate("frontier", self.frontier.price(description)),
            PriceEstimate("random_forest", self.random_forest.price(description)),
        ]

    def price(self, description: str) -> float:
        estimates = self.collect_estimates(description)
        values = [estimate.price for estimate in estimates]
        if self.blender is None:
            result = mean(values)
        else:
            import pandas as pd

            features = pd.DataFrame(
                {
                    "Specialist": [values[0]],
                    "Frontier": [values[1]],
                    "RandomForest": [values[2]],
                    "Min": [min(values)],
                    "Max": [max(values)],
                }
            )
            result = float(self.blender.predict(features)[0])
        self.log(f"ensemble estimate is ${result:.2f}")
        return max(0.0, result)

