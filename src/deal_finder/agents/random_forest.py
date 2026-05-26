"""Random Forest pricing agent."""

from __future__ import annotations

from pathlib import Path

from deal_finder.agents.base import Agent
from deal_finder.rag.vector_store import EMBEDDING_MODEL


class RandomForestAgent(Agent):
    """Estimate product price from sentence-transformer features."""

    name = "Random Forest Agent"

    def __init__(
        self,
        model_path: str | Path = "models/random_forest_model.joblib",
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.embedding_model = embedding_model
        self._model = None
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    @property
    def model(self):
        if self._model is None:
            import joblib

            self._model = joblib.load(self.model_path)
        return self._model

    def price(self, description: str) -> float:
        self.log("running Random Forest price regression")
        vector = self.encoder.encode([description])
        prediction = float(self.model.predict(vector)[0])
        return max(0.0, prediction)

