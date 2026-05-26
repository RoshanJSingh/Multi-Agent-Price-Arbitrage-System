"""Frontier LLM pricing agent with RAG context."""

from __future__ import annotations

import re

from deal_finder.agents.base import Agent
from deal_finder.rag.vector_store import ProductVectorStore, SimilarProduct


class FrontierAgent(Agent):
    """Estimate prices with GPT-4-class models and retrieved product context."""

    name = "Frontier Agent"

    def __init__(self, vector_store: ProductVectorStore, model: str = "gpt-4o-mini") -> None:
        super().__init__()
        self.vector_store = vector_store
        self.model = model

    def retrieve_context(self, description: str) -> list[SimilarProduct]:
        self.log("retrieving similar products from ChromaDB")
        return self.vector_store.query(description, n_results=5)

    def build_context(self, similars: list[SimilarProduct]) -> str:
        lines = ["Comparable products from the Amazon baseline:"]
        for product in similars:
            lines.append(f"- {product.document}\n  Ground-truth price: ${product.price:.2f}")
        return "\n".join(lines)

    def price(self, description: str) -> float:
        similars = self.retrieve_context(description)
        context = self.build_context(similars)
        self.log(f"calling {self.model} with retrieved pricing context")

        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Estimate product prices. Reply only with a USD number."},
                {"role": "user", "content": f"{context}\n\nProduct to price:\n{description}"},
            ],
            seed=42,
            max_tokens=10,
        )
        return extract_price(response.choices[0].message.content or "")


def extract_price(text: str) -> float:
    """Extract a numeric price from model output."""

    normalized = text.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", normalized)
    return float(match.group()) if match else 0.0

