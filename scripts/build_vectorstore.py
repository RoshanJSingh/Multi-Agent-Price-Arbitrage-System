"""Build the local Chroma vector store from the Amazon baseline sample."""

from __future__ import annotations

from deal_finder.config import get_settings
from deal_finder.data.amazon_loader import load_baseline_sample
from deal_finder.rag.vector_store import ProductVectorStore


def main() -> None:
    settings = get_settings()
    records = load_baseline_sample(settings.amazon_baseline_records)
    store = ProductVectorStore(settings.vectorstore_path)
    inserted = store.add_products(records)
    print(f"Inserted {inserted} products into {settings.vectorstore_path}")


if __name__ == "__main__":
    main()

