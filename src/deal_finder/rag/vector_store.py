"""ChromaDB-backed product vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deal_finder.schemas import ProductRecord


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384


@dataclass(frozen=True, slots=True)
class SimilarProduct:
    """Retrieved product context for pricing prompts."""

    document: str
    price: float
    category: str
    distance: float | None = None


class ProductVectorStore:
    """Persist and query product embeddings in ChromaDB."""

    def __init__(
        self,
        path: str | Path,
        collection_name: str = "products",
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode text into 384-D sentence-transformer vectors."""

        vectors = self.encoder.encode(texts)
        return vectors.astype(float).tolist()

    def add_products(self, records: list[ProductRecord], batch_size: int = 256) -> int:
        """Add normalized products to the vector store."""

        inserted = 0
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            documents = [record.as_document() for record in batch]
            ids = [record.source_id or f"product-{start + idx}" for idx, record in enumerate(batch)]
            metadatas = [
                {
                    "price": record.price,
                    "category": record.category,
                    **record.metadata,
                }
                for record in batch
            ]
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=self.embed(documents),
                metadatas=metadatas,
            )
            inserted += len(batch)
        return inserted

    def query(self, description: str, n_results: int = 5) -> list[SimilarProduct]:
        """Return similar products and their ground-truth prices."""

        result = self.collection.query(
            query_embeddings=self.embed([description]),
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            SimilarProduct(
                document=document,
                price=float(metadata.get("price", 0.0)),
                category=str(metadata.get("category", "unknown")),
                distance=distances[idx] if idx < len(distances) else None,
            )
            for idx, (document, metadata) in enumerate(zip(documents, metadatas))
        ]

