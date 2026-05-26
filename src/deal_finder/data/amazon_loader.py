"""Amazon 2023 product metadata ingestion utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

from deal_finder.config import get_settings
from deal_finder.schemas import ProductRecord


PRICE_PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")


def clean_text(value: Any) -> str:
    """Normalize text-like values from Hugging Face product metadata."""

    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item)
    text = re.sub(r"\s+", " ", str(value))
    return text.strip()


def normalize_price(value: Any) -> float | None:
    """Convert common Amazon price formats into a float."""

    if value in (None, "", "None"):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    match = PRICE_PATTERN.search(str(value).replace(",", ""))
    if not match:
        return None
    price = float(match.group())
    return price if price > 0 else None


def normalize_record(raw: dict[str, Any], category: str = "unknown") -> ProductRecord | None:
    """Convert a raw Amazon metadata row into the training/RAG schema."""

    title = clean_text(raw.get("title"))
    description = clean_text(raw.get("description"))
    features = clean_text(raw.get("features"))
    details = clean_text(raw.get("details"))
    price = normalize_price(raw.get("price") or raw.get("average_rating_price"))

    if not title or price is None:
        return None

    body = "\n".join(part for part in [description, features, details] if part)
    if len(body) < 80:
        return None

    return ProductRecord(
        title=title,
        description=body,
        category=clean_text(raw.get("main_category")) or category,
        price=price,
        source_id=clean_text(raw.get("parent_asin") or raw.get("asin")) or None,
        metadata={
            "rating": raw.get("average_rating"),
            "rating_count": raw.get("rating_number"),
            "store": raw.get("store"),
        },
    )


def iter_normalized_records(rows: Iterable[dict[str, Any]], category: str) -> Iterator[ProductRecord]:
    """Yield normalized rows while dropping incomplete or unsupported products."""

    for row in rows:
        record = normalize_record(row, category=category)
        if record is not None:
            yield record


def stream_amazon_records(limit: int | None = None) -> Iterator[ProductRecord]:
    """Stream normalized Amazon 2023 metadata rows from Hugging Face."""

    settings = get_settings()
    from datasets import load_dataset

    dataset = load_dataset(
        settings.amazon_dataset,
        settings.amazon_subset,
        split="full",
        streaming=True,
    )
    yielded = 0
    for record in iter_normalized_records(dataset, category=settings.amazon_subset):
        yield record
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def load_baseline_sample(limit: int | None = None) -> list[ProductRecord]:
    """Load the default baseline sample used by evaluation notebooks."""

    settings = get_settings()
    target = settings.amazon_baseline_records if limit is None else limit
    return list(stream_amazon_records(limit=target))

