"""Runtime configuration helpers for the deal finder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings sourced from environment variables."""

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    modal_app_name: str = os.getenv("MODAL_APP_NAME", "pricer-service")
    modal_class_name: str = os.getenv("MODAL_CLASS_NAME", "Pricer")
    vectorstore_path: Path = Path(os.getenv("VECTORSTORE_PATH", "products_vectorstore"))
    deal_threshold: float = float(os.getenv("DEAL_THRESHOLD", "50"))
    amazon_dataset: str = os.getenv("AMAZON_DATASET", "McAuley-Lab/Amazon-Reviews-2023")
    amazon_subset: str = os.getenv("AMAZON_SUBSET", "raw_meta_All_Beauty")
    amazon_baseline_records: int = int(os.getenv("AMAZON_BASELINE_RECORDS", "40000"))


def get_settings() -> Settings:
    """Return a fresh settings object so tests can override environment values."""

    return Settings()

