"""RSS deal feed ingestion."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup


DEFAULT_FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]


@dataclass(frozen=True, slots=True)
class ScrapedDeal:
    """Deal candidate retrieved from an RSS feed."""

    title: str
    summary: str
    url: str
    details: str = ""
    features: str = ""

    def describe(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Summary: {self.summary}\n"
            f"Details: {self.details.strip()}\n"
            f"Features: {self.features.strip()}\n"
            f"URL: {self.url}"
        )


def extract_text(html_snippet: str) -> str:
    """Extract visible text from RSS HTML snippets."""

    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet = soup.find("div", class_="snippet summary")
    text = snippet.get_text(" ", strip=True) if snippet else soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def fetch_deal_page(url: str, timeout: int = 20) -> tuple[str, str]:
    """Fetch detailed content and split features when the page provides them."""

    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    content_section = soup.find("div", class_="content-section")
    content = content_section.get_text(" ", strip=True) if content_section else ""
    content = content.replace(" more", " ").strip()
    if "Features" in content:
        details, features = content.split("Features", maxsplit=1)
    else:
        details, features = content, ""
    return details.strip(), features.strip()


def fetch_rss_deals(
    feeds: Iterable[str] = DEFAULT_FEEDS,
    max_entries_per_feed: int = 10,
    page_delay_seconds: float = 0.5,
) -> list[ScrapedDeal]:
    """Fetch deal candidates from RSS feeds and their detail pages."""

    import feedparser

    deals: list[ScrapedDeal] = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:max_entries_per_feed]:
            url = entry["links"][0]["href"]
            details, features = fetch_deal_page(url)
            deals.append(
                ScrapedDeal(
                    title=entry.get("title", ""),
                    summary=extract_text(entry.get("summary", "")),
                    url=url,
                    details=details,
                    features=features,
                )
            )
            time.sleep(page_delay_seconds)
    return deals

