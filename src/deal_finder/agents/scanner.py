"""Scanner agent for selecting high-quality RSS deal candidates."""

from __future__ import annotations

import json
from typing import Iterable

from deal_finder.agents.base import Agent
from deal_finder.data.rss_feeds import ScrapedDeal, fetch_rss_deals
from deal_finder.schemas import DealSelection


class ScannerAgent(Agent):
    """Fetch RSS deals and ask a frontier model for strict JSON selection."""

    name = "Scanner Agent"
    model = "gpt-4o-mini"

    system_prompt = (
        "You identify the five most detailed product deals with clear prices. "
        "Respond only as JSON with the shape {\"deals\": [{\"product_description\": "
        "string, \"price\": number, \"url\": string}]}."
    )

    def fetch(self, known_urls: set[str] | None = None) -> list[ScrapedDeal]:
        known_urls = known_urls or set()
        self.log("fetching RSS deal candidates")
        return [deal for deal in fetch_rss_deals() if deal.url not in known_urls]

    def build_prompt(self, scraped_deals: Iterable[ScrapedDeal]) -> str:
        deals_text = "\n\n".join(deal.describe() for deal in scraped_deals)
        return (
            "Select exactly five promising product deals with clear prices. "
            "Rewrite each description around the product itself, not coupon terms.\n\n"
            f"Deals:\n\n{deals_text}\n\nRespond only in JSON."
        )

    def scan(self, known_urls: set[str] | None = None) -> DealSelection | None:
        scraped = self.fetch(known_urls=known_urls)
        if not scraped:
            self.log("no unseen RSS deals found")
            return None

        self.log("calling OpenAI for structured deal selection")
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_prompt(scraped)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return parse_deal_selection(content)


def parse_deal_selection(content: str) -> DealSelection:
    """Parse and validate scanner JSON output."""

    payload = json.loads(content)
    selection = DealSelection.model_validate(payload)
    selection.deals = [deal for deal in selection.deals if deal.price > 0]
    return selection

