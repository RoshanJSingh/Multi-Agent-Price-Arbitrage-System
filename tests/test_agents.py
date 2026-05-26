import json
import tempfile
import unittest

from deal_finder.agents.ensemble import EnsembleAgent
from deal_finder.agents.frontier import extract_price
from deal_finder.agents.memory import DealMemory
from deal_finder.agents.planner import PlannerAgent
from deal_finder.agents.scanner import parse_deal_selection
from deal_finder.schemas import DealSelection


class FixedPricer:
    def __init__(self, value):
        self.value = value

    def price(self, description):
        return self.value


class ScannerStub:
    def scan(self, known_urls=None):
        return DealSelection.model_validate(
            {
                "deals": [
                    {
                        "product_description": "Premium mechanical keyboard with aluminum frame and hot swap switches.",
                        "price": 80,
                        "url": "https://example.com/deal",
                    }
                ]
            }
        )


class MessengerStub:
    def __init__(self):
        self.sent = []

    def alert(self, opportunity):
        self.sent.append(opportunity)


class AgentTests(unittest.TestCase):
    def test_extract_price_from_model_text(self):
        self.assertEqual(extract_price("Price is $1,299.50"), 1299.50)

    def test_parse_deal_selection_validates_json(self):
        payload = {
            "deals": [
                {
                    "product_description": "A detailed product description with enough content.",
                    "price": 42.5,
                    "url": "https://example.com",
                }
            ]
        }
        selection = parse_deal_selection(json.dumps(payload))
        self.assertEqual(len(selection.deals), 1)

    def test_ensemble_uses_mean_without_blender_model(self):
        ensemble = EnsembleAgent(FixedPricer(100), FixedPricer(130), FixedPricer(160), blender_path="missing.joblib")
        self.assertEqual(ensemble.price("test product"), 130)

    def test_planner_persists_and_alerts_profitable_deal(self):
        with tempfile.TemporaryDirectory() as tmp:
            messenger = MessengerStub()
            planner = PlannerAgent(
                scanner=ScannerStub(),
                ensemble=FixedPricer(150),
                messenger=messenger,
                memory=DealMemory(f"{tmp}/memory.json"),
                deal_threshold=25,
            )
            opportunity = planner.plan()
            self.assertIsNotNone(opportunity)
            self.assertEqual(opportunity.discount, 70)
            self.assertEqual(len(messenger.sent), 1)


if __name__ == "__main__":
    unittest.main()

