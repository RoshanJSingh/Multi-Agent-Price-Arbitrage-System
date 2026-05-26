import unittest

from deal_finder.data.amazon_loader import normalize_price, normalize_record


class DataLoaderTests(unittest.TestCase):
    def test_normalize_price_accepts_currency_text(self):
        self.assertEqual(normalize_price("$1,249.99"), 1249.99)

    def test_normalize_record_drops_incomplete_rows(self):
        self.assertIsNone(normalize_record({"title": "Missing price", "description": ["long enough text"]}))

    def test_normalize_record_builds_product_record(self):
        record = normalize_record(
            {
                "title": "Wireless Noise Cancelling Headphones",
                "description": ["Premium headphones for travel and work."],
                "features": ["Bluetooth with long battery life and adaptive ANC."],
                "details": "Includes case, cable, and USB-C charging.",
                "price": "$199.99",
                "main_category": "Electronics",
                "parent_asin": "ASIN123",
            }
        )
        self.assertIsNotNone(record)
        self.assertEqual(record.category, "Electronics")
        self.assertEqual(record.price, 199.99)


if __name__ == "__main__":
    unittest.main()

