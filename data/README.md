# Data

The project is designed around Amazon Reviews 2023 product metadata from Hugging Face. Raw data is intentionally excluded from git because local exports and model artifacts can become large.

Use the data loader in `deal_finder.data.amazon_loader` to stream, normalize, and cap records for experiments. The default project baseline targets 40,000 curated product rows.
