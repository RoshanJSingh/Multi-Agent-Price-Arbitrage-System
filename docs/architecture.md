# Architecture

The system is organized around a seven-agent workflow:

1. Scanner Agent ingests RSS deal feeds and asks a frontier model for strict JSON deal candidates.
2. Frontier Agent retrieves similar Amazon products from ChromaDB and asks a GPT-4-class model for a contextual price estimate.
3. Specialist Agent calls a Modal-hosted LoRA fine-tuned LLM for product-price estimation.
4. Random Forest Agent predicts price from 384-D sentence-transformer embeddings.
5. Ensemble Agent blends specialist, frontier, and Random Forest estimates.
6. Planner Agent ranks deals by estimated discount and persists surfaced opportunities.
7. Messaging Agent sends notifications through Pushover or Twilio when a deal clears threshold.

The dashboard uses the same planner factory as the CLI/runtime code so demos and experiments exercise the production workflow.

