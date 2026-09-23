# Agentic AI Deal Finder

Multi-agent price arbitrage system for finding underpriced e-commerce products from deal feeds, estimating fair market value with a hybrid AI/ML ensemble, and surfacing high-confidence opportunities in a Gradio dashboard.

## Project Highlights

- Engineered a 7-agent workflow with scanner, frontier, specialist, random forest, ensemble, planner, and messaging agents.
- Built RSS ingestion and strict JSON deal selection for repeatable downstream pricing.
- Combined a LoRA fine-tuned pricing model, GPT-4-class frontier pricing, and Random Forest regression into an ensemble estimator.
- Added a ChromaDB-backed RAG pipeline using 384-dimensional `all-MiniLM-L6-v2` sentence-transformer embeddings.
- Designed the data pipeline around 40k+ Amazon 2023 product records from Hugging Face for curation, normalization, and evaluation baselines.
- Shipped an interactive Gradio dashboard for logs, discovered deals, estimated discounts, and embedding-space inspection.

## Repository Structure

```text
src/deal_finder/
  agents/        Multi-agent orchestration and pricing agents
  app/           Gradio dashboard entrypoint
  data/          Amazon 2023 loading and normalization utilities
  evaluation/    Metrics for model and deal-quality baselines
  rag/           ChromaDB vector store and embedding pipeline
  training/      LoRA fine-tuning configuration helpers
docs/            Architecture and evaluation notes
tests/           Unit tests for parsing, scoring, and orchestration helpers
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m deal_finder.app.gradio_dashboard
```

External services are configured through environment variables. Copy `.env.example` to `.env` for local development.

## Validation

```bash
python -m unittest discover -s tests
```

The unit tests cover data normalization, JSON scanner parsing, price extraction, ensemble fallback behavior, and planner persistence.
