# Agentic AI Deal Finder

This project looks for underpriced products in online deal feeds. It reads deals from RSS feeds, estimates what each product is really worth, and shows the best finds in a Gradio dashboard.

## How it works

The pipeline is split across seven agents:

- **Scanner** reads the RSS feeds and asks an LLM to pick the clearest deals, returned as strict JSON.
- **Specialist** prices a product with a LoRA fine-tuned model hosted on Modal.
- **Frontier** prices it with a GPT-4 class model, using similar products pulled from ChromaDB as context.
- **Random Forest** gives a classic ML price estimate.
- **Ensemble** blends the three estimates into one number.
- **Planner** runs the whole flow: scan, price, rank and save the results.
- **Messaging** sends an alert when a deal looks good enough.

The retrieval part uses ChromaDB with 384-dimensional `all-MiniLM-L6-v2` sentence embeddings. The data comes from about 40k Amazon 2023 product records on Hugging Face, which I cleaned and normalized for training and evaluation.

The dashboard shows the agent logs, the deals found so far, the estimated discount on each one, and a view of the embedding space.

## Repository layout

```text
src/deal_finder/
  agents/        the seven agents and how they are wired together
  app/           Gradio dashboard
  data/          loading and cleaning the Amazon 2023 data
  evaluation/    metrics for the models and for deal quality
  rag/           ChromaDB vector store and embeddings
  training/      LoRA fine-tuning config
docs/            architecture and evaluation notes
tests/           unit tests
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate        # on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m deal_finder.app.gradio_dashboard
```

API keys and other settings are read from environment variables. Copy `.env.example` to `.env` and fill it in.

## Tests

```bash
python -m unittest discover -s tests
```

The tests cover data cleaning, parsing the scanner's JSON, price extraction, what the ensemble does when one model fails, and how the planner saves its state.
