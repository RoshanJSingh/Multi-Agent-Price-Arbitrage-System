# Resume Alignment

Resume point coverage:

- 7-agent workflow: implemented under `src/deal_finder/agents` and wired in `src/deal_finder/framework.py`.
- RSS ingestion and JSON outputs: `data/rss_feeds.py` and `agents/scanner.py`.
- Gradio dashboard: `app/gradio_dashboard.py`.
- Hybrid AI/ML prediction: `agents/specialist.py`, `agents/frontier.py`, `agents/random_forest.py`, and `agents/ensemble.py`.
- LoRA fine-tuning: `training/lora_config.py` and frontier fine-tuning notebooks.
- RAG with ChromaDB and 384-D embeddings: `rag/vector_store.py`.
- 40k+ Amazon 2023 records: `data/amazon_loader.py` and `scripts/build_vectorstore.py`.

