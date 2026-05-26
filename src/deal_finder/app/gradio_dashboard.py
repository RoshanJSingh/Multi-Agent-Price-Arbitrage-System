"""Gradio dashboard for the multi-agent deal finder."""

from __future__ import annotations

import logging
import queue
import threading
import time

import gradio as gr
import plotly.graph_objects as go

from deal_finder.agents.base import configure_logging
from deal_finder.framework import build_planner


class QueueHandler(logging.Handler):
    """Forward log records to the live Gradio console."""

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def make_plot() -> go.Figure:
    """Placeholder plot shown until a local vector store is populated."""

    fig = go.Figure()
    fig.update_layout(
        title="Product embedding space",
        height=420,
        margin={"r": 10, "b": 10, "l": 10, "t": 40},
    )
    return fig


def opportunity_table(planner) -> list[list[str]]:
    rows = []
    for opportunity in planner.memory.read():
        rows.append(
            [
                opportunity.deal.product_description,
                f"${opportunity.deal.price:.2f}",
                f"${opportunity.estimate:.2f}",
                f"${opportunity.discount:.2f}",
                str(opportunity.deal.url),
            ]
        )
    return rows


def create_dashboard():
    """Create the Gradio Blocks interface."""

    configure_logging()
    planner = build_planner()

    with gr.Blocks(title="Agentic AI Deal Finder", fill_width=True) as ui:
        gr.Markdown("# Agentic AI Deal Finder")
        gr.Markdown("Seven-agent workflow for RSS deal discovery, RAG pricing, ensemble scoring, and alerts.")

        run_button = gr.Button("Run scan", variant="primary")
        log_output = gr.HTML()
        table = gr.Dataframe(
            headers=["Deal", "Price", "Estimate", "Discount", "URL"],
            wrap=True,
            row_count=10,
            col_count=5,
        )
        plot = gr.Plot(value=make_plot())

        def run_once():
            log_queue: queue.Queue[str] = queue.Queue()
            handler = QueueHandler(log_queue)
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
            logging.getLogger("deal_finder").addHandler(handler)

            result_queue: queue.Queue[object] = queue.Queue()

            def worker():
                result_queue.put(planner.plan())

            threading.Thread(target=worker, daemon=True).start()
            logs: list[str] = []
            while result_queue.empty():
                while not log_queue.empty():
                    logs.append(log_queue.get())
                yield "<br>".join(logs[-20:]), opportunity_table(planner), plot
                time.sleep(0.2)
            while not log_queue.empty():
                logs.append(log_queue.get())
            yield "<br>".join(logs[-20:]), opportunity_table(planner), plot

        run_button.click(run_once, outputs=[log_output, table, plot])

    return ui


def main() -> None:
    create_dashboard().launch(share=False, inbrowser=True)


if __name__ == "__main__":
    main()

