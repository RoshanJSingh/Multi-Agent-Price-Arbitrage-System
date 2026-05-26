"""LoRA fine-tuning configuration for the specialist pricing model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoraTrainingConfig:
    """Configuration used by the specialist pricing model experiments."""

    base_model: str = "meta-llama/Meta-Llama-3.1-8B"
    adapter_name: str = "amazon-price-pricer-lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_prompt_tokens: int = 180
    learning_rate: float = 2e-4
    epochs: int = 1
    train_records: int = 40_000


def build_pricing_prompt(title: str, description: str, price: float | None = None) -> str:
    """Create the instruction format shared by training and inference."""

    prompt = (
        "How much does this cost to the nearest dollar?\n\n"
        f"{title.strip()}\n{description.strip()}\n\nPrice is $"
    )
    if price is not None:
        prompt += f"{round(price)}.00"
    return prompt

