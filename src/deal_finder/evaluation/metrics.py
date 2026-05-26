"""Metrics for price prediction and deal ranking."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    """Return mean absolute error for price predictions."""

    _validate_lengths(actual, predicted)
    return mean(abs(a - p) for a, p in zip(actual, predicted))


def root_mean_squared_error(actual: list[float], predicted: list[float]) -> float:
    """Return root mean squared error for price predictions."""

    _validate_lengths(actual, predicted)
    mse = mean((a - p) ** 2 for a, p in zip(actual, predicted))
    return mse**0.5


def median_absolute_percentage_error(actual: list[float], predicted: list[float]) -> float:
    """Return median absolute percentage error, ignoring zero actuals."""

    _validate_lengths(actual, predicted)
    errors = [abs((a - p) / a) for a, p in zip(actual, predicted) if a != 0]
    return median(errors) if errors else 0.0


def discount_score(list_price: float, estimated_price: float) -> float:
    """Score how much estimated fair value exceeds the listed deal price."""

    if list_price <= 0:
        return 0.0
    return max(0.0, estimated_price - list_price)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Compact summary for model comparison tables."""

    model_name: str
    mae: float
    rmse: float
    mdape: float
    records: int


def summarize_predictions(
    model_name: str,
    actual: list[float],
    predicted: list[float],
) -> EvaluationSummary:
    """Build a reusable summary row for experiments."""

    return EvaluationSummary(
        model_name=model_name,
        mae=mean_absolute_error(actual, predicted),
        rmse=root_mean_squared_error(actual, predicted),
        mdape=median_absolute_percentage_error(actual, predicted),
        records=len(actual),
    )


def _validate_lengths(actual: list[float], predicted: list[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lists must have equal length")
    if not actual:
        raise ValueError("metrics require at least one prediction")

