"""Forecast evaluation helpers: zero-safe metrics and time-based validation splits.

MAPE is deliberately NOT used: daily demand is frequently zero, which makes MAPE undefined or explosive.
WAPE (weighted absolute percentage error = sum|error| / sum(actual)) stays well defined whenever total demand > 0.
"""
from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    """sum|a-p| / sum(a). NaN when there was no demand at all in the evaluation window."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    denom = float(np.sum(np.abs(a)))
    return float(np.sum(np.abs(a - p)) / denom) if denom > 0 else float("nan")


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean(pred - actual): positive = over-forecasting."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(p - a)) if a.size else float("nan")


def score(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"mae": mae(actual, pred), "wape": wape(actual, pred), "rmse": rmse(actual, pred), "bias": bias(actual, pred),
            "n": int(np.asarray(actual).size)}


def time_split_index(n: int, horizon: int, min_train: int = 56, val_days: int = 56) -> int | None:
    """Index of the first validation day (cutoff). Training targets are strictly before it.

    The validation block is the most recent ``val_days`` (never random). Returns None when the series is
    too short to hold at least one full backtest of ``horizon`` days after ``min_train`` training days.
    """
    val_days = min(val_days, max(horizon, n // 4))
    cutoff = n - val_days
    if cutoff < min_train or cutoff + horizon > n:
        return None
    return cutoff


def backtest_origins(cutoff: int, n: int, horizon: int, step: int = 7) -> list[int]:
    """Forecast origins (index of the last observed day) for rolling-origin evaluation.

    Origin ``o`` forecasts days o+1..o+horizon; every origin satisfies o >= cutoff-1 and o+horizon <= n-1,
    so no forecast ever sees data after its origin and every target lies in the validation block.
    """
    first = cutoff - 1
    return [o for o in range(first, n - horizon, step)]
