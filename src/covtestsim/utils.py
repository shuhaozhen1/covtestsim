"""Shared numerical utilities for covariance-test simulations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np


EPS = np.finfo(float).eps


@dataclass(slots=True)
class MethodResult:
    method: str
    statistic: float
    p_value: float
    reject: bool
    diagnostics: dict[str, float | int | str | bool]


class Timer:
    """Small context timer used for per-method elapsed-time diagnostics."""

    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = perf_counter() - self.start


def center_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)


def sample_cov_n(x: np.ndarray) -> np.ndarray:
    xc = center_rows(x)
    return (xc.T @ xc) / x.shape[0]


def vech_indices(p: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(p)


def vech(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[vech_indices(matrix.shape[0])]


def covariance_coordinate_residuals(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return centered covariance-coordinate residuals.

    The covariance denominator is n, matching the proposed statistic in the
    manuscript. The residual matrix has one row per observation and one column
    per upper-triangular covariance coordinate.
    """

    xc = center_rows(x)
    n, p = xc.shape
    tri = vech_indices(p)
    cov = (xc.T @ xc) / n
    products = xc[:, tri[0]] * xc[:, tri[1]]
    residuals = products - cov[tri]
    return residuals, cov[tri], cov, tri


def clip_positive(values: np.ndarray, floor: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if floor is None:
        positive = values[values > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        floor = max(EPS, 1e-12 * scale)
    return np.maximum(values, floor)


def diagonal_diagnostics(d: np.ndarray) -> dict[str, float]:
    positive = d[d > 0]
    median = float(np.median(positive)) if positive.size else float("nan")
    mean = float(np.mean(positive)) if positive.size else float("nan")
    sd = float(np.std(positive)) if positive.size else float("nan")
    return {
        "studentizer_min": float(np.min(d)),
        "studentizer_median": median,
        "studentizer_cv": float(sd / mean) if mean and np.isfinite(mean) else float("nan"),
        "studentizer_nonpositive": int(np.sum(d <= 0)),
    }


def apply_transform(x: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    if transform is None:
        return np.asarray(x, dtype=float)
    return np.asarray(x, dtype=float) @ transform.T


def batched_bootstrap_count(
    b: int,
    batch_size: int,
    draw_and_score: Callable[[int], np.ndarray],
    observed: float,
) -> tuple[int, int]:
    """Count bootstrap scores at least as large as the observed statistic."""

    ge = 0
    total = 0
    while total < b:
        m = min(batch_size, b - total)
        scores = draw_and_score(m)
        ge += int(np.sum(scores >= observed))
        total += m
    return ge, total


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

