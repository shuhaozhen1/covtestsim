"""Test-statistic implementations for the clean simulation restart."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import chi2, norm

from .target import adaptive_target_transform
from .utils import (
    MethodResult,
    Timer,
    apply_transform,
    batched_bootstrap_count,
    center_rows,
    clip_positive,
    covariance_coordinate_residuals,
    diagonal_diagnostics,
    sample_cov_n,
)


def ours_l2_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    B: int,
    alpha: float = 0.05,
    transform: np.ndarray | None = None,
    studentized: bool = True,
    rng: np.random.Generator | None = None,
    method_name: str | None = None,
    batch_size: int = 250,
) -> MethodResult:
    """Studentized or raw L2 multiplier-bootstrap covariance test."""

    rng = np.random.default_rng() if rng is None else rng
    with Timer() as timer:
        xt = apply_transform(x, transform)
        yt = apply_transform(y, transform)
        h1, s1v, _, _ = covariance_coordinate_residuals(xt)
        h2, s2v, _, _ = covariance_coordinate_residuals(yt)
        n1, d = h1.shape
        n2 = h2.shape[0]
        neff = n1 * n2 / (n1 + n2)
        w = np.sqrt(neff) * (s1v - s2v)
        diag = neff * (np.sum(h1 * h1, axis=0) / n1**2 + np.sum(h2 * h2, axis=0) / n2**2)
        diag_clipped = clip_positive(diag)

        if studentized:
            observed = float(np.mean((w * w) / diag_clipped))
        else:
            observed = float(np.mean(w * w))

        def draw_and_score(m: int) -> np.ndarray:
            e1 = rng.standard_normal((m, n1))
            e2 = rng.standard_normal((m, n2))
            wb = np.sqrt(neff) * ((e1 @ h1) / n1 - (e2 @ h2) / n2)
            if studentized:
                return np.mean((wb * wb) / diag_clipped[None, :], axis=1)
            return np.mean(wb * wb, axis=1)

        ge, total = batched_bootstrap_count(B, batch_size, draw_and_score, observed)
        p_value = ge / total if total else float("nan")

    diagnostics = diagonal_diagnostics(diag)
    diagnostics.update(
        {
            "n_eff": float(neff),
            "d": int(d),
            "bootstrap_ge": int(ge),
            "bootstrap_total": int(total),
            "elapsed_sec": float(timer.elapsed),
        }
    )
    name = method_name or ("Ours-I" if studentized else "Raw-L2")
    return MethodResult(name, observed, p_value, bool(p_value <= alpha), diagnostics)


def ours_c_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    B: int,
    alpha: float = 0.05,
    rho_tar: float = 0.5,
    rng: np.random.Generator | None = None,
    batch_size: int = 250,
    method_name: str = "Ours-C",
) -> MethodResult:
    """Adaptive target-transformed studentized L2 bootstrap."""

    transform, target_diag = adaptive_target_transform(x, y, rho_tar=rho_tar)
    result = ours_l2_bootstrap(
        x,
        y,
        B=B,
        alpha=alpha,
        transform=transform,
        studentized=True,
        rng=rng,
        method_name=method_name,
        batch_size=batch_size,
    )
    result.diagnostics.update(target_diag)
    return result


def max_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    B: int,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    batch_size: int = 250,
) -> MethodResult:
    """Max-type covariance-coordinate multiplier-bootstrap test."""

    rng = np.random.default_rng() if rng is None else rng
    with Timer() as timer:
        h1, s1v, _, _ = covariance_coordinate_residuals(x)
        h2, s2v, _, _ = covariance_coordinate_residuals(y)
        n1, d = h1.shape
        n2 = h2.shape[0]
        neff = n1 * n2 / (n1 + n2)
        w = np.sqrt(neff) * (s1v - s2v)
        diag = neff * (np.sum(h1 * h1, axis=0) / n1**2 + np.sum(h2 * h2, axis=0) / n2**2)
        diag_clipped = clip_positive(diag)
        scale = np.sqrt(diag_clipped)
        observed = float(np.max(np.abs(w) / scale))

        def draw_and_score(m: int) -> np.ndarray:
            e1 = rng.standard_normal((m, n1))
            e2 = rng.standard_normal((m, n2))
            wb = np.sqrt(neff) * ((e1 @ h1) / n1 - (e2 @ h2) / n2)
            return np.max(np.abs(wb) / scale[None, :], axis=1)

        ge, total = batched_bootstrap_count(B, batch_size, draw_and_score, observed)
        p_value = ge / total if total else float("nan")

    diagnostics = diagonal_diagnostics(diag)
    diagnostics.update(
        {
            "n_eff": float(neff),
            "d": int(d),
            "bootstrap_ge": int(ge),
            "bootstrap_total": int(total),
            "elapsed_sec": float(timer.elapsed),
        }
    )
    return MethodResult("Max-bootstrap", observed, p_value, bool(p_value <= alpha), diagnostics)


def _center_gram_features(kernel: np.ndarray) -> np.ndarray:
    row_mean = kernel.mean(axis=1, keepdims=True)
    col_mean = kernel.mean(axis=0, keepdims=True)
    grand = float(kernel.mean())
    return kernel - row_mean - col_mean + grand


def wang_gram_traces(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Trace quantities for Wang's induced covariance matrices.

    The induced observations are vec(x_i x_i^T), but all traces are computed
    through n-by-n Gram matrices with entries (x_i^T x_j)^2.
    """

    x0 = center_rows(x)
    y0 = center_rows(y)
    n1 = x0.shape[0]
    n2 = y0.shape[0]
    k11 = (x0 @ x0.T) ** 2
    k22 = (y0 @ y0.T) ** 2
    k12 = (x0 @ y0.T) ** 2
    g11 = _center_gram_features(k11)
    g22 = _center_gram_features(k22)
    g12 = _center_gram_features(k12)

    tr1 = float(np.trace(g11) / (n1 - 1))
    tr2 = float(np.trace(g22) / (n2 - 1))
    tr1_sq = float(np.trace(g11 @ g11) / (n1 - 1) ** 2)
    tr2_sq = float(np.trace(g22 @ g22) / (n2 - 1) ** 2)
    tr1_cu = float(np.trace(g11 @ g11 @ g11) / (n1 - 1) ** 3)
    tr2_cu = float(np.trace(g22 @ g22 @ g22) / (n2 - 1) ** 3)
    tr12 = float(np.sum(g12 * g12) / ((n1 - 1) * (n2 - 1)))
    tr1_sq_2 = float(np.trace(g11 @ g12 @ g12.T) / ((n1 - 1) ** 2 * (n2 - 1)))
    tr1_2_sq = float(np.trace(g22 @ g12.T @ g12) / ((n1 - 1) * (n2 - 1) ** 2))
    return {
        "tr1": tr1,
        "tr2": tr2,
        "tr1_sq": tr1_sq,
        "tr2_sq": tr2_sq,
        "tr1_cu": tr1_cu,
        "tr2_cu": tr2_cu,
        "tr12": tr12,
        "tr1_sq_2": tr1_sq_2,
        "tr1_2_sq": tr1_2_sq,
    }


def wang_normal_reference(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> MethodResult:
    """Wang normal-reference / three-cumulant matched chi-square test."""

    with Timer() as timer:
        x0 = center_rows(x)
        y0 = center_rows(y)
        n1 = x0.shape[0]
        n2 = y0.shape[0]
        s1 = (x0.T @ x0) / n1
        s2 = (y0.T @ y0) / n2
        traces = wang_gram_traces(x, y)

        statistic = float(np.sum((s1 - s2) ** 2) - traces["tr1"] / n1 - traces["tr2"] / n2)
        tr1_sq_unb = ((n1 - 1) ** 2 / ((n1 - 2) * (n1 + 1))) * (
            traces["tr1_sq"] - traces["tr1"] ** 2 / (n1 - 1)
        )
        tr2_sq_unb = ((n2 - 1) ** 2 / ((n2 - 2) * (n2 + 1))) * (
            traces["tr2_sq"] - traces["tr2"] ** 2 / (n2 - 1)
        )
        k2_hat = 2.0 * (
            tr1_sq_unb / (n1 * (n1 - 1))
            + 2.0 * traces["tr12"] / (n1 * n2)
            + tr2_sq_unb / (n2 * (n2 - 1))
        )

        tr1_cu_unb = ((n1 - 1) ** 4 / ((n1**2 + n1 - 6) * (n1**2 - 2 * n1 - 3))) * (
            traces["tr1_cu"]
            - 3.0 * traces["tr1"] * traces["tr1_sq"] / (n1 - 1)
            + 2.0 * traces["tr1"] ** 3 / (n1 - 1) ** 2
        )
        tr2_cu_unb = ((n2 - 1) ** 4 / ((n2**2 + n2 - 6) * (n2**2 - 2 * n2 - 3))) * (
            traces["tr2_cu"]
            - 3.0 * traces["tr2"] * traces["tr2_sq"] / (n2 - 1)
            + 2.0 * traces["tr2"] ** 3 / (n2 - 1) ** 2
        )
        tr1_sq_2_unb = ((n1 - 1) / ((n1 - 2) * (n1 + 1))) * (
            (n1 - 1) * traces["tr1_sq_2"] - traces["tr12"] * traces["tr1"]
        )
        tr1_2_sq_unb = ((n2 - 1) / ((n2 - 2) * (n2 + 1))) * (
            (n2 - 1) * traces["tr1_2_sq"] - traces["tr12"] * traces["tr2"]
        )
        k3_hat = 8.0 * (
            (n1 - 2) * tr1_cu_unb / (n1**2 * (n1 - 1) ** 2)
            + 3.0 * tr1_sq_2_unb / (n1**2 * n2)
            + 3.0 * tr1_2_sq_unb / (n1 * n2**2)
            + (n2 - 2) * tr2_cu_unb / (n2**2 * (n2 - 1) ** 2)
        )

        valid = bool(k2_hat > 0 and k3_hat > 0 and np.isfinite(k2_hat) and np.isfinite(k3_hat))
        if valid:
            beta0 = -2.0 * k2_hat**2 / k3_hat
            beta1 = k3_hat / (4.0 * k2_hat)
            df = 8.0 * k2_hat**3 / k3_hat**2
            threshold = (statistic - beta0) / beta1
            p_value = 1.0 if threshold <= 0 else float(chi2.sf(threshold, df))
        else:
            beta0 = beta1 = df = threshold = float("nan")
            p_value = float("nan")

    diagnostics = {
        "k2_hat": float(k2_hat),
        "k3_hat": float(k3_hat),
        "beta0": float(beta0),
        "beta1": float(beta1),
        "df_hat": float(df),
        "chi2_threshold": float(threshold),
        "valid_cumulants": valid,
        "elapsed_sec": float(timer.elapsed),
    }
    return MethodResult("Wang-NR", statistic, p_value, bool(p_value <= alpha), diagnostics)


def li_chen_a_component(x: np.ndarray) -> float:
    """Exact Li-Chen A_n component from equation (2.1)."""

    n = x.shape[0]
    if n < 4:
        raise ValueError("Li-Chen A_n requires at least four observations.")
    g = np.asarray(x, dtype=float) @ np.asarray(x, dtype=float).T
    np.fill_diagonal(g, 0.0)
    total_off = float(g.sum())
    row_sum = g.sum(axis=1)
    row_sq = np.sum(g * g, axis=1)

    term1 = float(np.sum(g * g)) / (n * (n - 1))
    term2_num = float(np.sum(row_sum * row_sum - row_sq))
    term2 = 2.0 * term2_num / (n * (n - 1) * (n - 2))

    incident = 2.0 * row_sum[:, None] + 2.0 * row_sum[None, :] - 2.0 * g
    allowed_sum = total_off - incident
    term3_num = float(np.sum(g * allowed_sum))
    term3 = term3_num / (n * (n - 1) * (n - 2) * (n - 3))
    return term1 - term2 + term3


def li_chen_c_component(x: np.ndarray, y: np.ndarray) -> float:
    """Exact Li-Chen C_n1,n2 component from equation (2.2)."""

    n1 = x.shape[0]
    n2 = y.shape[0]
    if n1 < 2 or n2 < 2:
        raise ValueError("Li-Chen C component requires at least two observations per group.")
    g = np.asarray(x, dtype=float) @ np.asarray(y, dtype=float).T
    row_sum = g.sum(axis=1)
    col_sum = g.sum(axis=0)
    row_sq = np.sum(g * g, axis=1)
    col_sq = np.sum(g * g, axis=0)
    total = float(g.sum())

    term1 = float(np.sum(g * g)) / (n1 * n2)
    term2a = float(np.sum(col_sum * col_sum - col_sq)) / (n1 * n2 * (n1 - 1))
    term2b = float(np.sum(row_sum * row_sum - row_sq)) / (n1 * n2 * (n2 - 1))
    allowed = total - row_sum[:, None] - col_sum[None, :] + g
    term4 = float(np.sum(g * allowed)) / (n1 * n2 * (n1 - 1) * (n2 - 1))
    return term1 - term2a - term2b + term4


def li_chen_exact(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> MethodResult:
    """Li-Chen whole covariance-matrix U-statistic test."""

    with Timer() as timer:
        a1 = li_chen_a_component(np.asarray(x, dtype=float))
        a2 = li_chen_a_component(np.asarray(y, dtype=float))
        c12 = li_chen_c_component(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        statistic_raw = float(a1 + a2 - 2.0 * c12)
        denominator_raw = 2.0 * a1 / y.shape[0] + 2.0 * a2 / x.shape[0]
        denominator = max(float(denominator_raw), np.finfo(float).tiny)
        statistic = float(statistic_raw / denominator)
        p_value = float(norm.sf(statistic))

    diagnostics = {
        "A1": float(a1),
        "A2": float(a2),
        "C12": float(c12),
        "T_raw": statistic_raw,
        "denominator_raw": float(denominator_raw),
        "denominator_clipped": float(denominator),
        "denominator_was_clipped": bool(denominator_raw <= 0),
        "elapsed_sec": float(timer.elapsed),
    }
    return MethodResult("Li-Chen", statistic, p_value, bool(p_value <= alpha), diagnostics)


def ours_l2_simplified(
    x: np.ndarray,
    y: np.ndarray,
    B: int,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    batch_size: int = 250,
) -> MethodResult:
    """One-sample simplified statistic for n1 much larger than n2."""

    rng = np.random.default_rng() if rng is None else rng
    with Timer() as timer:
        h2, s2v, _, _ = covariance_coordinate_residuals(y)
        s1 = sample_cov_n(x)
        s2 = sample_cov_n(y)
        n2, d = h2.shape
        tri = np.triu_indices(s1.shape[0])
        w = np.sqrt(n2) * (s1[tri] - s2[tri])
        diag = clip_positive(np.sum(h2 * h2, axis=0) / n2)
        observed = float(np.mean((w * w) / diag))

        def draw_and_score(m: int) -> np.ndarray:
            e = rng.standard_normal((m, n2))
            wb = (e @ h2) / np.sqrt(n2)
            return np.mean((wb * wb) / diag[None, :], axis=1)

        ge, total = batched_bootstrap_count(B, batch_size, draw_and_score, observed)
        p_value = ge / total if total else float("nan")

    diagnostics = diagonal_diagnostics(diag)
    diagnostics.update(
        {
            "d": int(d),
            "bootstrap_ge": int(ge),
            "bootstrap_total": int(total),
            "elapsed_sec": float(timer.elapsed),
        }
    )
    return MethodResult("Ours-I-simplified", observed, p_value, bool(p_value <= alpha), diagnostics)


MAIN_METHODS: Mapping[str, object] = {
    "Ours-I": ours_l2_bootstrap,
    "Ours-C": ours_c_bootstrap,
    "Max-bootstrap": max_bootstrap,
    "Wang-NR": wang_normal_reference,
    "Li-Chen": li_chen_exact,
}
