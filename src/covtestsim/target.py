"""Target-transformation helpers for Ours-C."""

from __future__ import annotations

import numpy as np

from .covariances import equicorrelation, make_positive_definite
from .utils import sample_cov_n


def _matrix_power_spd(a: np.ndarray, power: float, eig_floor: float = 1e-10) -> np.ndarray:
    a = make_positive_definite(a, floor=eig_floor)
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, eig_floor)
    return (vecs * (vals**power)) @ vecs.T


def matrix_sqrt_spd(a: np.ndarray, eig_floor: float = 1e-10) -> np.ndarray:
    return _matrix_power_spd(a, 0.5, eig_floor=eig_floor)


def matrix_invsqrt_spd(a: np.ndarray, eig_floor: float = 1e-10) -> np.ndarray:
    return _matrix_power_spd(a, -0.5, eig_floor=eig_floor)


def adaptive_target_transform(
    x: np.ndarray,
    y: np.ndarray,
    rho_tar: float = 0.5,
    ridge_multiplier: float = 1e-6,
    eig_floor: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate C_hat = Sigma_tar^{1/2}(Sigma_pool_hat + lambda I)^{-1/2}."""

    n1, p = x.shape
    n2 = y.shape[0]
    s1 = sample_cov_n(x)
    s2 = sample_cov_n(y)
    pooled = (n1 * s1 + n2 * s2) / (n1 + n2)
    avg_var = float(np.mean(np.diag(pooled)))
    ridge = ridge_multiplier * max(avg_var, eig_floor)
    sigma_tar = equicorrelation(p, rho_tar)
    transform = matrix_sqrt_spd(sigma_tar, eig_floor=eig_floor) @ matrix_invsqrt_spd(
        pooled + ridge * np.eye(p), eig_floor=eig_floor
    )
    vals = np.linalg.eigvalsh(make_positive_definite(pooled + ridge * np.eye(p)))
    return transform, {
        "target_rho": float(rho_tar),
        "target_ridge": float(ridge),
        "pooled_min_eig": float(vals.min()),
        "pooled_max_eig": float(vals.max()),
        "pooled_cond": float(vals.max() / vals.min()),
    }


def oracle_target_transform(
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    n1: int,
    n2: int,
    rho_tar: float = 0.5,
    ridge_multiplier: float = 1e-6,
    eig_floor: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Deterministic C0 diagnostic transformation using population covariances."""

    p = sigma1.shape[0]
    pooled = (n1 * sigma1 + n2 * sigma2) / (n1 + n2)
    avg_var = float(np.mean(np.diag(pooled)))
    ridge = ridge_multiplier * max(avg_var, eig_floor)
    sigma_tar = equicorrelation(p, rho_tar)
    transform = matrix_sqrt_spd(sigma_tar, eig_floor=eig_floor) @ matrix_invsqrt_spd(
        pooled + ridge * np.eye(p), eig_floor=eig_floor
    )
    vals = np.linalg.eigvalsh(make_positive_definite(pooled + ridge * np.eye(p)))
    return transform, {
        "target_rho": float(rho_tar),
        "target_ridge": float(ridge),
        "oracle_pooled_min_eig": float(vals.min()),
        "oracle_pooled_max_eig": float(vals.max()),
        "oracle_pooled_cond": float(vals.max() / vals.min()),
    }

