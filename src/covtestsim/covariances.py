"""Covariance generators and sampling helpers."""

from __future__ import annotations

import numpy as np


def equicorrelation(p: int, rho: float, scale: float = 1.0) -> np.ndarray:
    if rho <= -1.0 / (p - 1) or rho >= 1.0:
        raise ValueError(f"Equicorrelation rho={rho} is not positive definite for p={p}.")
    mat = np.full((p, p), rho, dtype=float)
    np.fill_diagonal(mat, 1.0)
    return scale * mat


def toeplitz_covariance(p: int, rho: float) -> np.ndarray:
    if abs(rho) >= 1:
        raise ValueError("Toeplitz AR(1) rho must satisfy |rho| < 1.")
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def make_positive_definite(sigma: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    sigma = (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, floor)
    return (eigvecs * eigvals) @ eigvecs.T


def scaled_equicorrelation(
    p: int,
    rho: float,
    scales: tuple[float, float, float] = (0.8, 1.0, 1.25),
) -> np.ndarray:
    cuts = np.array_split(np.arange(p), 3)
    std = np.empty(p)
    for block, scale in zip(cuts, scales, strict=True):
        std[block] = scale
    corr = equicorrelation(p, rho)
    return (std[:, None] * corr) * std[None, :]


def scaled_equicorrelation_block_alt(
    p: int,
    rho: float = 0.3,
    delta: float = 0.12,
    scales: tuple[float, float, float] = (0.8, 1.0, 1.25),
) -> np.ndarray:
    """Moderately heterogeneous dense block perturbation for Study 4."""

    cuts = np.array_split(np.arange(p), 3)
    corr = equicorrelation(p, rho)
    block = cuts[1]
    offdiag = np.ones((block.size, block.size), dtype=bool)
    np.fill_diagonal(offdiag, False)
    sub = corr[np.ix_(block, block)]
    sub[offdiag] = np.minimum(sub[offdiag] + delta, 0.95)
    corr[np.ix_(block, block)] = sub

    std = np.empty(p)
    for idx, scale in zip(cuts, scales, strict=True):
        std[idx] = scale
    return make_positive_definite((std[:, None] * corr) * std[None, :])


def covariance_from_family(family: str, p: int, rho: float) -> np.ndarray:
    if family == "equicorrelation":
        return equicorrelation(p, rho)
    if family == "toeplitz":
        return toeplitz_covariance(p, rho)
    if family == "identity":
        return np.eye(p)
    raise ValueError(f"Unknown covariance family: {family}")


def draw_samples(
    rng: np.random.Generator,
    n: int,
    sigma: np.ndarray,
    innovation: str = "gaussian",
) -> np.ndarray:
    """Draw n observations with covariance sigma.

    Non-Gaussian innovations are standardized to mean zero and variance one
    before applying the Cholesky factor.
    """

    p = sigma.shape[0]
    chol = np.linalg.cholesky(make_positive_definite(sigma))
    if innovation == "gaussian":
        z = rng.standard_normal((n, p))
    elif innovation == "chisq1":
        z = (rng.chisquare(df=1, size=(n, p)) - 1.0) / np.sqrt(2.0)
    elif innovation == "t5":
        z = rng.standard_t(df=5, size=(n, p)) / np.sqrt(5.0 / 3.0)
    elif innovation == "laplace":
        z = rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=(n, p))
    else:
        raise ValueError(f"Unknown innovation: {innovation}")
    return z @ chol.T
