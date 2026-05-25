"""Simulation framework for high-dimensional two-sample covariance tests."""

from .methods import (
    li_chen_exact,
    max_bootstrap,
    ours_c_bootstrap,
    ours_l2_bootstrap,
    ours_l2_simplified,
    wang_normal_reference,
)

__all__ = [
    "li_chen_exact",
    "max_bootstrap",
    "ours_c_bootstrap",
    "ours_l2_bootstrap",
    "ours_l2_simplified",
    "wang_normal_reference",
]

