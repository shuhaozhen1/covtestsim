"""Study definitions and execution helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .covariances import (
    covariance_from_family,
    draw_samples,
    equicorrelation,
    make_positive_definite,
)
from .methods import (
    li_chen_exact,
    max_bootstrap,
    ours_c_bootstrap,
    ours_l2_bootstrap,
    ours_l2_simplified,
    wang_normal_reference,
)
from .seeds import BASE_SEED, stable_seed
from .utils import sample_cov_n


@dataclass(slots=True)
class Scenario:
    study: str
    scenario_id: str
    p: int
    n1: int
    n2: int
    sigma1: np.ndarray
    sigma2: np.ndarray
    family: str
    rho0: float | None
    rho_alt: float | None
    methods: tuple[str, ...]
    innovation: str = "gaussian"
    design: str = ""
    alternative: str = "null"
    extra: dict[str, Any] = field(default_factory=dict)


PROFILE_DEFAULTS = {
    "smoke": {"R": 3, "B": 10, "batch_size": 10},
    "debug": {"R": 50, "B": 500, "batch_size": 100},
    "final": {"R": None, "B": 2000, "batch_size": 250},
}

MAIN_METHODS = ("Ours-I", "Ours-C", "Max-bootstrap", "Wang-NR", "Li-Chen")
SIZE_INNOVATIONS = ("gaussian", "chisq1", "t5")
TARGET_SIZE_INNOVATIONS = ("gaussian", "chisq1", "laplace")
TARGET_RHO_GRID = (0.3, 0.5, 0.7)
TARGET_GRID_METHODS = (
    "Ours-I",
    "Ours-C-rho0p3",
    "Ours-C-rho0p5",
    "Ours-C-rho0p7",
    "Max-bootstrap",
    "Wang-NR",
    "Li-Chen",
)


def _scenario_id(*parts: object) -> str:
    return "_".join(str(part).replace(".", "p").replace("-", "m") for part in parts)


def _study1_scenarios(p: int = 100) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for rho in (0.0, 0.5, 0.9):
        sigma = equicorrelation(p, rho)
        scenarios.append(
            Scenario(
                study="study1",
                scenario_id=_scenario_id("size", "eq", p, rho),
                p=p,
                n1=120,
                n2=120,
                sigma1=sigma,
                sigma2=sigma,
                family="equicorrelation",
                rho0=rho,
                rho_alt=rho,
                methods=MAIN_METHODS,
                design="null_size",
            )
        )
    for rho in (0.1, 0.5, 0.9):
        sigma = covariance_from_family("toeplitz", p, rho)
        scenarios.append(
            Scenario(
                study="study1",
                scenario_id=_scenario_id("size", "toep", p, rho),
                p=p,
                n1=120,
                n2=120,
                sigma1=sigma,
                sigma2=sigma,
                family="toeplitz",
                rho0=rho,
                rho_alt=rho,
                methods=MAIN_METHODS,
                design="null_size",
            )
        )
    return scenarios


def _study1_dist_scenarios(p: int = 100) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for base in _study1_scenarios(p):
        for innovation in SIZE_INNOVATIONS:
            scenarios.append(
                Scenario(
                    study="study1_dist",
                    scenario_id=_scenario_id(
                        "size_dist",
                        innovation,
                        base.family,
                        p,
                        base.rho0,
                    ),
                    p=base.p,
                    n1=base.n1,
                    n2=base.n2,
                    sigma1=base.sigma1,
                    sigma2=base.sigma2,
                    family=base.family,
                    rho0=base.rho0,
                    rho_alt=base.rho_alt,
                    methods=MAIN_METHODS,
                    innovation=innovation,
                    design="null_size_by_innovation",
                    alternative="null",
                )
            )
    return scenarios


def _study1_hd_scenarios(p: int = 200) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for rho in (0.1, 0.5, 0.9):
        sigma = equicorrelation(p, rho)
        for innovation in SIZE_INNOVATIONS:
            scenarios.append(
                Scenario(
                    study="study1_hd",
                    scenario_id=_scenario_id(
                        "size_hd",
                        innovation,
                        "equicorrelation",
                        p,
                        rho,
                    ),
                    p=p,
                    n1=150,
                    n2=150,
                    sigma1=sigma,
                    sigma2=sigma,
                    family="equicorrelation",
                    rho0=rho,
                    rho_alt=rho,
                    methods=MAIN_METHODS,
                    innovation=innovation,
                    design="high_dimensional_null_size",
                    alternative="null",
                )
            )
    return scenarios


def _study_size_target_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for p in (100, 200):
        for family in ("equicorrelation", "toeplitz"):
            for rho in (0.1, 0.5, 0.9):
                sigma = covariance_from_family(family, p, rho)
                for n1, n2 in ((80, 100), (120, 150)):
                    for innovation in TARGET_SIZE_INNOVATIONS:
                        scenarios.append(
                            Scenario(
                                study="study_size_target",
                                scenario_id=_scenario_id(
                                    "size_target",
                                    innovation,
                                    family,
                                    p,
                                    n1,
                                    n2,
                                    f"rho{rho}",
                                ),
                                p=p,
                                n1=n1,
                                n2=n2,
                                sigma1=sigma,
                                sigma2=sigma,
                                family=family,
                                rho0=rho,
                                rho_alt=rho,
                                methods=TARGET_GRID_METHODS,
                                innovation=innovation,
                                design="target_rho_grid_null_size",
                                alternative="null",
                            )
                        )
    return scenarios


def _study_size_target_toeplitz_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for p in (100, 200):
        sigma = covariance_from_family("toeplitz", p, 0.5)
        for n1, n2 in ((80, 100), (120, 150)):
            for innovation in TARGET_SIZE_INNOVATIONS:
                scenarios.append(
                    Scenario(
                        study="study_size_target_toeplitz",
                        scenario_id=_scenario_id(
                            "size_target",
                            innovation,
                            "toeplitz",
                            p,
                            n1,
                            n2,
                            "rho0p5",
                        ),
                        p=p,
                        n1=n1,
                        n2=n2,
                        sigma1=sigma,
                        sigma2=sigma,
                        family="toeplitz",
                        rho0=0.5,
                        rho_alt=0.5,
                        methods=TARGET_GRID_METHODS,
                        innovation=innovation,
                        design="target_rho_grid_null_size",
                        alternative="null",
                    )
                )
    return scenarios


def _study_size_smalln_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for p in (100, 200):
        for family in ("equicorrelation", "toeplitz"):
            for rho in (0.1, 0.5, 0.9):
                sigma = covariance_from_family(family, p, rho)
                for innovation in TARGET_SIZE_INNOVATIONS:
                    scenarios.append(
                        Scenario(
                            study="study_size_smalln",
                            scenario_id=_scenario_id(
                                "size_smalln",
                                innovation,
                                family,
                                p,
                                60,
                                60,
                                f"rho{rho}",
                            ),
                            p=p,
                            n1=60,
                            n2=60,
                            sigma1=sigma,
                            sigma2=sigma,
                            family=family,
                            rho0=rho,
                            rho_alt=rho,
                            methods=TARGET_GRID_METHODS,
                            innovation=innovation,
                            design="small_equal_sample_null_size",
                            alternative="null",
                        )
                    )
    return scenarios


def _study_size_unbalanced_ratio_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    rho = 0.5
    for p in (100, 200):
        for family in ("equicorrelation", "toeplitz"):
            sigma = covariance_from_family(family, p, rho)
            for n1, n2 in ((300, 60), (180, 60), (120, 60), (90, 60)):
                for innovation in TARGET_SIZE_INNOVATIONS:
                    scenarios.append(
                        Scenario(
                            study="study_size_unbalanced_ratio",
                            scenario_id=_scenario_id(
                                "size_unbalanced",
                                innovation,
                                family,
                                p,
                                n1,
                                n2,
                                "rho0p5",
                            ),
                            p=p,
                            n1=n1,
                            n2=n2,
                            sigma1=sigma,
                            sigma2=sigma,
                            family=family,
                            rho0=rho,
                            rho_alt=rho,
                            methods=TARGET_GRID_METHODS,
                            innovation=innovation,
                            design="unbalanced_ratio_null_size",
                            alternative="null",
                        )
                    )
    return scenarios


def _study2_scenarios(p: int = 100, include_diagnostic: bool = False) -> list[Scenario]:
    scenarios: list[Scenario] = []
    grids = [
        (0.5, (0.5, 0.6, 0.7, 0.8, 0.9), "main"),
        (0.9, (0.9, 0.8, 0.7, 0.6, 0.5), "main"),
    ]
    if include_diagnostic:
        grids.append((0.0, (0.0, 0.025, 0.05, 0.075, 0.1), "diagnostic"))
    for rho0, alts, role in grids:
        sigma1 = equicorrelation(p, rho0)
        for rho_alt in alts:
            sigma2 = equicorrelation(p, rho_alt)
            scenarios.append(
                Scenario(
                    study="study2",
                    scenario_id=_scenario_id("eqpower", role, p, rho0, rho_alt),
                    p=p,
                    n1=120,
                    n2=120,
                    sigma1=sigma1,
                    sigma2=sigma2,
                    family="equicorrelation",
                    rho0=rho0,
                    rho_alt=rho_alt,
                    methods=MAIN_METHODS,
                    design=role,
                    alternative="null" if rho0 == rho_alt else "alternative",
                )
            )
    return scenarios


def _study3_scenarios(p: int = 100) -> list[Scenario]:
    scenarios: list[Scenario] = []
    sigma1 = np.eye(p)
    for rho in (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1):
        sigma2 = equicorrelation(p, rho)
        scenarios.append(
            Scenario(
                study="study3",
                scenario_id=_scenario_id("identity_to_eq", p, rho),
                p=p,
                n1=120,
                n2=120,
                sigma1=sigma1,
                sigma2=sigma2,
                family="identity_to_equicorrelation",
                rho0=0.0,
                rho_alt=rho,
                methods=MAIN_METHODS,
                design="dense_identity_alt",
                alternative="null" if rho == 0 else "alternative",
            )
        )
    return scenarios


def _study_power_target_eq0_scenarios() -> list[Scenario]:
    p = 200
    n1 = 120
    n2 = 150
    rho0 = 0.0
    sigma1 = equicorrelation(p, rho0)
    rho_grid = (
        0.0,
        0.0005,
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
    )
    scenarios: list[Scenario] = []
    for rho_alt in rho_grid:
        sigma2 = equicorrelation(p, rho_alt)
        scenarios.append(
            Scenario(
                study="study_power_target_eq0",
                scenario_id=_scenario_id("eq0_power_target", p, n1, n2, rho_alt),
                p=p,
                n1=n1,
                n2=n2,
                sigma1=sigma1,
                sigma2=sigma2,
                family="equicorrelation",
                rho0=rho0,
                rho_alt=rho_alt,
                methods=TARGET_GRID_METHODS,
                innovation="gaussian",
                design="identity_to_equicorrelation_target_power",
                alternative="null" if rho_alt == rho0 else "alternative",
            )
        )
    return scenarios


def _study_power_target_eq05_scenarios() -> list[Scenario]:
    p = 200
    n1 = 120
    n2 = 150
    rho0 = 0.5
    sigma1 = equicorrelation(p, rho0)
    rho_grid = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    scenarios: list[Scenario] = []
    for rho_alt in rho_grid:
        sigma2 = equicorrelation(p, rho_alt)
        scenarios.append(
            Scenario(
                study="study_power_target_eq05",
                scenario_id=_scenario_id("eq05_power_target", p, n1, n2, rho_alt),
                p=p,
                n1=n1,
                n2=n2,
                sigma1=sigma1,
                sigma2=sigma2,
                family="equicorrelation",
                rho0=rho0,
                rho_alt=rho_alt,
                methods=TARGET_GRID_METHODS,
                innovation="gaussian",
                design="equicorrelation_target_power_rho0p5",
                alternative="null" if rho_alt == rho0 else "alternative",
            )
        )
    return scenarios


def _heterogeneous_equicorrelation(p: int, rho: float, scale_c: float = 1.0) -> np.ndarray:
    variances = scale_c * np.linspace(1.0, np.sqrt(float(p)), p)
    std = np.sqrt(variances)
    corr = equicorrelation(p, rho)
    return (std[:, None] * corr) * std[None, :]


def _heterogeneous_low_variance_block(
    p: int,
    rho0: float,
    rho_block: float,
    block_size: int,
) -> np.ndarray:
    variances = np.linspace(1.0, np.sqrt(float(p)), p)
    std = np.sqrt(variances)
    corr = equicorrelation(p, rho0)
    block = np.arange(block_size)
    sub = corr[np.ix_(block, block)].copy()
    offdiag = ~np.eye(block_size, dtype=bool)
    sub[offdiag] = rho_block
    corr[np.ix_(block, block)] = sub
    return make_positive_definite((std[:, None] * corr) * std[None, :])


def _study_power_variance_profile_scenarios() -> list[Scenario]:
    p = 200
    n1 = 120
    n2 = 150
    scale_grid = (1.0, 1.02, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0)
    scenarios: list[Scenario] = []
    for rho0 in (0.5,):
        sigma1 = _heterogeneous_equicorrelation(p, rho0, scale_c=1.0)
        for scale_c in scale_grid:
            sigma2 = _heterogeneous_equicorrelation(p, rho0, scale_c=scale_c)
            scenarios.append(
                Scenario(
                    study="study_power_variance_profile",
                    scenario_id=_scenario_id("var_profile_power", p, n1, n2, rho0, scale_c),
                    p=p,
                    n1=n1,
                    n2=n2,
                    sigma1=sigma1,
                    sigma2=sigma2,
                    family="heterogeneous_equicorrelation",
                    rho0=rho0,
                    rho_alt=scale_c,
                    methods=TARGET_GRID_METHODS,
                    innovation="gaussian",
                    design="heterogeneous_variance_scale_power",
                    alternative="null" if scale_c == 1.0 else "alternative",
                    extra={
                        "scale_c": scale_c,
                        "variance_min_group1": 1.0,
                        "variance_max_group1": float(np.sqrt(float(p))),
                        "variance_min_group2": scale_c,
                        "variance_max_group2": float(scale_c * np.sqrt(float(p))),
                    },
                )
            )
    return scenarios


def _study_power_lowvar_block_scenarios() -> list[Scenario]:
    p = 200
    n1 = 120
    n2 = 150
    rho0 = 0.5
    block_size = 180
    rho_block_grid = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    sigma1 = _heterogeneous_low_variance_block(p, rho0, rho0, block_size)
    scenarios: list[Scenario] = []
    for rho_block in rho_block_grid:
        sigma2 = _heterogeneous_low_variance_block(p, rho0, rho_block, block_size)
        scenarios.append(
            Scenario(
                study="study_power_lowvar_block",
                scenario_id=_scenario_id("lowvar_block_power", p, n1, n2, rho0, block_size, rho_block),
                p=p,
                n1=n1,
                n2=n2,
                sigma1=sigma1,
                sigma2=sigma2,
                family="heterogeneous_low_variance_block",
                rho0=rho0,
                rho_alt=rho_block,
                methods=TARGET_GRID_METHODS,
                innovation="gaussian",
                design="low_variance_block_correlation_power",
                alternative="null" if rho_block == rho0 else "alternative",
                extra={
                    "block_size": block_size,
                    "variance_min": 1.0,
                    "variance_max": float(np.sqrt(float(p))),
                    "rho_block": rho_block,
                },
            )
        )
    return scenarios


def _study4_scenarios() -> list[Scenario]:
    methods = ("Ours-I", "Raw-L2")
    p = 100
    n1 = 120
    n2 = 120
    rho = 0.5
    multipliers = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0)
    sigma1 = equicorrelation(p, rho)
    scenarios: list[Scenario] = []
    for multiplier in multipliers:
        std = np.ones(p)
        std[0] = np.sqrt(multiplier)
        sigma2 = (std[:, None] * sigma1) * std[None, :]
        scenarios.append(
            Scenario(
                study="study4",
                scenario_id=_scenario_id("studentized_raw_var1", p, multiplier),
                p=p,
                n1=n1,
                n2=n2,
                sigma1=sigma1,
                sigma2=sigma2,
                family="equicorrelation_variance_spike",
                rho0=rho,
                rho_alt=multiplier,
                methods=methods,
                design="first_coordinate_variance",
                alternative="null" if multiplier == 1.0 else "alternative",
                extra={"variance_multiplier": multiplier, "background_rho": rho},
            )
        )
    return scenarios


def _study5_scenarios(include_supplement: bool = False) -> list[Scenario]:
    methods = ("Ours-I", "Ours-I-simplified")
    p = 50
    scenarios = [
        Scenario(
            study="study5",
            scenario_id="unbalanced_toep_p50_n5000_100_rho0p8",
            p=p,
            n1=5000,
            n2=100,
            sigma1=covariance_from_family("toeplitz", p, 0.8),
            sigma2=covariance_from_family("toeplitz", p, 0.8),
            family="toeplitz",
            rho0=0.8,
            rho_alt=0.8,
            methods=methods,
            design="unbalanced_size_large_reference",
        )
    ]
    if include_supplement:
        scenarios.append(
            Scenario(
                study="study5",
                scenario_id="unbalanced_eq_p50_n1000_200_rho0p9",
                p=p,
                n1=1000,
                n2=200,
                sigma1=equicorrelation(p, 0.9),
                sigma2=equicorrelation(p, 0.9),
                family="equicorrelation",
                rho0=0.9,
                rho_alt=0.9,
                methods=methods,
                design="unbalanced_size_supplement",
            )
        )
    return scenarios


def _study6_scenarios() -> list[Scenario]:
    methods = ("Ours-I", "Max-bootstrap", "Wang-NR", "Li-Chen")
    p = 100
    sigma = equicorrelation(p, 0.5)
    scenarios: list[Scenario] = []
    for innovation in ("gaussian", "chisq1"):
        scenarios.append(
            Scenario(
                study="study6",
                scenario_id=_scenario_id("nongaussian_size", innovation),
                p=p,
                n1=120,
                n2=120,
                sigma1=sigma,
                sigma2=sigma,
                family="equicorrelation",
                rho0=0.5,
                rho_alt=0.5,
                methods=methods,
                innovation=innovation,
                design="nonnull_robustness_size",
            )
        )
    return scenarios


def _realdata_based_scenarios(cache_dir: str | Path | None = None) -> list[Scenario]:
    from .realdata import HALLMARK_GMT_URL, _candidate_by_id, download_file, load_candidate_program, parse_gmt

    cache = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[2] / "data_cache"
    gmt_path = download_file(HALLMARK_GMT_URL, cache / "h.all.v2025.1.Hs.symbols.gmt")
    gene_sets = parse_gmt(gmt_path)
    program = load_candidate_program(
        _candidate_by_id("BRCA_subtype_cell_cycle_core"),
        cache=cache,
        gene_sets=gene_sets,
        max_p=500,
    )
    if program is None:
        raise RuntimeError("Could not load the TCGA-BRCA cell-cycle core panel for real-data-based simulation.")

    x = program.x
    y = program.y
    sx = sample_cov_n(x)
    sy = sample_cov_n(y)
    pooled = make_positive_definite((program.n1 * sx + program.n2 * sy) / (program.n1 + program.n2), floor=1e-6)
    sx = make_positive_definite(sx, floor=1e-6)
    sy = make_positive_definite(sy, floor=1e-6)
    common_extra = {
        "realdata_source": "TCGA-BRCA",
        "realdata_contrast": "Basal-like_vs_Luminal_A",
        "realdata_panel": "Hallmark_E2F_targets_plus_G2M_checkpoint",
        "matched_e2f_genes": 187,
        "matched_g2m_genes": 183,
    }
    return [
        Scenario(
            study="realdata_sim",
            scenario_id="tcga_brca_cell_cycle_core_size_pooled",
            p=program.p,
            n1=program.n1,
            n2=program.n2,
            sigma1=pooled,
            sigma2=pooled,
            family="tcga_brca_cell_cycle_core",
            rho0=None,
            rho_alt=None,
            methods=MAIN_METHODS,
            innovation="gaussian",
            design="pooled_covariance_size",
            alternative="null",
            extra=common_extra,
        ),
        Scenario(
            study="realdata_sim",
            scenario_id="tcga_brca_cell_cycle_core_power_separate",
            p=program.p,
            n1=program.n1,
            n2=program.n2,
            sigma1=sx,
            sigma2=sy,
            family="tcga_brca_cell_cycle_core",
            rho0=None,
            rho_alt=None,
            methods=MAIN_METHODS,
            innovation="gaussian",
            design="subtype_specific_covariance_power",
            alternative="alternative",
            extra=common_extra,
        ),
    ]


def build_scenarios(
    study: str,
    include_supplement: bool = False,
    cache_dir: str | Path | None = None,
) -> list[Scenario]:
    if study == "study1":
        scenarios = _study1_scenarios(100)
        if include_supplement:
            scenarios += _study1_scenarios(200)
        return scenarios
    if study == "study1_dist":
        scenarios = _study1_dist_scenarios(100)
        if include_supplement:
            scenarios += _study1_dist_scenarios(200)
        return scenarios
    if study == "study1_hd":
        return _study1_hd_scenarios(200)
    if study == "study_size_target":
        return _study_size_target_scenarios()
    if study == "study_size_target_toeplitz":
        return _study_size_target_toeplitz_scenarios()
    if study == "study_size_smalln":
        return _study_size_smalln_scenarios()
    if study == "study_size_unbalanced_ratio":
        return _study_size_unbalanced_ratio_scenarios()
    if study == "study2":
        scenarios = _study2_scenarios(100, include_diagnostic=include_supplement)
        if include_supplement:
            scenarios += _study2_scenarios(200, include_diagnostic=False)
        return scenarios
    if study == "study3":
        return _study3_scenarios(100)
    if study == "study_power_target_eq0":
        return _study_power_target_eq0_scenarios()
    if study == "study_power_target_eq05":
        return _study_power_target_eq05_scenarios()
    if study == "study_power_variance_profile":
        return _study_power_variance_profile_scenarios()
    if study == "study_power_lowvar_block":
        return _study_power_lowvar_block_scenarios()
    if study == "study4":
        return _study4_scenarios()
    if study == "study5":
        return _study5_scenarios(include_supplement=include_supplement)
    if study == "study6":
        return _study6_scenarios()
    if study == "realdata_sim":
        return _realdata_based_scenarios(cache_dir=cache_dir)
    raise ValueError(f"Unknown study: {study}")


def final_repetitions(study: str) -> int:
    if study in {"study1_dist", "study3", "study4", "study5", "study6"}:
        return 1000
    if study == "study1":
        return 2000
    if study == "study1_hd":
        return 1000
    if study == "study2":
        return 1000
    if study == "realdata_sim":
        return 500
    if study in {
        "study_size_target",
        "study_size_target_toeplitz",
        "study_size_smalln",
        "study_size_unbalanced_ratio",
    }:
        return 1000
    if study == "study_power_target_eq0":
        return 100
    if study == "study_power_target_eq05":
        return 100
    if study == "study_power_variance_profile":
        return 200
    if study == "study_power_lowvar_block":
        return 200
    raise ValueError(f"Unknown study: {study}")


def _run_method(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    B: int,
    alpha: float,
    seed_boot: int,
    batch_size: int,
) -> tuple[Any, int]:
    rng = np.random.default_rng(seed_boot)
    if method == "Ours-I":
        return ours_l2_bootstrap(x, y, B=B, alpha=alpha, rng=rng, batch_size=batch_size), B
    if method == "Ours-C":
        return ours_c_bootstrap(x, y, B=B, alpha=alpha, rng=rng, batch_size=batch_size), B
    if method.startswith("Ours-C-rho"):
        rho_text = method.removeprefix("Ours-C-rho").replace("p", ".")
        rho_tar = float(rho_text)
        return (
            ours_c_bootstrap(
                x,
                y,
                B=B,
                alpha=alpha,
                rho_tar=rho_tar,
                rng=rng,
                batch_size=batch_size,
                method_name=f"Ours-C({rho_tar:.1f})",
            ),
            B,
        )
    if method == "Max-bootstrap":
        return max_bootstrap(x, y, B=B, alpha=alpha, rng=rng, batch_size=batch_size), B
    if method == "Wang-NR":
        return wang_normal_reference(x, y, alpha=alpha), 0
    if method == "Li-Chen":
        return li_chen_exact(x, y, alpha=alpha), 0
    if method == "Raw-L2":
        return (
            ours_l2_bootstrap(
                x,
                y,
                B=B,
                alpha=alpha,
                studentized=False,
                rng=rng,
                method_name="Raw-L2",
                batch_size=batch_size,
            ),
            B,
        )
    if method == "Ours-I-simplified":
        return ours_l2_simplified(x, y, B=B, alpha=alpha, rng=rng, batch_size=batch_size), B
    raise ValueError(f"Unknown method: {method}")


def run_repetition(
    scenario: Scenario,
    rep: int,
    B: int,
    alpha: float,
    batch_size: int,
    base_seed: int = BASE_SEED,
) -> list[dict[str, Any]]:
    seed_data = stable_seed(scenario.study, scenario.scenario_id, rep, "data", base_seed=base_seed)
    rng_data = np.random.default_rng(seed_data)
    x = draw_samples(rng_data, scenario.n1, scenario.sigma1, scenario.innovation)
    y = draw_samples(rng_data, scenario.n2, scenario.sigma2, scenario.innovation)

    rows: list[dict[str, Any]] = []
    for method in scenario.methods:
        seed_boot = stable_seed(
            scenario.study, scenario.scenario_id, rep, method, "boot", base_seed=base_seed
        )
        result, b_used = _run_method(method, x, y, B=B, alpha=alpha, seed_boot=seed_boot, batch_size=batch_size)
        diagnostics = dict(result.diagnostics)
        elapsed = diagnostics.get("elapsed_sec", np.nan)
        row: dict[str, Any] = {
            "study": scenario.study,
            "scenario_id": scenario.scenario_id,
            "p": scenario.p,
            "n1": scenario.n1,
            "n2": scenario.n2,
            "n1_over_n2": scenario.n1 / scenario.n2,
            "family": scenario.family,
            "rho0": scenario.rho0,
            "rho_alt": scenario.rho_alt,
            "design": scenario.design,
            "alternative": scenario.alternative,
            "innovation": scenario.innovation,
            "method": result.method,
            "rep": rep,
            "seed_data": seed_data,
            "seed_boot": seed_boot if b_used else "",
            "statistic": result.statistic,
            "p_value": result.p_value,
            "reject": result.reject,
            "alpha": alpha,
            "B": b_used,
            "diagnostics": json.dumps(diagnostics, sort_keys=True),
            "elapsed_sec": elapsed,
        }
        for key, value in diagnostics.items():
            if isinstance(value, (int, float, bool, str, np.integer, np.floating, np.bool_)):
                row[f"diag_{key}"] = value
        row.update(scenario.extra)
        rows.append(row)
    return rows


def run_study(
    study: str,
    profile: str = "smoke",
    out_dir: str | Path = "results",
    include_supplement: bool = False,
    R: int | None = None,
    B: int | None = None,
    alpha: float = 0.05,
    n_jobs: int = 1,
    base_seed: int = BASE_SEED,
    rep_start: int = 1,
    rep_end: int | None = None,
    append_existing: bool = False,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    defaults = PROFILE_DEFAULTS[profile]
    repetitions = R if R is not None else defaults["R"]
    if repetitions is None:
        repetitions = final_repetitions(study)
    if rep_start < 1:
        raise ValueError("rep_start must be at least 1.")
    if rep_end is None:
        rep_end = repetitions
    if rep_end < rep_start:
        raise ValueError("rep_end must be greater than or equal to rep_start.")
    if rep_end > repetitions:
        raise ValueError("rep_end cannot exceed the requested total repetitions R.")
    bootstrap_b = B if B is not None else defaults["B"]
    batch_size = int(defaults["batch_size"])
    scenarios = build_scenarios(study, include_supplement=include_supplement, cache_dir=cache_dir)

    tasks = [(scenario, rep) for scenario in scenarios for rep in range(rep_start, rep_end + 1)]
    if n_jobs == 1:
        nested = [
            run_repetition(scenario, rep, bootstrap_b, alpha, batch_size, base_seed=base_seed)
            for scenario, rep in tasks
        ]
    else:
        nested = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_repetition)(scenario, rep, bootstrap_b, alpha, batch_size, base_seed)
            for scenario, rep in tasks
        )
    rows = [row for chunk in nested for row in chunk]
    raw = pd.DataFrame(rows)
    if append_existing:
        raw_path = Path(out_dir) / "raw" / f"{study}_{profile}_raw.csv"
        if raw_path.exists():
            previous = pd.read_csv(raw_path)
            raw = pd.concat([previous, raw], ignore_index=True)
            raw = raw.drop_duplicates(["scenario_id", "method", "rep"], keep="last")
            raw = raw.sort_values(["scenario_id", "rep", "method"]).reset_index(drop=True)
    if "alternative" in raw.columns:
        raw["alternative"] = raw["alternative"].replace("", np.nan).fillna("null")
    write_outputs(raw, study=study, profile=profile, out_dir=out_dir)
    return raw


def _summary_group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "study",
        "scenario_id",
        "p",
        "n1",
        "n2",
        "n1_over_n2",
        "family",
        "rho0",
        "rho_alt",
        "design",
        "alternative",
        "innovation",
        "method",
        "alpha",
        "B",
    ]
    return [col for col in candidates if col in df.columns]


def summarize_rejections(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = _summary_group_columns(raw)
    summary = (
        raw.groupby(group_cols, dropna=False)
        .agg(
            R=("reject", "size"),
            rejection_rate=("reject", "mean"),
            mean_p_value=("p_value", "mean"),
            mean_statistic=("statistic", "mean"),
            median_statistic=("statistic", "median"),
            mean_elapsed_sec=("elapsed_sec", "mean"),
        )
        .reset_index()
    )
    summary["MCSE"] = np.sqrt(summary["rejection_rate"] * (1.0 - summary["rejection_rate"]) / summary["R"])
    return summary


def study5_closeness_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or raw["study"].iloc[0] != "study5":
        return pd.DataFrame()
    pivot = raw.pivot_table(
        index=["scenario_id", "rep"],
        columns="method",
        values=["statistic", "p_value", "reject"],
        aggfunc="first",
    )
    rows = []
    for scenario_id, block in pivot.groupby(level=0):
        block = block.droplevel(0)
        full_stat = block[("statistic", "Ours-I")]
        simp_stat = block[("statistic", "Ours-I-simplified")]
        full_p = block[("p_value", "Ours-I")]
        simp_p = block[("p_value", "Ours-I-simplified")]
        meta = raw.loc[raw["scenario_id"] == scenario_id].iloc[0]
        size_full = raw[(raw["scenario_id"] == scenario_id) & (raw["method"] == "Ours-I")]["reject"].mean()
        size_simp = raw[(raw["scenario_id"] == scenario_id) & (raw["method"] == "Ours-I-simplified")][
            "reject"
        ].mean()
        r = int(len(full_stat))
        rows.append(
            {
                "scenario_id": scenario_id,
                "n1": int(meta["n1"]),
                "n2": int(meta["n2"]),
                "n1_over_n2": float(meta["n1_over_n2"]),
                "covariance": f"{meta['family']}({meta['rho0']})",
                "R": r,
                "size_full": float(size_full),
                "MCSE_full": float(np.sqrt(size_full * (1 - size_full) / r)),
                "size_simp": float(size_simp),
                "MCSE_simp": float(np.sqrt(size_simp * (1 - size_simp) / r)),
                "median_abs_T_full_minus_simp": float(np.median(np.abs(full_stat - simp_stat))),
                "median_abs_p_full_minus_simp": float(np.median(np.abs(full_p - simp_p))),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(raw: pd.DataFrame, study: str, profile: str, out_dir: str | Path) -> None:
    raw = raw.copy()
    if "alternative" in raw.columns:
        raw["alternative"] = raw["alternative"].replace("", np.nan).fillna("null")
    out = Path(out_dir)
    raw_dir = out / "raw"
    summary_dir = out / "summary"
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    for directory in (raw_dir, summary_dir, tables_dir, figures_dir, out / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    suffix = f"{study}_{profile}"
    raw.to_csv(raw_dir / f"{suffix}_raw.csv", index=False)
    summary = summarize_rejections(raw)
    summary.to_csv(summary_dir / f"{suffix}_summary_long.csv", index=False)

    wide = summary.pivot_table(
        index=[
            col
            for col in [
                "study",
                "scenario_id",
                "p",
                "n1",
                "n2",
                "family",
                "rho0",
                "rho_alt",
                "design",
                "alternative",
                "innovation",
            ]
            if col in summary.columns
        ],
        columns="method",
        values="rejection_rate",
        aggfunc="first",
    ).reset_index()
    wide.to_csv(summary_dir / f"{suffix}_summary_wide.csv", index=False)

    selected_cols = [
        col
        for col in [
            "family",
            "rho0",
            "rho_alt",
            "design",
            "alternative",
            "innovation",
            "method",
            "R",
            "rejection_rate",
            "MCSE",
        ]
        if col in summary.columns
    ]
    summary[selected_cols].to_latex(
        tables_dir / f"{suffix}_summary_long.tex", index=False, float_format="%.3f", escape=True
    )
    wide.to_latex(tables_dir / f"{suffix}_summary_wide.tex", index=False, float_format="%.3f", escape=True)

    if study == "study5":
        close = study5_closeness_summary(raw)
        close.to_csv(summary_dir / f"{suffix}_closeness.csv", index=False)
        if not close.empty:
            close.to_latex(tables_dir / f"{suffix}_closeness.tex", index=False, float_format="%.3f", escape=True)

    write_paper_tables(summary, raw, study=study, profile=profile, summary_dir=summary_dir, tables_dir=tables_dir)
    write_plots(summary, study=study, profile=profile, figures_dir=figures_dir)


def _paper_table(summary: pd.DataFrame, columns: list[str], rename: dict[str, str]) -> pd.DataFrame:
    table = summary[columns].copy()
    table = table.rename(columns=rename)
    return table


def write_paper_tables(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    study: str,
    profile: str,
    summary_dir: Path,
    tables_dir: Path,
) -> None:
    suffix = f"{study}_{profile}"
    if study in {
        "study1",
        "study1_dist",
        "study1_hd",
        "study_size_target",
        "study_size_target_toeplitz",
        "study_size_smalln",
        "study_size_unbalanced_ratio",
    }:
        table = _paper_table(
            summary,
            [
                col
                for col in ["p", "n1", "n2", "innovation", "family", "rho0", "method", "R", "rejection_rate", "MCSE"]
                if col in summary.columns
            ],
            {
                "family": "covariance family",
                "rho0": "rho",
                "rejection_rate": "empirical size",
            },
        )
        table.to_csv(summary_dir / f"{suffix}_paper_size.csv", index=False)
        table.to_latex(tables_dir / f"{suffix}_paper_size.tex", index=False, float_format="%.3f", escape=True)
        index_cols = [
            col for col in ["p", "n1", "n2", "innovation", "covariance family", "rho"] if col in table.columns
        ]
        wide = table.pivot_table(
            index=index_cols,
            columns="method",
            values="empirical size",
            aggfunc="first",
        ).reset_index()
        wide.to_csv(summary_dir / f"{suffix}_paper_size_wide.csv", index=False)
        wide.to_latex(tables_dir / f"{suffix}_paper_size_wide.tex", index=False, float_format="%.3f", escape=True)
    elif study in {
        "study2",
        "study3",
        "study_power_target_eq0",
        "study_power_target_eq05",
        "study_power_variance_profile",
        "study_power_lowvar_block",
    }:
        table = _paper_table(
            summary,
            ["family", "rho0", "rho_alt", "method", "R", "rejection_rate", "MCSE"],
            {
                "rho0": "rho0",
                "rejection_rate": "rejection rate",
            },
        )
        table.to_csv(summary_dir / f"{suffix}_paper_power.csv", index=False)
        table.to_latex(tables_dir / f"{suffix}_paper_power.tex", index=False, float_format="%.3f", escape=True)
    elif study == "study4":
        table = _paper_table(
            summary,
            ["rho_alt", "method", "alternative", "R", "rejection_rate"],
            {
                "rho_alt": "variance multiplier",
                "method": "statistic type",
                "rejection_rate": "rejection rate",
            },
        ).sort_values(["variance multiplier", "statistic type"])
        table.to_csv(summary_dir / f"{suffix}_paper_studentized_raw.csv", index=False)
        table.to_latex(
            tables_dir / f"{suffix}_paper_studentized_raw.tex", index=False, float_format="%.3f", escape=True
        )
    elif study == "study5":
        close = study5_closeness_summary(raw)
        if not close.empty:
            close.to_csv(summary_dir / f"{suffix}_paper_unbalanced.csv", index=False)
            close.to_latex(
                tables_dir / f"{suffix}_paper_unbalanced.tex", index=False, float_format="%.3f", escape=True
            )
    elif study == "study6":
        table = _paper_table(
            summary,
            ["innovation", "method", "R", "rejection_rate", "MCSE"],
            {"rejection_rate": "empirical size"},
        )
        table.to_csv(summary_dir / f"{suffix}_paper_nongaussian.csv", index=False)
        table.to_latex(tables_dir / f"{suffix}_paper_nongaussian.tex", index=False, float_format="%.3f", escape=True)
    elif study == "realdata_sim":
        table = _paper_table(
            summary,
            ["design", "alternative", "method", "R", "rejection_rate", "MCSE", "mean_p_value"],
            {"rejection_rate": "rejection rate"},
        )
        table.to_csv(summary_dir / f"{suffix}_paper_realdata_based.csv", index=False)
        table.to_latex(
            tables_dir / f"{suffix}_paper_realdata_based.tex",
            index=False,
            float_format="%.3f",
            escape=True,
        )
        rows = []
        for method in MAIN_METHODS:
            block = summary[summary["method"].eq(method)]
            null = block[block["alternative"].eq("null") | block["design"].astype(str).str.contains("size")]
            alt = block[block["alternative"].eq("alternative")]
            if null.empty and alt.empty:
                continue
            rows.append(
                {
                    "method": method,
                    "size": float(null["rejection_rate"].iloc[0]) if not null.empty else np.nan,
                    "power": float(alt["rejection_rate"].iloc[0]) if not alt.empty else np.nan,
                }
            )
        compact = pd.DataFrame(rows)
        compact.to_csv(summary_dir / f"{suffix}_paper_realdata_based_compact.csv", index=False)
        compact.to_latex(
            tables_dir / f"{suffix}_paper_realdata_based_compact.tex",
            index=False,
            float_format="%.3f",
            escape=True,
        )


def write_plots(summary: pd.DataFrame, study: str, profile: str, figures_dir: Path) -> None:
    if (
        study
        not in {
            "study2",
            "study3",
            "study_power_target_eq0",
            "study_power_target_eq05",
            "study_power_variance_profile",
            "study_power_lowvar_block",
        }
        or summary.empty
    ):
        return
    import matplotlib.pyplot as plt

    style_map = {
        "Ours-C(0.3)": {"linestyle": "-", "marker": "o", "linewidth": 2.0},
        "Ours-C(0.5)": {"linestyle": "-", "marker": "s", "linewidth": 2.0},
        "Ours-C(0.7)": {"linestyle": "-", "marker": "^", "linewidth": 2.0},
        "Ours-I": {"linestyle": (0, (5, 2)), "marker": "D", "linewidth": 1.8},
        "Max-bootstrap": {"linestyle": "--", "marker": "v", "linewidth": 1.8},
        "Wang-NR": {"linestyle": "-.", "marker": "P", "linewidth": 1.8},
        "Li-Chen": {"linestyle": ":", "marker": "X", "linewidth": 2.0},
    }
    method_order = [
        "Ours-C(0.3)",
        "Ours-C(0.5)",
        "Ours-C(0.7)",
        "Ours-I",
        "Max-bootstrap",
        "Wang-NR",
        "Li-Chen",
    ]

    def plot_method_curves(block: pd.DataFrame, x_col: str) -> None:
        for method in method_order:
            method_block = block[block["method"].eq(method)]
            if method_block.empty:
                continue
            method_block = method_block.sort_values(x_col)
            plt.plot(
                method_block[x_col],
                method_block["rejection_rate"],
                label=method,
                **style_map.get(method, {"linestyle": "-", "marker": "o", "linewidth": 1.8}),
            )

    if study == "study2":
        for rho0, block in summary.groupby("rho0", dropna=False):
            plt.figure(figsize=(6.5, 4.2))
            plot_method_curves(block, "rho_alt")
            plt.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
            plt.xlabel(r"$\rho_{\mathrm{alt}}$")
            plt.ylabel("Rejection rate")
            plt.title(rf"Study 2: $\rho_0={rho0}$")
            plt.ylim(-0.02, 1.02)
            plt.legend(fontsize=8)
            plt.tight_layout()
            safe_rho = str(rho0).replace(".", "p")
            plt.savefig(figures_dir / f"{study}_{profile}_rho0_{safe_rho}.png", dpi=200)
            plt.close()
    elif study == "study3":
        plt.figure(figsize=(6.5, 4.2))
        plot_method_curves(summary, "rho_alt")
        plt.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
        plt.xlabel(r"$\rho$")
        plt.ylabel("Rejection rate")
        plt.title("Study 3: identity to equicorrelation")
        plt.ylim(-0.02, 1.02)
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{study}_{profile}_power.png", dpi=200)
        plt.close()
    elif study == "study_power_target_eq0":
        plt.figure(figsize=(6.5, 4.2))
        plot_method_curves(summary, "rho_alt")
        plt.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
        plt.xlabel(r"$\rho_1$")
        plt.ylabel("Rejection rate")
        plt.title(r"Power: $\rho_0=0$ to equicorrelation alternatives")
        plt.ylim(-0.02, 1.02)
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{study}_{profile}_power.png", dpi=200)
        plt.close()
    elif study == "study_power_target_eq05":
        plt.figure(figsize=(6.5, 4.2))
        plot_method_curves(summary, "rho_alt")
        plt.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0.5, color="gray", linewidth=0.8, linestyle=":")
        plt.xlabel(r"$\rho_1$")
        plt.ylabel("Rejection rate")
        plt.title(r"Power: equicorrelation alternatives around $\rho_0=0.5$")
        plt.ylim(-0.02, 1.02)
        plt.legend(fontsize=8, loc="lower left")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{study}_{profile}_power.png", dpi=200)
        plt.close()
    elif study == "study_power_variance_profile":
        for rho0, block in summary.groupby("rho0", dropna=False):
            plt.figure(figsize=(6.5, 4.2))
            plot_method_curves(block, "rho_alt")
            plt.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
            plt.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
            plt.xlabel(r"Scale multiplier $c$")
            plt.ylabel("Rejection rate")
            plt.title(rf"Heterogeneous variances: $\rho_0={rho0}$")
            plt.ylim(-0.02, 1.02)
            plt.legend(fontsize=8, loc="lower right")
            plt.tight_layout()
            safe_rho = str(rho0).replace(".", "p")
            plt.savefig(figures_dir / f"{study}_{profile}_rho0_{safe_rho}_power.png", dpi=200)
            plt.close()
    elif study == "study_power_lowvar_block":
        plt.figure(figsize=(6.5, 4.2))
        plot_method_curves(summary, "rho_alt")
        plt.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
        plt.axvline(0.5, color="gray", linewidth=0.8, linestyle=":")
        plt.xlabel(r"Within-block correlation $\rho_{\mathrm{block}}$")
        plt.ylabel("Rejection rate")
        plt.title(r"Low-variance dense block: $\rho_0=0.5$")
        plt.ylim(-0.02, 1.02)
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{study}_{profile}_power.png", dpi=200)
        plt.close()
