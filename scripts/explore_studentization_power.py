"""Explore variance-heterogeneous alternatives where studentization helps.

The diagnostic is separate from the main simulation CLI.  It searches
low-variance dense-block alternatives under strong nuisance variance
heterogeneity, comparing the studentized L2 statistic with an otherwise
identical raw L2 statistic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from covtestsim.covariances import equicorrelation
from covtestsim.covariances import make_positive_definite
from covtestsim.seeds import BASE_SEED
from covtestsim.studies import Scenario, run_repetition, summarize_rejections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore studentization-power diagnostics.")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--profile", default="diagnostic")
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--B", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--include-target", action="store_true")
    parser.add_argument("--target-rhos", default="0.5")
    parser.add_argument("--block-sizes", default="40,60,80")
    parser.add_argument("--high-variances", default="25,100,400")
    parser.add_argument("--rho-grid", default="0,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--baseline-rho", type=float, default=0.0)
    return parser.parse_args()


def _lowvar_block_cov(
    p: int,
    block_size: int,
    high_variance: float,
    baseline_rho: float,
    rho_block: float,
) -> np.ndarray:
    variances = np.ones(p)
    variances[block_size:] = high_variance
    std = np.sqrt(variances)
    corr = equicorrelation(p, baseline_rho)
    block = np.arange(block_size)
    sub = corr[np.ix_(block, block)].copy()
    offdiag = ~np.eye(block_size, dtype=bool)
    sub[offdiag] = rho_block
    corr[np.ix_(block, block)] = sub
    return make_positive_definite((std[:, None] * corr) * std[None, :])


def _parse_int_grid(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _parse_float_grid(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def build_scenarios(
    include_target: bool,
    target_rhos: tuple[float, ...],
    block_grid: tuple[int, ...],
    high_variance_grid: tuple[float, ...],
    rho_grid: tuple[float, ...],
    baseline_rho: float,
) -> list[Scenario]:
    p = 200
    n1 = 120
    n2 = 150
    methods = ("Ours-I", "Raw-L2", "Max-bootstrap", "Wang-NR", "Li-Chen")
    if include_target:
        target_methods = tuple(f"Ours-C-rho{str(rho).replace('.', 'p')}" for rho in target_rhos)
        methods = ("Ours-I", *target_methods, "Raw-L2", "Max-bootstrap", "Wang-NR", "Li-Chen")

    scenarios: list[Scenario] = []
    for block_size in block_grid:
        for high_variance in high_variance_grid:
            sigma1 = _lowvar_block_cov(p, block_size, high_variance, baseline_rho, baseline_rho)
            for rho_block in rho_grid:
                sigma2 = _lowvar_block_cov(p, block_size, high_variance, baseline_rho, rho_block)
                scenarios.append(
                    Scenario(
                        study="study_power_studentized_search",
                        scenario_id=(
                            f"studentized_lowblock_p{p}_b{block_size}"
                            f"_hv{high_variance:g}_rho{rho_block:g}"
                        ).replace(".", "p"),
                        p=p,
                        n1=n1,
                        n2=n2,
                        sigma1=sigma1,
                        sigma2=sigma2,
                        family="diagonal_high_variance_low_block",
                        rho0=baseline_rho,
                        rho_alt=rho_block,
                        methods=methods,
                        innovation="gaussian",
                        design="studentized_low_variance_dense_block_search",
                        alternative="null" if rho_block == baseline_rho else "alternative",
                        extra={
                            "block_size": block_size,
                            "high_variance": high_variance,
                            "rho_block": rho_block,
                            "baseline_rho": baseline_rho,
                        },
                    )
                )
    return scenarios


def write_outputs(raw: pd.DataFrame, out_dir: Path, profile: str) -> None:
    suffix = f"study_power_studentized_search_{profile}"
    raw_dir = out_dir / "raw"
    summary_dir = out_dir / "summary"
    table_dir = out_dir / "tables"
    for directory in (raw_dir, summary_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = raw.sort_values(["block_size", "high_variance", "rho_block", "rep", "method"])
    raw.to_csv(raw_dir / f"{suffix}_raw.csv", index=False)
    summary = summarize_rejections(raw)
    meta_cols = ["scenario_id", "block_size", "high_variance", "rho_block"]
    meta = raw[meta_cols].drop_duplicates("scenario_id")
    summary = summary.merge(meta, on="scenario_id", how="left")
    summary.to_csv(summary_dir / f"{suffix}_summary_long.csv", index=False)

    index_cols = [
        "block_size",
        "high_variance",
        "rho_block",
        "alternative",
        "R",
    ]
    wide = summary.pivot_table(
        index=index_cols,
        columns="method",
        values="rejection_rate",
        aggfunc="first",
    ).reset_index()
    wide.to_csv(summary_dir / f"{suffix}_power_wide.csv", index=False)

    alt = wide[wide["rho_block"].gt(0)].copy()
    ours_c_cols = sorted(col for col in alt.columns if col.startswith("Ours-C("))
    ours_cols = [col for col in ["Ours-I", *ours_c_cols] if col in alt.columns]
    method_cols = [
        col
        for col in [*ours_cols, "Raw-L2", "Max-bootstrap", "Wang-NR", "Li-Chen"]
        if col in alt.columns
    ]
    if method_cols:
        benchmark_cols = [col for col in method_cols if col not in set(ours_cols)]
        alt["best_ours"] = alt[ours_cols].max(axis=1)
        alt["best_benchmark"] = alt[benchmark_cols].max(axis=1)
        alt["studentized_gain_vs_raw"] = alt["Ours-I"] - alt["Raw-L2"]
        alt["gain_vs_best_benchmark"] = alt["best_ours"] - alt["best_benchmark"]
        alt.sort_values(
            ["studentized_gain_vs_raw", "gain_vs_best_benchmark", "best_ours"],
            ascending=[False, False, False],
        ).to_csv(summary_dir / f"{suffix}_top_gains.csv", index=False)


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios(
        args.include_target,
        _parse_float_grid(args.target_rhos),
        _parse_int_grid(args.block_sizes),
        _parse_float_grid(args.high_variances),
        _parse_float_grid(args.rho_grid),
        args.baseline_rho,
    )
    tasks: list[tuple[Any, int]] = [
        (scenario, rep) for scenario in scenarios for rep in range(1, args.R + 1)
    ]
    nested = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(run_repetition)(
            scenario,
            rep,
            args.B,
            args.alpha,
            args.batch_size,
            args.base_seed,
        )
        for scenario, rep in tasks
    )
    raw = pd.DataFrame([row for chunk in nested for row in chunk])
    write_outputs(raw, ROOT / args.out_dir, args.profile)
    print(
        f"Wrote diagnostic search for R={args.R}, B={args.B}, "
        f"{raw['scenario_id'].nunique()} scenarios, {len(raw)} rows."
    )


if __name__ == "__main__":
    main()
