"""Small timing diagnostics for p=200 simulation bottlenecks.

This script is intentionally separate from the main simulation runner.  It does
not change seeds, method definitions, or result-generation logic; it only times
selected kernels so that long final runs can be planned more safely.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from covtestsim.covariances import equicorrelation, draw_samples
from covtestsim.methods import (
    li_chen_exact,
    max_bootstrap,
    ours_c_bootstrap,
    ours_l2_bootstrap,
    wang_normal_reference,
)
from covtestsim.utils import clip_positive, covariance_coordinate_residuals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark simulation acceleration options.")
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--n1", type=int, default=120)
    parser.add_argument("--n2", type=int, default=150)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--B", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[100, 250, 500, 1000])
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--out", default="results/summary/acceleration_benchmark.md")
    return parser.parse_args()


def time_call(fn: Callable[[], object], repeat: int) -> tuple[float, list[float]]:
    values: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        fn()
        values.append(perf_counter() - start)
    return median(values), values


def make_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    sigma = equicorrelation(args.p, args.rho)
    x = draw_samples(rng, args.n1, sigma, innovation="gaussian")
    y = draw_samples(rng, args.n2, sigma, innovation="gaussian")
    return x, y


def numpy_bootstrap_kernel(
    x: np.ndarray,
    y: np.ndarray,
    B: int,
    batch_size: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    h1, s1v, _, _ = covariance_coordinate_residuals(x)
    h2, s2v, _, _ = covariance_coordinate_residuals(y)
    n1, d = h1.shape
    n2 = h2.shape[0]
    neff = n1 * n2 / (n1 + n2)
    w = np.sqrt(neff) * (s1v - s2v)
    diag = neff * (np.sum(h1 * h1, axis=0) / n1**2 + np.sum(h2 * h2, axis=0) / n2**2)
    diag = clip_positive(diag)
    observed = float(np.mean((w * w) / diag))
    ge = 0
    total = 0
    while total < B:
        m = min(batch_size, B - total)
        e1 = rng.standard_normal((m, n1))
        e2 = rng.standard_normal((m, n2))
        wb = np.sqrt(neff) * ((e1 @ h1) / n1 - (e2 @ h2) / n2)
        scores = np.mean((wb * wb) / diag[None, :], axis=1)
        ge += int(np.sum(scores >= observed))
        total += m
    return ge / total


def torch_bootstrap_kernel(
    x: np.ndarray,
    y: np.ndarray,
    B: int,
    batch_size: int,
    seed: int,
    device: str,
) -> float:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("torch is not installed")
    import torch

    rng = np.random.default_rng(seed)
    h1_np, s1v, _, _ = covariance_coordinate_residuals(x)
    h2_np, s2v, _, _ = covariance_coordinate_residuals(y)
    n1, _ = h1_np.shape
    n2 = h2_np.shape[0]
    neff = n1 * n2 / (n1 + n2)
    w = np.sqrt(neff) * (s1v - s2v)
    diag_np = neff * (
        np.sum(h1_np * h1_np, axis=0) / n1**2 + np.sum(h2_np * h2_np, axis=0) / n2**2
    )
    diag_np = clip_positive(diag_np)
    observed = float(np.mean((w * w) / diag_np))

    torch_device = torch.device(device)
    h1 = torch.as_tensor(h1_np, dtype=torch.float64, device=torch_device)
    h2 = torch.as_tensor(h2_np, dtype=torch.float64, device=torch_device)
    diag = torch.as_tensor(diag_np, dtype=torch.float64, device=torch_device)
    neff_sqrt = float(np.sqrt(neff))
    ge = 0
    total = 0
    while total < B:
        m = min(batch_size, B - total)
        e1_np = rng.standard_normal((m, n1))
        e2_np = rng.standard_normal((m, n2))
        e1 = torch.as_tensor(e1_np, dtype=torch.float64, device=torch_device)
        e2 = torch.as_tensor(e2_np, dtype=torch.float64, device=torch_device)
        wb = neff_sqrt * ((e1 @ h1) / n1 - (e2 @ h2) / n2)
        scores = torch.mean((wb * wb) / diag[None, :], dim=1)
        ge += int(torch.sum(scores >= observed).detach().cpu().item())
        total += m
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return ge / total


def format_seconds(values: list[float]) -> str:
    return ", ".join(f"{x:.3f}" for x in values)


def main() -> None:
    args = parse_args()
    x, y = make_data(args)
    d = args.p * (args.p + 1) // 2

    lines: list[str] = []
    lines.append("# Acceleration Benchmark")
    lines.append("")
    lines.append(
        f"Design: `p={args.p}`, `d={d}`, `n1={args.n1}`, `n2={args.n2}`, "
        f"`rho={args.rho}`, `B={args.B}`, `repeat={args.repeat}`."
    )
    lines.append("")

    method_specs: list[tuple[str, Callable[[], object]]] = [
        ("Ours-I", lambda: ours_l2_bootstrap(x, y, B=args.B, rng=np.random.default_rng(args.seed + 11))),
        ("Ours-C(0.5)", lambda: ours_c_bootstrap(x, y, B=args.B, rng=np.random.default_rng(args.seed + 12))),
        ("Max-bootstrap", lambda: max_bootstrap(x, y, B=args.B, rng=np.random.default_rng(args.seed + 13))),
        ("Wang-NR", lambda: wang_normal_reference(x, y)),
        ("Li-Chen", lambda: li_chen_exact(x, y)),
    ]
    lines.append("## End-to-End Method Timings")
    lines.append("")
    lines.append("| Method | Median seconds | Runs |")
    lines.append("| --- | ---: | --- |")
    for name, fn in method_specs:
        med, values = time_call(fn, args.repeat)
        lines.append(f"| {name} | {med:.3f} | {format_seconds(values)} |")
    lines.append("")

    lines.append("## Bootstrap Batch-Size Timing")
    lines.append("")
    lines.append("| Batch size | Median seconds | Runs |")
    lines.append("| ---: | ---: | --- |")
    for batch_size in args.batch_sizes:
        med, values = time_call(
            lambda bs=batch_size: numpy_bootstrap_kernel(x, y, args.B, bs, args.seed + 101),
            args.repeat,
        )
        lines.append(f"| {batch_size} | {med:.3f} | {format_seconds(values)} |")
    lines.append("")

    lines.append("## NumPy versus Torch Bootstrap Kernel")
    lines.append("")
    lines.append("| Backend | Device | Median seconds | Runs | Note |")
    lines.append("| --- | --- | ---: | --- | --- |")
    best_batch = max(args.batch_sizes)
    med, values = time_call(
        lambda: numpy_bootstrap_kernel(x, y, args.B, best_batch, args.seed + 201),
        args.repeat,
    )
    lines.append(f"| NumPy | CPU | {med:.3f} | {format_seconds(values)} | reference |")

    if importlib.util.find_spec("torch") is None:
        lines.append("| Torch | CPU | NA | NA | torch not installed |")
        lines.append("| Torch | CUDA | NA | NA | torch not installed |")
    else:
        import torch

        med, values = time_call(
            lambda: torch_bootstrap_kernel(x, y, args.B, best_batch, args.seed + 202, "cpu"),
            args.repeat,
        )
        lines.append(f"| Torch | CPU | {med:.3f} | {format_seconds(values)} | includes tensor conversion |")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            med, values = time_call(
                lambda: torch_bootstrap_kernel(x, y, args.B, best_batch, args.seed + 203, "cuda"),
                args.repeat,
            )
            lines.append(f"| Torch | CUDA | {med:.3f} | {format_seconds(values)} | {device_name} |")
        else:
            lines.append("| Torch | CUDA | NA | NA | CUDA device unavailable in this environment |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The dominant p=200 bootstrap kernel multiplies Gaussian weights by "
        "`n x d` residual matrices with `d=20100`."
    )
    lines.append(
        "- Larger batch sizes reduce Python-loop overhead but increase memory use; "
        "for p=200, batch size 500--1000 is usually still modest on a typical desktop."
    )
    lines.append(
        "- GPU acceleration is only worth pursuing if a CUDA backend is available and "
        "the bootstrap matrices remain on device across many draws.  Otherwise, "
        "CPU BLAS plus repetition-level parallelism is usually safer."
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
