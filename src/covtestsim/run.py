"""Command-line runner for the clean simulation restart."""

from __future__ import annotations

import argparse
from pathlib import Path

from .studies import run_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run covariance-test simulation studies.")
    parser.add_argument(
        "--study",
        choices=[
            "study1",
            "study1_dist",
            "study1_hd",
            "study2",
            "study3",
            "study4",
            "study5",
            "study6",
            "realdata_sim",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--profile", choices=["smoke", "debug", "final"], default="smoke")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--include-supplement", action="store_true")
    parser.add_argument("--R", type=int, default=None, help="Override Monte Carlo repetitions.")
    parser.add_argument("--B", type=int, default=None, help="Override bootstrap repetitions.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--rep-start",
        type=int,
        default=1,
        help="First Monte Carlo repetition to run; useful for resumable final runs.",
    )
    parser.add_argument(
        "--rep-end",
        type=int,
        default=None,
        help="Last Monte Carlo repetition to run. Defaults to --R when omitted.",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append the requested repetition range to the existing raw file before rewriting summaries.",
    )
    parser.add_argument("--cache-dir", default=None, help="Cache directory for real-data-based simulations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    studies = ["study1", "study2", "study3", "study4", "study5"] if args.study == "all" else [args.study]
    if args.study == "all" and args.include_supplement:
        studies.append("study6")
    for study in studies:
        print(f"Running {study} profile={args.profile}")
        run_study(
            study,
            profile=args.profile,
            out_dir=out_dir,
            include_supplement=args.include_supplement,
            R=args.R,
            B=args.B,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            rep_start=args.rep_start,
            rep_end=args.rep_end,
            append_existing=args.append_existing,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
