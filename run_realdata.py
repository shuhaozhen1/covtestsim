from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from covtestsim.realdata import rank_candidates, run_realdata_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or screen real-data covariance applications.")
    parser.add_argument("--mode", choices=["screen", "final-selected"], default="screen")
    parser.add_argument("--profile", choices=["debug", "final"], default="debug")
    parser.add_argument("--B", type=int, default=None, help="Bootstrap draws for bootstrap methods.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cache-dir", default=str(ROOT / "data_cache"))
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--max-p", type=int, default=500, help="Maximum biologically predefined panel size to run.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--selected-candidate", default=None)
    parser.add_argument("--cohorts", nargs="*", default=None, help="Optional cohort filter for screening, e.g. BRCA LUAD.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Development-only cap for smoke tests.")
    args = parser.parse_args()

    raw = run_realdata_pipeline(
        profile=args.profile,
        B=args.B,
        alpha=args.alpha,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        max_p=args.max_p,
        batch_size=args.batch_size,
        mode=args.mode,
        selected_candidate=args.selected_candidate,
        cohorts=args.cohorts,
        max_candidates=args.max_candidates,
    )
    print(raw[["candidate_id", "program", "method", "n1", "n2", "p", "statistic", "p_value", "reject"]].to_string(index=False))
    ranked = rank_candidates(raw)
    if not ranked.empty:
        print("\nRanked candidates:")
        cols = ["candidate_id", "cohort", "program", "n1", "n2", "p", "ours_i_p", "ours_c_p", "benchmark_rejects"]
        print(ranked[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
