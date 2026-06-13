"""Collect formal simulation outputs into a single archive directory."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


FORMAL_STUDIES = (
    "study_size_target",
    "study_size_smalln",
    "study_size_unbalanced_ratio",
    "study_power_target_eq0",
    "study_power_target_eq05",
    "study_power_studentized_search",
    "study5",
)


STUDY_DESCRIPTIONS = {
    "study_size_target": (
        "Main size grid: p in {100,200}, n in {(80,100),(120,150)}, "
        "equicorrelation/Toeplitz, rho0 in {0.1,0.5,0.9}, Gaussian/"
        "centered chi-square(1)/Laplace innovations."
    ),
    "study_size_smalln": (
        "Small-sample size grid: p in {100,200}, n1=n2=60, "
        "equicorrelation/Toeplitz, rho0 in {0.1,0.5,0.9}, Gaussian/"
        "centered chi-square(1)/Laplace innovations."
    ),
    "study_size_unbalanced_ratio": (
        "Unbalanced ratio null diagnostic: p in {100,200}, n2=60, "
        "(n1,n2) in {(300,60),(180,60),(120,60),(90,60)}, "
        "equicorrelation/Toeplitz with rho0=0.5."
    ),
    "study_power_target_eq0": (
        "Power curve from rho0=0 to equicorrelation alternatives; "
        "p=200, n1=120, n2=150."
    ),
    "study_power_target_eq05": (
        "Power curve around rho0=0.5; p=200, n1=120, n2=150."
    ),
    "study_power_studentized_search": (
        "Retained variance-heterogeneous studentization power design with target "
        "methods, Raw-L2, and benchmark procedures."
    ),
    "study5": "Archived full versus simplified unbalanced statistic diagnostic.",
}

DISPLAY_ARTIFACTS = {
    "summary": (
        "size_main_combined_status.csv",
        "study_size_smalln_inclusion_diagnostic.csv",
    ),
    "tables": (
        "size_main_combined.tex",
        "size_main_p100.tex",
        "size_main_p200.tex",
        "size_unbalanced_ratio_supp.tex",
        "size_main_combined.csv",
        "size_main_p100.csv",
        "size_main_p200.csv",
        "size_unbalanced_ratio_supp.csv",
        "study_power_target_eq0_final_paper_power.csv",
        "study_power_target_eq05_final_paper_power.csv",
    ),
    "figures": (
        "study_power_target_eq0_final_power.png",
        "study_power_target_eq05_final_power.png",
        "study_power_studentized_targets_final_power.png",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package formal simulation outputs.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--archive-dir", default="results/formal_simulation_archive")
    parser.add_argument("--profile", default="final")
    return parser.parse_args()


def _copy_matching(src_dir: Path, dst_dir: Path, patterns: list[str]) -> list[str]:
    copied: list[str] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            if src.is_file():
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                copied.append(str(dst.relative_to(dst_dir.parent)))
    return copied


def package(results_dir: Path, archive_dir: Path, profile: str) -> Path:
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# Formal Simulation Archive",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source results directory: `{results_dir}`",
        "",
        "This archive contains formal simulation outputs only. Smoke/debug runs, "
        "benchmark timing files, and real-data application outputs are intentionally omitted.",
        "",
        "## Formal Studies",
        "",
    ]

    for study in FORMAL_STUDIES:
        patterns = [f"{study}_{profile}_*", f"{study}_{profile}.*"]
        copied = []
        for subdir in ("raw", "summary", "tables", "figures"):
            copied.extend(_copy_matching(results_dir / subdir, archive_dir / subdir, patterns))
        manifest_lines.extend(
            [
                f"### {study}",
                "",
                STUDY_DESCRIPTIONS.get(study, "Formal simulation study."),
                "",
                "Used in main text: "
                + (
                    "yes"
                    if study
                    in {
                        "study_size_target",
                        "study_size_smalln",
                        "study_power_target_eq0",
                        "study_power_target_eq05",
                        "study_power_studentized_search",
                    }
                    else "archive/supplement only"
                ),
                "",
            ]
        )
        if copied:
            manifest_lines.append("Files:")
            manifest_lines.extend(f"- `{item}`" for item in copied)
        else:
            manifest_lines.append("Files: none found at packaging time.")
        manifest_lines.append("")

    manifest_lines.extend(["## Manuscript Display Artifacts", ""])
    display_copied: list[str] = []
    for subdir, names in DISPLAY_ARTIFACTS.items():
        dst_dir = archive_dir / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = results_dir / subdir / name
            if src.exists():
                dst = dst_dir / name
                shutil.copy2(src, dst)
                display_copied.append(str(dst.relative_to(archive_dir)))
    if display_copied:
        manifest_lines.append("Files:")
        manifest_lines.extend(f"- `{item}`" for item in display_copied)
    else:
        manifest_lines.append("Files: none found at packaging time.")
    manifest_lines.append("")

    manifest_lines.extend(
        [
            "## Reproducible Commands",
            "",
            "```powershell",
            "python run_simulations.py --study study_size_target --profile final --R 1000 --B 500 --n-jobs 6",
            "python run_simulations.py --study study_size_smalln --profile final --R 1000 --B 500 --n-jobs 6",
            "python run_simulations.py --study study_size_unbalanced_ratio --profile final --R 1000 --B 500 --n-jobs 6",
            "python run_simulations.py --study study_power_target_eq0 --profile final --R 500 --B 500 --n-jobs 6",
            "python run_simulations.py --study study_power_target_eq05 --profile final --R 500 --B 500 --n-jobs 6",
            (
                "python scripts/explore_studentization_power.py --profile final --R 500 --B 500 "
                "--n-jobs 6 --include-target --target-rhos 0.3,0.5,0.7 --block-sizes 80 "
                "--high-variances 25 --rho-grid 0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50"
            ),
            "python write_manuscript_section.py --results-dir results",
            "python scripts/package_formal_simulations.py --results-dir results",
            "```",
            "",
            "All size and power settings retain full method outputs in the raw and summary files, "
            "even when the main manuscript displays only a subset of methods.",
            "",
        ]
    )
    manifest = archive_dir / "manifest.md"
    manifest.write_text("\n".join(manifest_lines), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = package(Path(args.results_dir), Path(args.archive_dir), args.profile)
    print(manifest)


if __name__ == "__main__":
    main()
