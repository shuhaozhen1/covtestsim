# Formal Simulation Archive

Generated: 2026-06-09T00:17:59
Source results directory: `results`

This archive contains formal simulation outputs only. Smoke/debug runs, benchmark timing files, and real-data application outputs are intentionally omitted.

## Formal Studies

### study_size_target

Main size grid: p in {100,200}, n in {(80,100),(120,150)}, equicorrelation/Toeplitz, rho0 in {0.1,0.5,0.9}, Gaussian/centered chi-square(1)/Laplace innovations.

Used in main text: yes

Files:
- `raw\study_size_target_final_raw.csv`
- `summary\study_size_target_final_paper_size.csv`
- `summary\study_size_target_final_paper_size_wide.csv`
- `summary\study_size_target_final_size_tables.md`
- `summary\study_size_target_final_summary_long.csv`
- `summary\study_size_target_final_summary_wide.csv`
- `tables\study_size_target_final_paper_size.tex`
- `tables\study_size_target_final_paper_size_wide.tex`
- `tables\study_size_target_final_summary_long.tex`
- `tables\study_size_target_final_summary_wide.tex`

### study_size_smalln

Small-sample size grid: p in {100,200}, n1=n2=60, equicorrelation/Toeplitz, rho0 in {0.1,0.5,0.9}, Gaussian/centered chi-square(1)/Laplace innovations.

Used in main text: yes

Files:
- `raw\study_size_smalln_final_raw.csv`
- `summary\study_size_smalln_final_paper_size.csv`
- `summary\study_size_smalln_final_paper_size_wide.csv`
- `summary\study_size_smalln_final_summary_long.csv`
- `summary\study_size_smalln_final_summary_wide.csv`
- `tables\study_size_smalln_final_paper_size.tex`
- `tables\study_size_smalln_final_paper_size_wide.tex`
- `tables\study_size_smalln_final_summary_long.tex`
- `tables\study_size_smalln_final_summary_wide.tex`

### study_size_unbalanced_ratio

Unbalanced ratio null diagnostic: p in {100,200}, n2=60, (n1,n2) in {(300,60),(180,60),(120,60),(90,60)}, equicorrelation/Toeplitz with rho0=0.5.

Used in main text: archive/supplement only

Files:
- `raw\study_size_unbalanced_ratio_final_raw.csv`
- `summary\study_size_unbalanced_ratio_final_paper_size.csv`
- `summary\study_size_unbalanced_ratio_final_paper_size_wide.csv`
- `summary\study_size_unbalanced_ratio_final_summary_long.csv`
- `summary\study_size_unbalanced_ratio_final_summary_wide.csv`
- `tables\study_size_unbalanced_ratio_final_paper_size.tex`
- `tables\study_size_unbalanced_ratio_final_paper_size_wide.tex`
- `tables\study_size_unbalanced_ratio_final_summary_long.tex`
- `tables\study_size_unbalanced_ratio_final_summary_wide.tex`

### study_power_target_eq0

Power curve from rho0=0 to equicorrelation alternatives; p=200, n1=120, n2=150.

Used in main text: yes

Files:
- `raw\study_power_target_eq0_final_raw.csv`
- `summary\study_power_target_eq0_final_paper_power.csv`
- `summary\study_power_target_eq0_final_power_table.md`
- `summary\study_power_target_eq0_final_summary_long.csv`
- `summary\study_power_target_eq0_final_summary_wide.csv`
- `tables\study_power_target_eq0_final_paper_power.tex`
- `tables\study_power_target_eq0_final_summary_long.tex`
- `tables\study_power_target_eq0_final_summary_wide.tex`
- `figures\study_power_target_eq0_final_power.png`

### study_power_target_eq05

Power curve around rho0=0.5; p=200, n1=120, n2=150.

Used in main text: yes

Files:
- `raw\study_power_target_eq05_final_raw.csv`
- `summary\study_power_target_eq05_final_paper_power.csv`
- `summary\study_power_target_eq05_final_power_table.md`
- `summary\study_power_target_eq05_final_summary_long.csv`
- `summary\study_power_target_eq05_final_summary_wide.csv`
- `tables\study_power_target_eq05_final_paper_power.tex`
- `tables\study_power_target_eq05_final_summary_long.tex`
- `tables\study_power_target_eq05_final_summary_wide.tex`
- `figures\study_power_target_eq05_final_power.png`

### study_power_studentized_search

Retained variance-heterogeneous studentization power design with target methods, Raw-L2, and benchmark procedures.

Used in main text: yes

Files:
- `raw\study_power_studentized_search_final_raw.csv`
- `summary\study_power_studentized_search_final_power_wide.csv`
- `summary\study_power_studentized_search_final_summary_long.csv`
- `summary\study_power_studentized_search_final_top_gains.csv`

### study5

Archived full versus simplified unbalanced statistic diagnostic.

Used in main text: archive/supplement only

Files:
- `raw\study5_final_raw.csv`
- `summary\study5_final_closeness.csv`
- `summary\study5_final_paper_unbalanced.csv`
- `summary\study5_final_summary_long.csv`
- `summary\study5_final_summary_wide.csv`
- `tables\study5_final_closeness.tex`
- `tables\study5_final_paper_unbalanced.tex`
- `tables\study5_final_summary_long.tex`
- `tables\study5_final_summary_wide.tex`
- `tables\study5_final_unbalanced_compact.tex`
- `tables\study5_final_unbalanced_size_compact.tex`

## Manuscript Display Artifacts

Files:
- `summary\size_main_combined_status.csv`
- `summary\study_size_smalln_inclusion_diagnostic.csv`
- `tables\size_main_combined.tex`
- `tables\size_main_combined.csv`
- `tables\size_main_p100.tex`
- `tables\size_main_p100.csv`
- `tables\size_main_p200.tex`
- `tables\size_main_p200.csv`
- `tables\size_unbalanced_ratio_supp.tex`
- `tables\size_unbalanced_ratio_supp.csv`
- `tables\study_power_target_eq0_final_paper_power.csv`
- `tables\study_power_target_eq05_final_paper_power.csv`
- `figures\study_power_target_eq0_final_power.png`
- `figures\study_power_target_eq05_final_power.png`
- `figures\study_power_studentized_targets_final_power.png`

All important manuscript and supplement TeX tables under `tables\` now have readable same-stem CSV companions in the same directory when a tabular data representation is meaningful.

## Reproducible Commands

```powershell
python run_simulations.py --study study_size_target --profile final --R 1000 --B 500 --n-jobs 6
python run_simulations.py --study study_size_smalln --profile final --R 1000 --B 500 --n-jobs 6
python run_simulations.py --study study_size_unbalanced_ratio --profile final --R 1000 --B 500 --n-jobs 6
python run_simulations.py --study study_power_target_eq0 --profile final --R 500 --B 500 --n-jobs 6
python run_simulations.py --study study_power_target_eq05 --profile final --R 500 --B 500 --n-jobs 6
python scripts/explore_studentization_power.py --profile final --R 500 --B 500 --n-jobs 6 --include-target --target-rhos 0.3,0.5,0.7 --block-sizes 80 --high-variances 25 --rho-grid 0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50
python write_manuscript_section.py --results-dir results
python scripts/package_formal_simulations.py --results-dir results
```

All size and power settings retain full method outputs in the raw and summary files, even when the main manuscript displays only a subset of methods.
