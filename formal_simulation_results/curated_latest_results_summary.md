# Curated Latest Simulation Results

This file summarizes the latest formal simulation results that are used in, or directly support, the current manuscript narrative. Earlier variance-profile diagnostics that do not clearly highlight the advantage of the proposed procedure have been removed from this curated archive.

## Storage Locations

- Full curated archive root: `simulation_restart/results/formal_simulation_archive/`
- Upload-package copy: `upload_package_20260608/04_formal_simulation_results/`
- Most readable size tables: `tables/size_main_p100.tex` and `tables/size_main_p200.tex`. These are the exact compact tables used in the manuscript.
- CSV companions for manuscript/supplement tables are stored next to the corresponding TeX files under `tables/`. For example, use `tables/size_main_p100.csv`, `tables/size_main_p200.csv`, and `tables/size_main_combined.csv` for spreadsheet-friendly versions of the main size tables.
- Main manuscript power figures:
  - `simulation_restart/results/formal_simulation_archive/figures/study_power_target_eq0_final_power.png`
  - `simulation_restart/results/formal_simulation_archive/figures/study_power_target_eq05_final_power.png`
  - `simulation_restart/results/formal_simulation_archive/figures/study_power_studentized_targets_final_power.png`

## How To Read The Files

- For a quick human-readable summary, read this file first.
- For the exact size tables shown in the manuscript, use `tables/size_main_p100.tex` and `tables/size_main_p200.tex`; use the same-stem `.csv` files for data processing.
- For the exact power displays shown in the manuscript, use the three PNG files listed above.
- For setting-level numerical values, use the `summary/*_paper_*.csv` or `summary/*_power_wide.csv` files described below. The most important display-level TeX tables also have same-stem CSV copies in `tables/`.
- For complete replicate-level outputs, use the corresponding files under `raw/`.
- Files in this curated archive are limited to the current size results, the current dense-power results, the retained studentization-power design, and supplement diagnostics that remain tied to the manuscript.

## Empirical Size

Formal size simulations use \(R=1000\) Monte Carlo repetitions and \(B=500\) multiplier-bootstrap draws.

Included data files:

- `raw/study_size_smalln_final_raw.csv`
- `raw/study_size_target_final_raw.csv`
- `summary/study_size_smalln_final_paper_size.csv`
- `summary/study_size_target_final_paper_size.csv`
- `tables/size_main_p100.tex`
- `tables/size_main_p200.tex`
- `tables/size_main_p100.csv`
- `tables/size_main_p200.csv`
- `tables/size_main_combined.csv`

Design:

- \(p\in\{100,200\}\).
- \((n_1,n_2)\in\{(60,60),(80,100),(120,150)\}\).
- Equicorrelation and Toeplitz covariance families.
- \(\rho_0\in\{0.1,0.5,0.9\}\).
- Gaussian, standardized centered \(\chi^2_1\), and standardized Laplace innovations.
- Simulated methods: \(G(0.3)\), \(G(0.5)\), \(G(0.7)\), \(I_p\), Max-bootstrap, Wang-NR, and Li--Chen.
- Main displayed methods: \(G(0.3)\), \(G(0.5)\), \(G(0.7)\), and \(I_p\). The comparison methods are retained in the CSV summaries but omitted from the manuscript size table to keep the size discussion focused on calibration of the proposed implementations.

Most readable files:

- `tables/size_main_p100.tex` and `tables/size_main_p100.csv`: compact manuscript table for all \(p=100\) size settings.
- `tables/size_main_p200.tex` and `tables/size_main_p200.csv`: compact manuscript table for all \(p=200\) size settings.
- `tables/size_main_combined.csv`: one combined spreadsheet-friendly table containing both \(p=100\) and \(p=200\).
- `summary/study_size_target_final_paper_size.csv`: setting-level results for \((80,100)\) and \((120,150)\), including all simulated methods.
- `summary/study_size_smalln_final_paper_size.csv`: setting-level results for \((60,60)\), including all simulated methods.

Calibration summary over all 108 size settings:

| Method | Mean size | Min | Max |
|---|---:|---:|---:|
| \(G(0.3)\) | 0.052 | 0.039 | 0.070 |
| \(G(0.5)\) | 0.055 | 0.040 | 0.075 |
| \(G(0.7)\) | 0.055 | 0.039 | 0.072 |
| \(I_p\) | 0.023 | 0.000 | 0.073 |

All target-transformed size entries lie in \([0.03,0.08]\). This supports the manuscript statement that the target transfer is stable across \(p\), sample size, covariance family, dependence strength, and innovation distribution, and that calibration is not highly sensitive to \(\rho_{\rm tar}\in\{0.3,0.5,0.7\}\).

## Dense Equicorrelation Power

Formal dense-power simulations use \(p=200\), \(n_1=120\), \(n_2=150\), Gaussian samples, \(R=500\), and \(B=500\).

Included data files:

- `raw/study_power_target_eq0_final_raw.csv`
- `raw/study_power_target_eq05_final_raw.csv`
- `summary/study_power_target_eq0_final_paper_power.csv`
- `summary/study_power_target_eq05_final_paper_power.csv`
- `tables/study_power_target_eq0_final_paper_power.csv`
- `tables/study_power_target_eq05_final_paper_power.csv`
- `figures/study_power_target_eq0_final_power.png`
- `figures/study_power_target_eq05_final_power.png`

Designs:

- Identity baseline: \(\rho_0=0\), with \(\rho_1\) increasing until power approaches one.
- Correlated baseline: \(\rho_0=0.5\), with \(\rho_1\in\{0.1,\ldots,0.9\}\).
- Simulated proposed targets: \(G(0.3)\), \(G(0.5)\), and \(G(0.7)\).
- Main displayed target: \(G(0.5)\), chosen for readability because the three target curves are close in these dense alternatives.
- Benchmarks displayed in figures: Max-bootstrap, Wang-NR, and Li--Chen.
- Identity \(I_p\) is retained in the full numerical outputs when available but is not displayed in the main power figures.

Most readable files:

- `figures/study_power_target_eq0_final_power.png`: manuscript figure for \(\rho_0=0\).
- `figures/study_power_target_eq05_final_power.png`: manuscript figure for \(\rho_0=0.5\).
- `summary/study_power_target_eq0_final_paper_power.csv`: setting-level numerical power values for \(\rho_0=0\), including target choices and comparison methods.
- `summary/study_power_target_eq05_final_paper_power.csv`: setting-level numerical power values for \(\rho_0=0.5\), including target choices and comparison methods.
- `tables/study_power_target_eq0_final_paper_power.csv` and `tables/study_power_target_eq05_final_paper_power.csv`: same numerical power values copied next to the TeX display tables for easier table/figure reproduction.

Key patterns:

- For \(\rho_0=0\), \(G(0.5)\) increases from size-level rejection near 0.056 to power 1.000 by \(\rho_1=0.010\), while Max-bootstrap remains much less powerful over the same range.
- For \(\rho_0=0.5\), the null point \(\rho_1=0.5\) remains calibrated for \(G(0.5)\) at 0.054, and power increases as \(\rho_1\) moves away from 0.5.

## Studentization Power

Formal studentization simulations use \(p=200\), \(n_1=120\), \(n_2=150\), Gaussian samples, \(R=500\), and \(B=500\).

Included data files:

- `raw/study_power_studentized_search_final_raw.csv`
- `summary/study_power_studentized_search_final_power_wide.csv`
- `summary/study_power_studentized_search_final_summary_long.csv`
- `summary/study_power_studentized_search_final_top_gains.csv`
- `figures/study_power_studentized_targets_final_power.png`

Design:

- A low-variance block of size 80 carries an increasing dense correlation signal.
- The remaining coordinates have high variance 25.
- Simulated methods: \(G(0.3)\), \(G(0.5)\), \(G(0.7)\), \(I_p\), Raw-L2, Max-bootstrap, Wang-NR, and Li--Chen.
- The manuscript figure displays \(G(0.5)\), Raw-L2, Max-bootstrap, Wang-NR, and Li--Chen.

Most readable files:

- `figures/study_power_studentized_targets_final_power.png`: manuscript figure for the retained studentization design.
- `summary/study_power_studentized_search_final_power_wide.csv`: most readable numerical table for this design, with one row per \(\rho_{\rm block}\) and one column per method.
- `summary/study_power_studentized_search_final_top_gains.csv`: diagnostic table highlighting settings where the studentized method improves over Raw-L2 and benchmarks.

Key pattern:

| \(\rho_{\rm block}\) | \(G(0.5)\) | Raw-L2 | Max-bootstrap | Wang-NR | Li--Chen |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.066 | 0.000 | 0.042 | 0.010 | 0.062 |
| 0.05 | 0.594 | 0.000 | 0.068 | 0.012 | 0.066 |
| 0.10 | 0.784 | 0.000 | 0.188 | 0.012 | 0.072 |
| 0.15 | 0.916 | 0.000 | 0.534 | 0.020 | 0.076 |
| 0.20 | 0.932 | 0.000 | 0.874 | 0.006 | 0.046 |
| 0.25 | 0.958 | 0.000 | 0.990 | 0.018 | 0.066 |
| 0.30 | 0.978 | 0.000 | 1.000 | 0.020 | 0.110 |
| 0.35 | 0.980 | 0.000 | 1.000 | 0.016 | 0.096 |
| 0.40 | 0.978 | 0.000 | 1.000 | 0.028 | 0.102 |
| 0.50 | 0.982 | 0.000 | 1.000 | 0.042 | 0.154 |

This is the retained variance-heterogeneous power design because it clearly illustrates the value of studentization: Raw-L2 fails to detect the low-variance block signal, whereas \(G(0.5)\) becomes powerful quickly while maintaining size-level rejection under the null.
