# Clean Simulation Restart

This folder contains a standalone Python implementation of the restarted
simulation framework for the covariance-testing paper. It is intentionally
separate from the JASA template files.

## Layout

- `src/covtestsim/`: package source code.
- `tests/`: unit and smoke tests.
- `results/`: generated simulation and real-data outputs. This directory is
  created by the runners and is not tracked in the code repository.
- `data_cache/`: downloaded TCGA-BRCA and MSigDB files used by the real-data
  application. This directory is created on demand and is not tracked in the
  code repository.

## Implemented Methods

- `Ours-I`: studentized L2 multiplier-bootstrap covariance test with `C = I_p`.
- `Ours-C`: same statistic after adaptive target transformation
  `C_hat = Sigma_tar^{1/2}(Sigma_pool_hat + lambda I)^{-1/2}`.
  The default target is equicorrelation with `rho_tar = 0.5`; the ridge is
  `1e-6 * mean(diag(Sigma_pool_hat))`.
- `Max-bootstrap`: max over standardized vech covariance-coordinate differences,
  calibrated by the same multiplier-bootstrap residuals as `Ours-I`.
- `Wang-NR`: Wang normal-reference / three-cumulant matched chi-square method.
  Induced covariance-vector traces are computed from Gram matrices rather than
  forming the `p^2 x p^2` induced covariance matrix.
- `Li-Chen`: exact whole covariance-matrix U-statistic from Li and Chen (2012),
  equations (2.1)-(2.3), with the page-7 variance denominator
  `2 A_n1 / n2 + 2 A_n2 / n1`.
- `Raw-L2`: non-studentized L2 bootstrap, used only in Study 4.
- `Ours-I-simplified`: one-sample simplified bootstrap, used only in Study 5.

Internal L2-3C and cutoff-switching variants are not included in the main
study outputs.

## Seed Policy

The base seed is `20260521`. Data and bootstrap seeds are derived by stable
BLAKE2 hashing of the study, scenario, repetition, and method labels, so runs
are reproducible across Python processes.

All methods in a repetition receive the same generated datasets. Bootstrap
methods receive method-specific bootstrap seeds.

## Running

From this folder:

```powershell
python run_simulations.py --study all --profile smoke
```

Profiles:

- `smoke`: `R=3`, `B=10`; intended for path and regression checks.
- `debug`: `R=50`, `B=500`; intended for exploratory diagnostics.
- `final`: uses study defaults with `B=2000`; this can be computationally heavy.

Examples:

```powershell
python run_simulations.py --study study1 --profile final --n-jobs 4
python run_simulations.py --study study1_dist --profile final --R 1000 --n-jobs 4
python run_simulations.py --study study1_hd --profile final --R 1000 --B 2000 --n-jobs 8
python run_simulations.py --study study2 --profile debug --include-supplement
python run_simulations.py --study study3 --profile smoke --R 5 --B 20
```

To extend a partially completed final run without rerunning earlier Monte Carlo
replications, use the repetition-range options.  For example, after reps
1--500 of `study1_hd` have been saved, the following command runs reps
501--1000 and rewrites the combined summaries:

```powershell
python run_simulations.py --study study1_hd --profile final --R 1000 --B 2000 --n-jobs 8 --rep-start 501 --append-existing
```

Real-data application:

```powershell
python run_realdata.py --profile debug --B 20
python run_realdata.py --profile final --B 2000
```

The real-data pipeline downloads TCGA-BRCA expression and clinical annotations
from UCSC Xena, downloads MSigDB Hallmark gene sets, constructs four
Hallmark-derived biological program unions, and compares `Ours-I`, `Ours-C`,
`Max-bootstrap`, `Wang-NR`, and `Li-Chen`. Programs use complete Hallmark
components or clearly named Hallmark unions; no top-variance truncation is used
for the reported application. Eligible panels must satisfy `p >= min(n1, n2)`
and `p >= 100`.

Study 6 is optional and is included in `--study all` only when
`--include-supplement` is supplied; it can also be run directly with
`--study study6`.

## Real-Data Application

The real-data pipeline supports screening predefined Hallmark panels across
UCSC Xena TCGA candidates and finalizing one selected application. The current
manuscript application compares TCGA-BRCA Basal-like and Luminal A primary
tumors using PAM50 RNA-seq subtype labels from UCSC Xena. The selected panel is
the complete Hallmark cell-cycle core union, E2F targets plus G2M checkpoint.

Useful commands:

```powershell
python run_realdata.py --mode screen --profile debug --B 20 --cohorts BRCA
python run_realdata.py --mode final-selected --profile final --B 2000 --selected-candidate BRCA_subtype_cell_cycle_core
```

Outputs are saved as:

- `results/raw/realdata_final_raw.csv`
- `results/summary/realdata_final_method_results.csv`
- `results/summary/realdata_final_programs.csv`
- `results/tables/realdata_final_program_method_pvalues.tex`
- `results/tables/realdata_final_selected_program.tex`
- `results/figures/realdata_final_component_correlation_heatmap.png`

## Studies

- Study 1: null size certification for equicorrelation and Toeplitz nulls.
- Study 1_dist: null size certification stratified by innovation distribution.
  It uses the Study 1 covariance grid with Gaussian, standardized chi-square(1),
  and standardized t(5) innovations.
- Study 1_hd: high-dimensional null size certification with `p=200`,
  `n1=n2=150`, equicorrelation nulls at `rho in {0.1, 0.5, 0.9}`, and
  Gaussian, standardized chi-square(1), and standardized t(5) innovations.
- Study 2: dense equicorrelation power for `rho0 = 0.5` and `rho0 = 0.9`;
  the near-independent grid is controlled by `--include-supplement`.
- Study 3: identity-to-equicorrelation dense alternatives.
- Study 4: studentized versus raw L2 aggregation for a one-coordinate variance
  spike with variance multiplier ranging from 1 to 20.
- Study 5: full versus simplified Ours-I in an unbalanced null design
  (`p=50`, `n1=5000`, `n2=100`, Toeplitz `rho=0.8`).
- Study 6: optional non-Gaussian null robustness.

## Tests

```powershell
python -m unittest discover -s tests
```

The tests cover vech ordering, covariance positive definiteness, covariance
denominator convention, target-transform stability, Wang Gram-trace identities
against explicit induced matrices, Li-Chen inclusion-exclusion against brute
force, valid bootstrap p-values, real-data helper functions, and small smoke
runs.

## Computational Notes

Final runs use `B=2000` for bootstrap methods and `R=1000` for the main size
and power studies. The heaviest pieces are repeated matrix square roots for
`Ours-C`, bootstrap matrix multiplications for `p=100` or `p=200`, and exact
Li-Chen U-statistic components. Use `--n-jobs` to parallelize repetitions across
CPU cores when running final tables.
