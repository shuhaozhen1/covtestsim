# Full Manuscript With Supplement

This directory contains the current full manuscript and supplement.

## Contents

- `template_with_supplement.tex`: manuscript source.
- `template_with_supplement.pdf`: compiled manuscript.
- `references.bib` and `agsm.bst`: bibliography files.
- `figures/`: manuscript figures, including the R=1000 power figures.
- `tables/`: compact manuscript/supplement size tables.
- `simulation_summary_r1000/`: lightweight CSV summaries for the R=1000 power simulations used in the current figures.

Large replicate-level raw simulation outputs are intentionally excluded from git.
They can be regenerated from the simulation scripts or read from the external
formal simulation archive when available.

## Build

The manuscript can be compiled with Tectonic:

```powershell
tectonic template_with_supplement.tex
```
