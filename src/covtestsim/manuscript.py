"""Generate manuscript-ready simulation displays and the section include file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "Ours-C": r"$A=G$",
    "Ours-I": r"$A=I_p$",
    "Max-bootstrap": "Max",
    "Wang-NR": "Wang-NR",
    "Li-Chen": "Li--Chen",
    "Raw-L2": r"raw $\ell_2$",
}
METHOD_ORDER = ["Ours-C", "Ours-I", "Max-bootstrap", "Wang-NR", "Li-Chen"]
PLOT_COLORS = {
    "Ours-C": "#0072B2",
    "Ours-I": "#D55E00",
    "Max-bootstrap": "#009E73",
    "Wang-NR": "#CC79A7",
    "Li-Chen": "#4D4D4D",
    "Raw-L2": "#666666",
}
PLOT_MARKERS = {
    "Ours-C": "o",
    "Ours-I": "s",
    "Max-bootstrap": "^",
    "Wang-NR": "D",
    "Li-Chen": "v",
    "Raw-L2": "o",
}


def _write_latex_table(df: pd.DataFrame, path: Path, float_format: str = "%.3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_latex(path, index=False, float_format=float_format, escape=False)


def _write_gaussian_size_table(summary_dir: Path, table_dir: Path) -> None:
    path = summary_dir / "study1_final_paper_size_wide.csv"
    if not path.exists():
        return
    raw = pd.read_csv(path)
    table = raw.rename(
        columns={
            "covariance family": "covariance",
            "rho": r"$\rho$",
            **{method: METHOD_LABELS[method] for method in METHOD_ORDER if method in raw.columns},
        }
    )
    cols = ["covariance", r"$\rho$"] + [METHOD_LABELS[m] for m in METHOD_ORDER if m in raw.columns]
    _write_latex_table(table[cols], table_dir / "study1_final_size_main_compact.tex")


def _fmt_size(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _fmt_rho(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.1f}"


def _family_label(value: object) -> str:
    mapping = {
        "equicorrelation": "Equicorr.",
        "toeplitz": "Toeplitz",
    }
    return mapping.get(str(value), str(value))


def _innovation_label(value: object) -> str:
    mapping = {
        "gaussian": "Gaussian",
        "chisq1": r"Centered $\chi^2_1$",
        "t5": r"$t_5$",
    }
    return mapping.get(str(value), str(value))


def _write_combined_size_table(summary_dir: Path, table_dir: Path) -> None:
    base_path = summary_dir / "study1_final_paper_size_wide.csv"
    hd_path = summary_dir / "study1_hd_final_paper_size_wide.csv"
    if not base_path.exists() or not hd_path.exists():
        return

    base = pd.read_csv(base_path)
    hd = pd.read_csv(hd_path)
    method_cols = [m for m in METHOD_ORDER if m in base.columns and m in hd.columns]

    lines = [
        r"\begin{tabular}{@{}llll" + "r" * len(method_cols) + r"@{}}",
        r"\toprule",
        "setting & covariance & innovation & " + r"$\rho$ & " + " & ".join(METHOD_LABELS[m] for m in method_cols) + r" \\",
        r"\midrule",
    ]
    base = base.copy()
    family_order = {"equicorrelation": 0, "toeplitz": 1}
    base["_family_order"] = base["covariance family"].map(family_order).fillna(99)
    base = base.sort_values(["_family_order", "rho"]).reset_index(drop=True)
    for idx, row in base.iterrows():
        values = " & ".join(_fmt_size(row[m]) for m in method_cols)
        setting_cell = r"\multirow{6}{*}{\(p=100\)}" if idx == 0 else ""
        if idx == 0:
            covariance_cell = r"\multirow{3}{*}{Equicorr.}"
        elif idx == 3:
            covariance_cell = r"\multirow{3}{*}{Toeplitz}"
        else:
            covariance_cell = ""
        innovation_cell = r"\multirow{6}{*}{Gaussian}" if idx == 0 else ""
        lines.append(
            setting_cell
            + " & "
            + covariance_cell
            + " & "
            + innovation_cell
            + " & "
            + _fmt_rho(row["rho"])
            + " & "
            + values
            + r" \\"
        )

    lines.extend([r"\addlinespace[2pt]", r"\midrule", r"\addlinespace[2pt]"])
    innovation_order = {"gaussian": 0, "chisq1": 1, "t5": 2}
    hd = hd.copy()
    hd["_innovation_order"] = hd["innovation"].map(innovation_order).fillna(99)
    hd["_family_order"] = hd["covariance family"].map(family_order).fillna(99)
    hd = hd.sort_values(["_innovation_order", "_family_order", "rho"]).reset_index(drop=True)
    for idx, row in hd.iterrows():
        values = " & ".join(_fmt_size(row[m]) for m in method_cols)
        setting_cell = r"\multirow{9}{*}{\(p=200\)}" if idx == 0 else ""
        covariance_cell = r"\multirow{9}{*}{Equicorr.}" if idx == 0 else ""
        if idx in (3, 6):
            lines.append(r"\addlinespace[1pt]")
        if idx in (0, 3, 6):
            innovation_cell = r"\multirow{3}{*}{" + _innovation_label(row["innovation"]) + "}"
        else:
            innovation_cell = ""
        lines.append(
            setting_cell
            + " & "
            + covariance_cell
            + " & "
            + innovation_cell
            + " & "
            + _fmt_rho(row["rho"])
            + " & "
            + values
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "study1_combined_size.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_study4_table(summary_dir: Path, table_dir: Path) -> None:
    path = summary_dir / "study4_final_paper_studentized_raw.csv"
    if not path.exists():
        return
    raw = pd.read_csv(path)
    table = raw.copy()
    table = table.pivot_table(
        index="variance multiplier",
        columns="statistic type",
        values="rejection rate",
        aggfunc="first",
    ).reset_index()
    table = table.rename(
        columns={
            "variance multiplier": "variance multiplier",
            "Ours-I": r"$A=I_p$",
            "Raw-L2": r"raw $\ell_2$",
        }
    )
    table = table[["variance multiplier", r"$A=I_p$", r"raw $\ell_2$"]]
    _write_latex_table(table, table_dir / "study4_final_studentized_raw_compact.tex")


def _write_study4_figure(summary_dir: Path, figure_dir: Path) -> None:
    path = summary_dir / "study4_final_paper_studentized_raw.csv"
    if not path.exists():
        return
    raw = pd.read_csv(path).sort_values("variance multiplier")
    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    labels = {"Ours-I": r"studentized $A=I_p$", "Raw-L2": r"raw $\ell_2$"}
    for method in ["Ours-I", "Raw-L2"]:
        block = raw[raw["statistic type"].eq(method)]
        ax.plot(
            block["variance multiplier"],
            block["rejection rate"],
            marker=PLOT_MARKERS[method],
            color=PLOT_COLORS[method],
            linewidth=1.8,
            markersize=4.5,
            label=labels[method],
        )
    ax.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("variance multiplier")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([1, 2, 5, 10, 15, 20])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "study4_final_studentized_raw_variance_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_study5_table(summary_dir: Path, table_dir: Path) -> None:
    path = summary_dir / "study5_final_paper_unbalanced.csv"
    if not path.exists():
        return
    raw = pd.read_csv(path).iloc[0]
    table = pd.DataFrame(
        [
            {"statistic": "full", "empirical size": raw["size_full"]},
            {
                "statistic": "simplified",
                "empirical size": raw["size_simp"],
            },
        ]
    )
    _write_latex_table(table, table_dir / "study5_final_unbalanced_size_compact.tex")


def _write_size_heatmap(summary_dir: Path, figure_dir: Path) -> None:
    path = summary_dir / "study1_dist_final_paper_size_wide.csv"
    if not path.exists():
        return
    raw = pd.read_csv(path)
    methods = [m for m in METHOD_ORDER if m in raw.columns]
    raw = raw.copy()
    raw["row"] = raw.apply(
        lambda r: f"{r['innovation']}, {r['covariance family']}, $\\rho={r['rho']:.1f}$",
        axis=1,
    )
    order_innov = {"gaussian": 0, "chisq1": 1, "t5": 2}
    order_family = {"equicorrelation": 0, "toeplitz": 1}
    raw["_innov"] = raw["innovation"].map(order_innov)
    raw["_family"] = raw["covariance family"].map(order_family)
    raw = raw.sort_values(["_innov", "_family", "rho"]).reset_index(drop=True)
    values = raw[methods].to_numpy(dtype=float)
    deviation = values - 0.05

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    vmax = 0.16
    im = ax.imshow(deviation, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(raw)))
    ax.set_yticklabels(raw["row"], fontsize=8)
    ax.set_title("Empirical size minus nominal level")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if abs(deviation[i, j]) > 0.095 else "black"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"size $-0.05$")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "study1_dist_size_heatmap.png", dpi=300)
    plt.close(fig)


def _write_study2_combined_figure(summary_dir: Path, figure_dir: Path) -> None:
    path = summary_dir / "study2_final_paper_power.csv"
    if not path.exists():
        return
    raw = pd.read_csv(path)
    methods = [m for m in METHOD_ORDER if m in set(raw["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.2), sharey=True)
    for ax, rho0 in zip(axes, [0.5, 0.9], strict=True):
        block = raw[np.isclose(raw["rho0"], rho0)]
        for method in methods:
            mb = block[block["method"] == method].sort_values("rho_alt")
            ax.plot(
                mb["rho_alt"],
                mb["rejection rate"],
                marker=PLOT_MARKERS[method],
                color=PLOT_COLORS[method],
                linewidth=1.8,
                markersize=4.5,
                label=METHOD_LABELS[method],
            )
        ax.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\rho_0={rho0}$")
        ax.set_xlabel(r"$\rho_{\mathrm{alt}}$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel("Rejection rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "study2_final_power_combined.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "study2_final_power_color.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_study3_figure(summary_dir: Path, figure_dir: Path) -> None:
    path = summary_dir / "study3_final_paper_power.csv"
    if not path.exists():
        return
    raw = pd.read_csv(path)
    methods = [m for m in METHOD_ORDER if m in set(raw["method"])]
    fig, ax = plt.subplots(figsize=(5.7, 3.4))
    for method in methods:
        mb = raw[raw["method"] == method].sort_values("rho_alt")
        ax.plot(
            mb["rho_alt"],
            mb["rejection rate"],
            marker=PLOT_MARKERS[method],
            color=PLOT_COLORS[method],
            linewidth=1.8,
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
    ax.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "study3_final_power_refined.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "study3_final_power_color.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_manuscript_displays(results_dir: Path) -> None:
    summary_dir = results_dir / "summary"
    table_dir = results_dir / "tables"
    figure_dir = results_dir / "figures"
    _write_gaussian_size_table(summary_dir, table_dir)
    _write_combined_size_table(summary_dir, table_dir)
    _write_study4_table(summary_dir, table_dir)
    _write_study4_figure(summary_dir, figure_dir)
    _write_study5_table(summary_dir, table_dir)
    _write_size_heatmap(summary_dir, figure_dir)
    _write_study2_combined_figure(summary_dir, figure_dir)
    _write_study3_figure(summary_dir, figure_dir)


def _table_block(caption: str, label: str, path: str, *, size: str = r"\small") -> str:
    return rf"""
\begin{{table}}[!htbp]
\caption{{{caption}}}
\label{{{label}}}
\centering
{size}
\input{{{path}}}
\end{{table}}
"""


def _figure_block(caption: str, label: str, path: str, *, width: str = r"0.82\textwidth") -> str:
    return rf"""
\begin{{center}}
\centering
\includegraphics[width={width}]{{{path}}}
\captionof{{figure}}{{{caption}}}
\label{{{label}}}
\end{{center}}
"""


def build_section(results_dir: Path) -> str:
    del results_dir
    table_dir = "simulation_restart/results/tables"
    figure_dir = "simulation_restart/results/figures"
    text = r"""
We report simulation studies designed to examine the finite-sample behavior of
the proposed covariance tests.  The main goals are to assess empirical size,
compare different numerical implementations of the transformation \(A\), and
evaluate power under dense covariance alternatives.  Throughout this section,
the label \(A=I_p\) denotes the proposed statistic applied in the original
coordinate system.

We also consider a target-transformed implementation, denoted by \(A=G\).  This
choice is motivated by the power analysis: a common nonsingular transformation
preserves the null hypothesis while changing the geometry in which covariance
differences are aggregated.  In applications without a scientifically specified
transformation, we use a dense equicorrelation target as a default numerical
reference.  Let
\[
\widehat\Sigma_g
=
\frac1{n_g}\sum_{i=1}^{n_g}
(X_{gi}-\bar X_g)(X_{gi}-\bar X_g)^\top,
\qquad g=1,2,
\]
and define the pooled covariance estimator
\[
\widehat\Sigma_{\rm pool}
=
\frac{n_1\widehat\Sigma_1+n_2\widehat\Sigma_2}{n_1+n_2}.
\]
For \(\rho_{\rm tar}=0.5\), set
\[
\Sigma_{\rm tar}
=
\Sigma_{\rm tar}(\rho_{\rm tar})
=
(1-\rho_{\rm tar})I_p+\rho_{\rm tar}{\bf 1}_p{\bf 1}_p^\top.
\]
The empirical transformation used in the simulations is
\[
G
=
\Sigma_{\rm tar}^{1/2}
\left(\widehat\Sigma_{\rm pool}+\lambda I_p\right)^{-1/2},
\qquad
\lambda
=
10^{-6}\frac{\operatorname{tr}(\widehat\Sigma_{\rm pool})}{p}.
\]
The ridge term is used only to compute the inverse square root stably.  Thus
\[
G\left(\widehat\Sigma_{\rm pool}+\lambda I_p\right)G^\top
=
\Sigma_{\rm tar},
\]
so \(G\) maps the estimated pooled covariance geometry to the target geometry.
Because \(G\) is data dependent, it should be interpreted as a plug-in numerical
implementation of the fixed-\(A\) theory rather than as an additional theorem-level
object; conditional on its realized value, the same nonsingular transformation
is applied to both samples.

The comparison also includes three external procedures: a maximum-coordinate
multiplier bootstrap test, the normal-reference method of
\citet{wang_two-sample_2024}, and the two-sample covariance test of
\citet{li_two_2012}.  All simulations use independent samples with sample
covariance matrices computed with denominator \(n\), matching the definition of
the proposed statistic.  Unless stated otherwise, the samples are Gaussian, the
nominal level is \(\alpha=0.05\), and multiplier-bootstrap methods use
\(B=2000\) bootstrap replications.  To keep the presentation focused, the main
text reports compact tables and standard power curves; the raw repetition-level
output and the full summary files are retained in the simulation archive.

\subsection{Empirical size}

We first study size under \(H_0:\Sigma_1=\Sigma_2\).  The baseline Gaussian
grid uses \(p=100\), \(n_1=n_2=120\), and the equicorrelation and Toeplitz
covariance families.  The high-dimensional size check uses \(p=200\),
\(n_1=n_2=150\), equicorrelation nulls with
\(\rho\in\{0.1,0.5,0.9\}\), and Gaussian, standardized centered \(\chi^2_1\),
and standardized \(t_5\) innovations.  The first block of
Table~\ref{tab:study1_size_restart} reports the \(p=100\) Gaussian grid with
\(R=2000\), while the second block reports the \(p=200\) high-dimensional grid
with \(R=1000\).
"""
    text += _table_block(
        r"Empirical size under baseline Gaussian nulls and high-dimensional equicorrelation nulls.  The nominal level is 0.05.",
        "tab:study1_size_restart",
        f"{table_dir}/study1_combined_size.tex",
        size=r"\scriptsize",
    )
    text += r"""

The transformed implementation with \(A=G\) is stable across the reported null
settings, while the identity implementation \(A=I_p\) is accurate in correlated
settings but conservative in weak-correlation cases.  The max-bootstrap
procedure provides a sparse-oriented reference and is often conservative under
non-Gaussian innovations.  The normal-reference procedure of
\citet{wang_two-sample_2024} is accurate in some correlated Gaussian settings
but conservative in weaker-dependence designs, whereas the Li--Chen test
\citep{li_two_2012} can be liberal under several non-Gaussian nulls.  These
patterns motivate using the bootstrap-calibrated studentized statistic as the
primary procedure in the subsequent power experiments.

\subsection{Power against dense covariance alternatives}

The main power experiments target dense covariance differences, the regime for
which the \(\ell_2\) aggregation in \(\widehat T_n\) is designed.  In the first
power experiment,
\[
\Sigma_1=\Sigma_{\mathrm{eq}}(\rho_0),
\qquad
\Sigma_2=\Sigma_{\mathrm{eq}}(\rho_{\mathrm{alt}}),
\]
with \(p=100\) and \(n_1=n_2=120\).  We use two grids:
\(\rho_0=0.5\) with
\(\rho_{\mathrm{alt}}\in\{0.5,0.6,0.7,0.8,0.9\}\), and
\(\rho_0=0.9\) with
\(\rho_{\mathrm{alt}}\in\{0.9,0.8,0.7,0.6,0.5\}\).  The equality case is the
null row.  Figure~\ref{fig:study2_power_restart} reports the rejection rates
from \(R=1000\) repetitions.
"""
    text += _figure_block(
        "Dense equicorrelation power curves.  Dashed horizontal lines mark the nominal level 0.05.",
        "fig:study2_power_restart",
        f"{figure_dir}/study2_final_power_combined.png",
        width=r"0.92\textwidth",
    )
    text += r"""

Figure~\ref{fig:study2_power_restart} should be read together with the null
rows and the size study above.  Among the procedures that maintain
approximately correct size in these Gaussian null settings, the transformed
implementation with \(A=G\) is the most powerful for the dense equicorrelation
alternatives.  The identity implementation is less sensitive in these settings,
and the max-type bootstrap is not designed for aggregated dense departures.
Although Li--Chen can show a high rejection curve in parts of the display, its
null rejection rates are inflated in several correlated settings; hence those
high rejection rates should not be interpreted as a size-controlled power
advantage.  The same caution applies more generally when comparing raw
rejection rates across methods with different size behavior.

The second power experiment considers a weak dense departure from independence:
\[
\Sigma_1=I_p,
\qquad
\Sigma_2=\Sigma_{\mathrm{eq}}(\rho),
\]
where \(\rho\in\{0,0.01,0.02,0.03,0.05,0.075,0.1\}\), again with \(p=100\) and
\(n_1=n_2=120\).  Figure~\ref{fig:study3_power_restart} displays the power
curves.
"""
    text += _figure_block(
        r"Power for identity-to-equicorrelation alternatives.  The point \(\rho=0\) is the null row.",
        "fig:study3_power_restart",
        f"{figure_dir}/study3_final_power_refined.png",
        width=r"0.72\textwidth",
    )
    text += r"""

This experiment shows most clearly how the target transformation changes
the geometry of the alternative.  At the null row, the implementation with
\(A=G\) is close to nominal, and for small positive values of \(\rho\) it is
already highly sensitive.  This behavior is consistent with the role of \(A\)
in the power analysis: a common nonsingular transformation preserves the null
hypothesis, but it can change the standardized transformed signal
\(\theta_n^{(A)}\).  Methods with size distortion in the null comparisons are
not treated as having a genuine power advantage merely because their rejection
curves are higher under alternatives.

\subsection{Diagnostic studies}

We first examine the effect of diagonal studentization in a deliberately simple
variance-spike design.  Study~4 compares the proposed statistic with
\(A=I_p\) against a raw non-studentized \(\ell_2\) bootstrap.  The dimension is
\(p=100\), \(n_1=n_2=120\), and the first group has covariance
\(\Sigma_{\rm eq}(0.5)\), so the first coordinate has variance one under the
baseline.  In the second group we apply the diagonal scaling
\(\operatorname{diag}(\sqrt m,1,\ldots,1)\), so only the first coordinate
variance is multiplied by
\[
m\in\{1,1.25,1.5,2,3,5,8,12,16,20\}.
\]
Thus \(m=1\) is the null size point, and \(m>1\) gives a one-coordinate
variance alternative whose strength increases to a nearly certain rejection
regime for the raw statistic.
"""
    text += _figure_block(
        r"Studentized and raw \(\ell_2\) bootstrap rejection rates for a one-coordinate variance alternative.  The point \(m=1\) is the null size point; \(m>1\) reports power.",
        "fig:study4_studentized_raw_restart",
        f"{figure_dir}/study4_final_studentized_raw_variance_curve.png",
        width=r"0.72\textwidth",
    )
    text += r"""

Figure~\ref{fig:study4_studentized_raw_restart} shows that the two
aggregations answer different scale questions.  Both procedures are calibrated
at \(m=1\).  The raw quadratic statistic is highly sensitive to a large
absolute variance spike and reaches power near one as \(m\) grows.  The
studentized statistic remains close to its null rejection rate across this
pure scale path, reflecting the fact that diagonal standardization normalizes
covariance-coordinate magnitudes before the \(\ell_2\) aggregation.  We
therefore interpret studentization as a scale-normalized dense-covariance
default rather than as a uniformly more powerful choice for sparse variance
alternatives.

Finally, we check the simplified statistic for highly unbalanced sample sizes.
The design follows Section~\ref{subsec:unbalanced_effect}: \(p=50\),
\(n_1=5000\), \(n_2=100\), and
\(\Sigma_1=\Sigma_2=\Sigma_{\mathrm{toep}}(0.8)\).  We compare the full
two-sample statistic \(\widehat T_n\) with the simplified statistic
\(\widehat T_{2|1}\).
"""
    text += _table_block(
        r"Full and simplified statistics in the unbalanced null design.  Here \(n_1/n_2=50\) and \(R=1000\).",
        "tab:study5_unbalanced_restart",
        f"{table_dir}/study5_final_unbalanced_size_compact.tex",
    )
    text += r"""

The full statistic has empirical size 0.049 and the simplified statistic has
empirical size 0.056.  The omitted closeness diagnostics are also small: the
median absolute statistic difference is 0.025 and the median absolute
\(p\)-value difference is 0.0295.  These results are consistent with the
unbalanced-sample approximation: when \(n_1/n_2\) is large, the larger sample
acts mainly as a covariance reference and the smaller sample governs the
dominant stochastic fluctuation.

Overall, the simulations support three conclusions.  First, the proposed
studentized multiplier-bootstrap statistic with \(A=I_p\) is calibrated in
several correlated covariance regimes, although it can be conservative under
weak dependence.  Second, the implementation with \(A=G\) gives
stable empirical size in the reported null studies and improves power for the
dense alternatives considered here.  Third, the external benchmarks exhibit
complementary behavior: the max-type bootstrap is useful as a sparse-oriented
reference but is less sensitive to weak dense alternatives, whereas Li--Chen
and Wang-NR can be powerful but show size distortion or conservativeness in
some null settings.

\clearpage
"""
    return text


def write_section(results_dir: Path) -> Path:
    _write_manuscript_displays(results_dir)
    output = results_dir / "tables" / "simulation_section.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_section(results_dir), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Write simulation_section.tex include file.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    path = write_section(Path(args.results_dir))
    print(path)


if __name__ == "__main__":
    main()
