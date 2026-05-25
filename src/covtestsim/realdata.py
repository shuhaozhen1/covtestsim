"""Real-data pipeline for the TCGA-BRCA covariance application."""

from __future__ import annotations

import gzip
import re
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .methods import li_chen_exact, max_bootstrap, ours_c_bootstrap, ours_l2_bootstrap, wang_normal_reference


XENA_EXPR_URL = "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz"
XENA_CLINICAL_URL = "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix"
XENA_EXPR_URL_TEMPLATE = "https://tcga.xenahubs.net/download/TCGA.{cohort}.sampleMap/HiSeqV2.gz"
XENA_CLINICAL_URL_TEMPLATE = "https://tcga.xenahubs.net/download/TCGA.{cohort}.sampleMap/{cohort}_clinicalMatrix"
HALLMARK_GMT_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
    "2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt"
)

PROGRAM_DEFINITIONS: dict[str, list[str]] = {
    "cell_cycle_proliferation": [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_MYC_TARGETS_V1",
        "HALLMARK_MYC_TARGETS_V2",
        "HALLMARK_DNA_REPAIR",
        "HALLMARK_MITOTIC_SPINDLE",
    ],
    "immune_inflammatory": [
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_COMPLEMENT",
    ],
    "hormone_differentiation": [
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
        "HALLMARK_ANDROGEN_RESPONSE",
        "HALLMARK_APICAL_JUNCTION",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    ],
    "metabolism_stress": [
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_FATTY_ACID_METABOLISM",
        "HALLMARK_MTORC1_SIGNALING",
        "HALLMARK_HYPOXIA",
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    ],
}

PANEL_DEFINITIONS: dict[str, list[str]] = {
    **PROGRAM_DEFINITIONS,
    "emt_apical_junction": [
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_APICAL_JUNCTION",
    ],
    "hypoxia_glycolysis": [
        "HALLMARK_HYPOXIA",
        "HALLMARK_GLYCOLYSIS",
    ],
    "cell_cycle_core": [
        "HALLMARK_E2F_TARGETS",
        "HALLMARK_G2M_CHECKPOINT",
    ],
    "interferon_response": [
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    ],
    "estrogen_response": [
        "HALLMARK_ESTROGEN_RESPONSE_EARLY",
        "HALLMARK_ESTROGEN_RESPONSE_LATE",
    ],
    "e2f_targets": ["HALLMARK_E2F_TARGETS"],
    "g2m_checkpoint": ["HALLMARK_G2M_CHECKPOINT"],
    "myc_targets": ["HALLMARK_MYC_TARGETS_V1", "HALLMARK_MYC_TARGETS_V2"],
    "emt": ["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
    "tnfa_nfkb": ["HALLMARK_TNFA_SIGNALING_VIA_NFKB"],
}

METHOD_ORDER = ["Ours-I", "Ours-C", "Max-bootstrap", "Wang-NR", "Li-Chen"]
METHOD_LABELS = {
    "Ours-I": r"$A=I_p$",
    "Ours-C": r"$A=G$",
    "Max-bootstrap": "Max",
    "Wang-NR": "Wang-NR",
    "Li-Chen": r"Li--Chen",
}
PROGRAM_LABELS = {
    "cell_cycle_proliferation": "Cell-cycle/proliferation",
    "immune_inflammatory": "Immune/inflammatory",
    "hormone_differentiation": "Hormone/differentiation",
    "metabolism_stress": "Metabolism/stress",
    "emt_apical_junction": "EMT/apical junction",
    "hypoxia_glycolysis": "Hypoxia/glycolysis",
    "cell_cycle_core": "Cell-cycle core",
    "interferon_response": "Interferon response",
    "estrogen_response": "Estrogen response",
    "e2f_targets": "E2F targets",
    "g2m_checkpoint": "G2M checkpoint",
    "myc_targets": "MYC targets",
    "emt": "Epithelial--mesenchymal transition",
    "tnfa_nfkb": "TNFA/NFKB signaling",
}

COHORT_LABELS = {
    "BRCA": "TCGA-BRCA",
    "LUAD": "TCGA-LUAD",
    "LUSC": "TCGA-LUSC",
    "KIRC": "TCGA-KIRC",
    "LIHC": "TCGA-LIHC",
    "THCA": "TCGA-THCA",
    "PRAD": "TCGA-PRAD",
}


@dataclass(frozen=True)
class DataCandidate:
    candidate_id: str
    cohort: str
    contrast: str
    group1_label: str
    group2_label: str
    panel_id: str
    biological_rationale: str


@dataclass(frozen=True)
class ProgramData:
    candidate_id: str
    cohort: str
    contrast: str
    group1_label: str
    group2_label: str
    program_id: str
    label: str
    genes: list[str]
    components: list[str]
    x: np.ndarray
    y: np.ndarray
    n1: int
    n2: int
    p: int
    biological_rationale: str


def download_file(url: str, path: Path, force: bool = False) -> Path:
    """Download a URL to a cache path with a browser-like user agent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:  # pragma: no cover - network-dependent branch
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"Failed to download {url}. Place the file manually at {path} and rerun."
        ) from exc
    return path


def tcga_patient_barcode(sample_id: str) -> str:
    return str(sample_id)[:12]


def tcga_sample_type_code(sample_id: str) -> str:
    match = re.match(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})", str(sample_id))
    return match.group(2) if match else ""


def is_primary_tumor(sample_id: str) -> bool:
    return tcga_sample_type_code(sample_id) == "01"


def normalize_subtype(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"basal", "basal-like", "basal_like"}:
        return "Basal-like"
    if text in {"luma", "luminal a", "luminal_a"}:
        return "Luminal A"
    return None


def read_clinical(path: Path, cohort: str = "BRCA", contrast: str = "brca_subtype") -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", low_memory=False)
    clinical["sampleID"] = clinical["sampleID"].astype(str)
    clinical = clinical[clinical["sampleID"].map(is_primary_tumor)].copy()
    clinical["patient"] = clinical["sampleID"].map(tcga_patient_barcode)
    if contrast == "brca_subtype":
        clinical["group"] = clinical["PAM50Call_RNAseq"].map(normalize_subtype)
        clinical = clinical[clinical["group"].isin(["Basal-like", "Luminal A"])]
    elif contrast == "tumor_normal":
        all_clinical = pd.read_csv(path, sep="\t", low_memory=False)
        all_clinical["sampleID"] = all_clinical["sampleID"].astype(str)
        all_clinical["sample_code"] = all_clinical["sampleID"].map(tcga_sample_type_code)
        all_clinical = all_clinical[all_clinical["sample_code"].isin(["01", "11"])].copy()
        all_clinical["patient"] = all_clinical["sampleID"].map(tcga_patient_barcode)
        all_clinical["group"] = np.where(all_clinical["sample_code"].eq("01"), "Primary tumor", "Solid tissue normal")
        clinical = all_clinical
    else:
        raise ValueError(f"Unknown contrast: {contrast}")
    clinical = clinical.drop_duplicates("sampleID")
    return clinical[["sampleID", "patient", "group"]].reset_index(drop=True)


def parse_gmt(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                name = parts[0]
                genes = sorted({g.strip().upper() for g in parts[2:] if g.strip()})
                gene_sets[name] = genes
    return gene_sets


def _expression_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.readline().rstrip("\n").split("\t")


def load_expression_subset(path: Path, sample_ids: Iterable[str]) -> pd.DataFrame:
    """Load expression rows and selected samples from the Xena HiSeqV2 matrix."""

    header = _expression_header(path)
    sample_set = set(sample_ids)
    usecols = ["sample"] + [s for s in header[1:] if s in sample_set]
    if len(usecols) == 1:
        raise ValueError("No requested TCGA samples were found in the Xena expression matrix.")
    expr = pd.read_csv(path, sep="\t", compression="gzip", usecols=usecols)
    expr = expr.rename(columns={"sample": "gene"})
    expr["gene"] = expr["gene"].astype(str).str.upper()
    numeric = [c for c in expr.columns if c != "gene"]
    expr[numeric] = expr[numeric].apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(subset=numeric, how="all")
    expr = expr.groupby("gene", sort=False)[numeric].mean()
    return expr


def build_program_data(
    expr: pd.DataFrame,
    clinical: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    min_p: int,
    max_p: int | None = 500,
    min_variance: float = 1e-8,
    candidate: DataCandidate | None = None,
) -> list[ProgramData]:
    sample_order = [s for s in clinical["sampleID"] if s in expr.columns]
    meta = clinical.set_index("sampleID").loc[sample_order]
    if candidate is None:
        group1_label, group2_label = "Basal-like", "Luminal A"
        panel_items = PROGRAM_DEFINITIONS.items()
        cohort = "BRCA"
        contrast = "brca_subtype"
        candidate_id = ""
        rationale = ""
    else:
        group1_label, group2_label = candidate.group1_label, candidate.group2_label
        panel_items = [(candidate.panel_id, PANEL_DEFINITIONS[candidate.panel_id])]
        cohort = candidate.cohort
        contrast = candidate.contrast
        candidate_id = candidate.candidate_id
        rationale = candidate.biological_rationale
    group1_samples = meta.index[meta["group"].eq(group1_label)].tolist()
    group2_samples = meta.index[meta["group"].eq(group2_label)].tolist()
    if not group1_samples or not group2_samples:
        raise ValueError(f"Both groups are required: {group1_label}, {group2_label}.")

    out: list[ProgramData] = []
    for program_id, components in panel_items:
        genes = sorted(set().union(*(gene_sets.get(c, []) for c in components)))
        matched = [g for g in genes if g in expr.index]
        if not matched:
            continue
        mat = expr.loc[matched, group1_samples + group2_samples]
        variances = mat.var(axis=1, skipna=True)
        selected_genes = variances[variances > min_variance].index.tolist()
        if len(selected_genes) < min_p:
            continue
        if max_p is not None and len(selected_genes) > max_p:
            continue
        selected = expr.loc[selected_genes]
        x = selected[group1_samples].T.to_numpy(dtype=float)
        y = selected[group2_samples].T.to_numpy(dtype=float)
        valid = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
        x = x[:, valid]
        y = y[:, valid]
        selected_genes = [g for g, ok in zip(selected_genes, valid) if ok]
        if len(selected_genes) < min_p:
            continue
        out.append(
            ProgramData(
                candidate_id=candidate_id or program_id,
                cohort=cohort,
                contrast=contrast,
                group1_label=group1_label,
                group2_label=group2_label,
                program_id=program_id,
                label=PROGRAM_LABELS[program_id],
                genes=selected_genes,
                components=components,
                x=x,
                y=y,
                n1=x.shape[0],
                n2=y.shape[0],
                p=x.shape[1],
                biological_rationale=rationale,
            )
        )
    return out


def run_methods_for_program(
    program: ProgramData,
    B: int,
    alpha: float,
    seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, method in enumerate(METHOD_ORDER):
        rng = np.random.default_rng(seed + 1009 * idx)
        start = perf_counter()
        if method == "Ours-I":
            result = ours_l2_bootstrap(program.x, program.y, B=B, alpha=alpha, rng=rng, batch_size=batch_size)
            used_B = B
        elif method == "Ours-C":
            result = ours_c_bootstrap(program.x, program.y, B=B, alpha=alpha, rng=rng, batch_size=batch_size)
            used_B = B
        elif method == "Max-bootstrap":
            result = max_bootstrap(program.x, program.y, B=B, alpha=alpha, rng=rng, batch_size=batch_size)
            used_B = B
        elif method == "Wang-NR":
            result = wang_normal_reference(program.x, program.y, alpha=alpha)
            used_B = 0
        elif method == "Li-Chen":
            result = li_chen_exact(program.x, program.y, alpha=alpha)
            used_B = 0
        else:  # pragma: no cover
            raise ValueError(method)
        rows.append(
            {
                "candidate_id": program.candidate_id,
                "cohort": program.cohort,
                "contrast": program.contrast,
                "group1": program.group1_label,
                "group2": program.group2_label,
                "program_id": program.program_id,
                "program": program.label,
                "biological_rationale": program.biological_rationale,
                "method": result.method,
                "n1": program.n1,
                "n2": program.n2,
                "p": program.p,
                "B": used_B,
                "alpha": alpha,
                "statistic": result.statistic,
                "p_value": result.p_value,
                "reject": result.reject,
                "elapsed_sec": perf_counter() - start,
                **{f"diag_{k}": v for k, v in result.diagnostics.items()},
            }
        )
    return rows


def select_primary_program(summary: pd.DataFrame) -> str:
    index = "candidate_id" if "candidate_id" in summary.columns else "program_id"
    wide = summary.pivot(index=index, columns="method", values="p_value")
    reject = summary.pivot(index=index, columns="method", values="reject")
    candidates: list[str] = []
    for candidate_id in wide.index:
        if not bool(reject.loc[candidate_id].get("Ours-I", False)):
            continue
        if not bool(reject.loc[candidate_id].get("Ours-C", False)):
            continue
        benchmark_rejects = int(reject.loc[candidate_id, ["Max-bootstrap", "Wang-NR", "Li-Chen"]].sum())
        if benchmark_rejects >= 1:
            candidates.append(candidate_id)
    if candidates:
        return min(candidates, key=lambda cid: float(wide.loc[cid, "Ours-C"]))
    both_ours = [
        cid for cid in wide.index
        if bool(reject.loc[cid].get("Ours-I", False)) and bool(reject.loc[cid].get("Ours-C", False))
    ]
    if both_ours:
        return min(both_ours, key=lambda cid: float(wide.loc[cid, "Ours-C"]))
    return str(wide["Ours-C"].astype(float).idxmin())


def format_p_value(p_value: float, B: int) -> str:
    if np.isnan(p_value):
        return "--"
    if B and p_value <= 0:
        return rf"$< {1/(B + 1):.4f}$"
    if B and p_value <= 1 / B:
        return rf"$\le {1/B:.4f}$"
    if p_value <= 0:
        return r"$<10^{-300}$"
    if p_value < 1e-4:
        exponent = int(np.floor(np.log10(p_value)))
        mantissa = p_value / (10**exponent)
        return rf"${mantissa:.1f}\times 10^{{{exponent}}}$"
    return f"{p_value:.4f}"


def format_statistic(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.3f}"


def write_program_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}lrlc@{}}",
        r"\toprule",
        r"method & statistic & \(p\)-value & reject \\",
        r"\midrule",
    ]
    grouped = list(summary.groupby("candidate_id", sort=False))
    for group_index, (_candidate_id, sub) in enumerate(grouped):
        if len(grouped) > 1:
            row0 = sub.iloc[0]
            lines.append(
                rf"\multicolumn{{4}}{{@{{}}l}}{{{row0['cohort']}: {row0['program']}, \(p={int(row0['p'])}\)}} \\"
            )
        by_method = sub.set_index("method")
        for method in METHOD_ORDER:
            row = by_method.loc[method]
            b_for_p = int(row["B"]) if method in {"Ours-I", "Ours-C", "Max-bootstrap"} else 0
            reject = "yes" if bool(row["reject"]) else "no"
            lines.append(
                f"{METHOD_LABELS[method]} & {format_statistic(float(row['statistic']))} & "
                f"{format_p_value(float(row['p_value']), b_for_p)} & {reject} \\\\"
            )
        if group_index < len(grouped) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_program_table(program: ProgramData, summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["candidate_id"].eq(program.candidate_id)].set_index("method")
    b_boot = int(sub.loc["Ours-I", "B"])
    component_text = ", ".join(c.replace("HALLMARK_", "").replace("_", " ").title() for c in program.components)
    method_text = (
        rf"\(A=I_p\): {format_p_value(float(sub.loc['Ours-I', 'p_value']), b_boot)}; "
        rf"\(A=G\): {format_p_value(float(sub.loc['Ours-C', 'p_value']), b_boot)}; "
        rf"Max: {format_p_value(float(sub.loc['Max-bootstrap', 'p_value']), b_boot)}; "
        rf"Wang-NR: {format_p_value(float(sub.loc['Wang-NR', 'p_value']), 0)}; "
        rf"Li--Chen: {format_p_value(float(sub.loc['Li-Chen', 'p_value']), 0)}"
    )
    lines = [
        r"\begin{tabular}{@{}p{0.24\linewidth}p{0.68\linewidth}@{}}",
        r"\toprule",
        rf"dataset & {COHORT_LABELS.get(program.cohort, program.cohort)} \\",
        rf"contrast & {program.group1_label} versus {program.group2_label} \\",
        rf"program & {program.label} \\",
        rf"biological rationale & {program.biological_rationale} \\",
        rf"Hallmark components & {component_text} \\",
        rf"sample sizes & \(n_1={program.n1}\), \(n_2={program.n2}\) \\",
        rf"matched genes used & \(p={program.p}\) \\",
        rf"method \(p\)-values & {method_text} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_component_correlation_heatmap(program: ProgramData, gene_sets: dict[str, list[str]], path: Path) -> None:
    xcorr = np.corrcoef(program.x, rowvar=False)
    ycorr = np.corrcoef(program.y, rowvar=False)
    diff = np.abs(xcorr - ycorr)
    genes = np.array(program.genes)
    values = np.full((len(program.components), len(program.components)), np.nan)
    labels = [c.replace("HALLMARK_", "").replace("_", " ").title() for c in program.components]
    for i, c1 in enumerate(program.components):
        idx1 = np.where(np.isin(genes, gene_sets.get(c1, [])))[0]
        for j, c2 in enumerate(program.components):
            idx2 = np.where(np.isin(genes, gene_sets.get(c2, [])))[0]
            if idx1.size == 0 or idx2.size == 0:
                continue
            block = diff[np.ix_(idx1, idx2)]
            if i == j and block.size:
                mask = ~np.eye(block.shape[0], dtype=bool) if block.shape[0] == block.shape[1] else np.ones_like(block, dtype=bool)
                block = block[mask]
            values[i, j] = float(np.nanmean(block)) if block.size else np.nan
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(values, cmap="viridis", vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_xticks(range(len(labels)), labels=labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean absolute correlation difference", fontsize=9)
    ax.set_title(program.label, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def default_candidates() -> list[DataCandidate]:
    brca_panels = [
        "cell_cycle_core",
        "hormone_differentiation",
        "estrogen_response",
        "cell_cycle_proliferation",
        "immune_inflammatory",
        "emt_apical_junction",
        "hypoxia_glycolysis",
        "e2f_targets",
        "g2m_checkpoint",
        "emt",
    ]
    tumor_normal_cohorts = ["BRCA", "LUAD", "LUSC", "KIRC", "LIHC", "THCA", "PRAD"]
    tumor_normal_panels = [
        "cell_cycle_core",
        "cell_cycle_proliferation",
        "hormone_differentiation",
        "emt_apical_junction",
        "hypoxia_glycolysis",
        "immune_inflammatory",
        "e2f_targets",
        "g2m_checkpoint",
        "emt",
        "tnfa_nfkb",
    ]
    candidates: list[DataCandidate] = []
    for panel in brca_panels:
        candidates.append(
            DataCandidate(
                candidate_id=f"BRCA_subtype_{panel}",
                cohort="BRCA",
                contrast="brca_subtype",
                group1_label="Basal-like",
                group2_label="Luminal A",
                panel_id=panel,
                biological_rationale="PAM50 Basal-like and Luminal A tumors differ in proliferation, hormone signaling, and differentiation programs.",
            )
        )
    for cohort in tumor_normal_cohorts:
        for panel in tumor_normal_panels:
            candidates.append(
                DataCandidate(
                    candidate_id=f"{cohort}_tumor_normal_{panel}",
                    cohort=cohort,
                    contrast="tumor_normal",
                    group1_label="Primary tumor",
                    group2_label="Solid tissue normal",
                    panel_id=panel,
                    biological_rationale="Tumor and adjacent normal tissues are expected to differ in coordinated cancer pathway regulation.",
                )
            )
    return candidates


def xena_paths_for_cohort(cohort: str, cache: Path) -> tuple[Path, Path]:
    clinical = download_file(
        XENA_CLINICAL_URL_TEMPLATE.format(cohort=cohort),
        cache / f"{cohort}_clinicalMatrix.tsv",
    )
    expr = download_file(
        XENA_EXPR_URL_TEMPLATE.format(cohort=cohort),
        cache / f"{cohort}_HiSeqV2.gz",
    )
    return clinical, expr


def load_candidate_program(
    candidate: DataCandidate,
    cache: Path,
    gene_sets: dict[str, list[str]],
    max_p: int | None = 500,
) -> ProgramData | None:
    try:
        clinical_path, expr_path = xena_paths_for_cohort(candidate.cohort, cache)
        clinical = read_clinical(clinical_path, cohort=candidate.cohort, contrast=candidate.contrast)
        expr = load_expression_subset(expr_path, clinical["sampleID"])
    except Exception:
        return None
    matched_clinical = clinical[clinical["sampleID"].isin(expr.columns)].copy()
    if matched_clinical.empty:
        return None
    counts = matched_clinical["group"].value_counts()
    if candidate.group1_label not in counts or candidate.group2_label not in counts:
        return None
    min_p = max(100, int(min(counts[candidate.group1_label], counts[candidate.group2_label])))
    programs = build_program_data(
        expr,
        matched_clinical,
        gene_sets,
        min_p=min_p,
        max_p=max_p,
        candidate=candidate,
    )
    return programs[0] if programs else None


def rank_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, sub in raw.groupby("candidate_id", sort=False):
        by = sub.set_index("method")
        row0 = by.iloc[0]
        benchmark_rejects = int(by.loc[["Max-bootstrap", "Wang-NR", "Li-Chen"], "reject"].sum())
        both_ours = bool(by.loc["Ours-I", "reject"]) and bool(by.loc["Ours-C", "reject"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "cohort": row0["cohort"],
                "contrast": row0["contrast"],
                "group1": row0["group1"],
                "group2": row0["group2"],
                "program": row0["program"],
                "p": int(row0["p"]),
                "n1": int(row0["n1"]),
                "n2": int(row0["n2"]),
                "ours_i_p": float(by.loc["Ours-I", "p_value"]),
                "ours_c_p": float(by.loc["Ours-C", "p_value"]),
                "max_p": float(by.loc["Max-bootstrap", "p_value"]),
                "wang_p": float(by.loc["Wang-NR", "p_value"]),
                "li_chen_p": float(by.loc["Li-Chen", "p_value"]),
                "both_ours_reject": both_ours,
                "benchmark_rejects": benchmark_rejects,
                "selection_score": (0 if both_ours else 1, -benchmark_rejects, float(by.loc["Ours-C", "p_value"])),
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(["both_ours_reject", "benchmark_rejects", "ours_c_p"], ascending=[False, False, True])
    return ranked.drop(columns=["selection_score"])


def _candidate_by_id(candidate_id: str) -> DataCandidate:
    for candidate in default_candidates():
        if candidate.candidate_id == candidate_id:
            return candidate
    raise ValueError(f"Unknown real-data candidate_id: {candidate_id}")


def _program_meta(programs: list[ProgramData]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": p.candidate_id,
                "cohort": p.cohort,
                "contrast": p.contrast,
                "group1": p.group1_label,
                "group2": p.group2_label,
                "program_id": p.program_id,
                "program": p.label,
                "n1": p.n1,
                "n2": p.n2,
                "p": p.p,
                "biological_rationale": p.biological_rationale,
                "components": ";".join(p.components),
                "genes": ";".join(p.genes),
            }
            for p in programs
        ]
    )


def _export_realdata_outputs(
    raw: pd.DataFrame,
    programs: list[ProgramData],
    gene_sets: dict[str, list[str]],
    out: Path,
    tag: str,
) -> str:
    raw.to_csv(out / "raw" / f"{tag}_raw.csv", index=False)
    raw.to_csv(out / "summary" / f"{tag}_method_results.csv", index=False)
    _program_meta(programs).to_csv(out / "summary" / f"{tag}_programs.csv", index=False)
    ranked = rank_candidates(raw)
    ranked.to_csv(out / "summary" / f"{tag}_ranked_candidates.csv", index=False)
    primary_id = select_primary_program(raw)
    primary = next(p for p in programs if p.candidate_id == primary_id)
    write_program_table(raw, out / "tables" / f"{tag}_program_method_pvalues.tex")
    write_selected_program_table(primary, raw, out / "tables" / f"{tag}_selected_program.tex")
    plot_component_correlation_heatmap(primary, gene_sets, out / "figures" / f"{tag}_component_correlation_heatmap.png")
    (out / "summary" / f"{tag}_selected_candidate.txt").write_text(primary_id + "\n", encoding="utf-8")
    return primary_id


def _read_selected_candidate(out: Path) -> str | None:
    for name in [
        "realdata_screen_final_selected_candidate.txt",
        "realdata_screen_debug_selected_candidate.txt",
        "realdata_final_selected_candidate.txt",
    ]:
        path = out / "summary" / name
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    ranked_paths = [
        out / "summary" / "realdata_screen_final_ranked_candidates.csv",
        out / "summary" / "realdata_screen_debug_ranked_candidates.csv",
    ]
    for path in ranked_paths:
        if path.exists():
            ranked = pd.read_csv(path)
            if not ranked.empty:
                return str(ranked.iloc[0]["candidate_id"])
    return None


def run_realdata_pipeline(
    profile: str,
    B: int | None,
    alpha: float,
    cache_dir: str | Path,
    out_dir: str | Path,
    max_p: int | None = 500,
    batch_size: int = 100,
    seed: int = 20260521,
    mode: str = "screen",
    selected_candidate: str | None = None,
    cohorts: list[str] | None = None,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    cache = Path(cache_dir)
    out = Path(out_dir)
    for sub in ["raw", "summary", "tables", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    if B is None:
        B = 200 if profile == "debug" else 2000
    gmt_path = download_file(HALLMARK_GMT_URL, cache / "h.all.v2025.1.Hs.symbols.gmt")
    gene_sets = parse_gmt(gmt_path)

    if mode not in {"screen", "final-selected"}:
        raise ValueError("mode must be 'screen' or 'final-selected'.")

    if mode == "final-selected":
        selected_candidate = selected_candidate or _read_selected_candidate(out)
        if not selected_candidate:
            raise RuntimeError("No selected candidate was supplied or found from a previous screen run.")
        candidate_list = [_candidate_by_id(selected_candidate)]
        tag = "realdata_final" if profile == "final" else f"realdata_final_{profile}"
    else:
        candidate_list = default_candidates()
        if cohorts:
            allowed = {c.upper() for c in cohorts}
            candidate_list = [c for c in candidate_list if c.cohort.upper() in allowed]
        if max_candidates is not None:
            candidate_list = candidate_list[:max_candidates]
        tag = f"realdata_screen_{profile}"

    programs: list[ProgramData] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidate_list:
        program = load_candidate_program(candidate, cache=cache, gene_sets=gene_sets, max_p=max_p)
        if program is None:
            skipped.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "cohort": candidate.cohort,
                    "contrast": candidate.contrast,
                    "panel_id": candidate.panel_id,
                    "reason": "download/load failure or dimension filter",
                }
            )
            continue
        programs.append(program)
    if skipped:
        pd.DataFrame(skipped).to_csv(out / "logs" / f"{tag}_skipped_candidates.csv", index=False)
    if not programs:
        raise RuntimeError("No real-data candidate passed the dimension and loading filters.")

    rows: list[dict[str, Any]] = []
    for i, program in enumerate(programs):
        rows.extend(run_methods_for_program(program, B=B, alpha=alpha, seed=seed + 7919 * i, batch_size=batch_size))
    raw = pd.DataFrame(rows)
    _export_realdata_outputs(raw, programs, gene_sets, out, tag)
    return raw
