from __future__ import annotations

import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from covtestsim.covariances import draw_samples, equicorrelation, toeplitz_covariance
from covtestsim.methods import (
    li_chen_exact,
    li_chen_a_component,
    li_chen_c_component,
    ours_l2_bootstrap,
    wang_gram_traces,
)
from covtestsim.realdata import (
    DataCandidate,
    build_program_data,
    default_candidates,
    is_primary_tumor,
    normalize_subtype,
    parse_gmt,
    run_methods_for_program,
    tcga_patient_barcode,
)
from covtestsim.studies import run_study
from covtestsim.target import adaptive_target_transform
from covtestsim.utils import sample_cov_n, vech


def brute_li_chen_a(x: np.ndarray) -> float:
    n = x.shape[0]
    term1 = sum((x[i] @ x[j]) ** 2 for i in range(n) for j in range(n) if i != j)
    term1 /= n * (n - 1)
    term2 = 0.0
    for i, j, k in itertools.permutations(range(n), 3):
        term2 += (x[i] @ x[j]) * (x[j] @ x[k])
    term2 *= 2.0 / (n * (n - 1) * (n - 2))
    term3 = 0.0
    for i, j, k, ell in itertools.permutations(range(n), 4):
        term3 += (x[i] @ x[j]) * (x[k] @ x[ell])
    term3 /= n * (n - 1) * (n - 2) * (n - 3)
    return term1 - term2 + term3


def brute_li_chen_c(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = x.shape[0], y.shape[0]
    term1 = sum((x[i] @ y[j]) ** 2 for i in range(n1) for j in range(n2)) / (n1 * n2)
    term2a = 0.0
    for i, k in itertools.permutations(range(n1), 2):
        for j in range(n2):
            term2a += (x[i] @ y[j]) * (y[j] @ x[k])
    term2a /= n1 * n2 * (n1 - 1)
    term2b = 0.0
    for j, ell in itertools.permutations(range(n2), 2):
        for i in range(n1):
            term2b += (y[j] @ x[i]) * (x[i] @ y[ell])
    term2b /= n1 * n2 * (n2 - 1)
    term4 = 0.0
    for i, k in itertools.permutations(range(n1), 2):
        for j, ell in itertools.permutations(range(n2), 2):
            term4 += (x[i] @ y[j]) * (x[k] @ y[ell])
    term4 /= n1 * n2 * (n1 - 1) * (n2 - 1)
    return term1 - term2a - term2b + term4


class CoreTests(unittest.TestCase):
    def test_vech_ordering(self) -> None:
        mat = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
        np.testing.assert_allclose(vech(mat), np.array([1, 2, 3, 4, 5, 6], dtype=float))

    def test_covariances_are_positive_definite(self) -> None:
        for sigma in [equicorrelation(20, 0.9), toeplitz_covariance(20, 0.9)]:
            self.assertGreater(np.linalg.eigvalsh(sigma).min(), 0)

    def test_sample_covariance_denominator_n(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
        xc = x - x.mean(axis=0)
        np.testing.assert_allclose(sample_cov_n(x), xc.T @ xc / 3)

    def test_supported_innovations_draw_finite_samples(self) -> None:
        rng = np.random.default_rng(2026)
        sigma = equicorrelation(6, 0.3)
        for innovation in ["gaussian", "chisq1", "t5"]:
            x = draw_samples(rng, 20, sigma, innovation=innovation)
            self.assertEqual(x.shape, (20, 6))
            self.assertTrue(np.isfinite(x).all())

    def test_target_transform_stability(self) -> None:
        rng = np.random.default_rng(123)
        x = rng.standard_normal((12, 5))
        y = rng.standard_normal((13, 5))
        c_hat, diag = adaptive_target_transform(x, y)
        self.assertEqual(c_hat.shape, (5, 5))
        self.assertTrue(np.isfinite(c_hat).all())
        self.assertGreater(diag["target_ridge"], 0)

    def test_wang_traces_match_explicit_induced_covariances(self) -> None:
        rng = np.random.default_rng(1234)
        x = rng.standard_normal((7, 3))
        y = rng.standard_normal((8, 3))
        traces = wang_gram_traces(x, y)

        def induced(z: np.ndarray) -> np.ndarray:
            z0 = z - z.mean(axis=0, keepdims=True)
            w = np.array([np.outer(row, row).reshape(-1) for row in z0])
            return w - w.mean(axis=0, keepdims=True)

        w1 = induced(x)
        w2 = induced(y)
        o1 = w1.T @ w1 / (x.shape[0] - 1)
        o2 = w2.T @ w2 / (y.shape[0] - 1)
        self.assertAlmostEqual(traces["tr1"], np.trace(o1), places=10)
        self.assertAlmostEqual(traces["tr2"], np.trace(o2), places=10)
        self.assertAlmostEqual(traces["tr1_sq"], np.trace(o1 @ o1), places=10)
        self.assertAlmostEqual(traces["tr2_sq"], np.trace(o2 @ o2), places=10)
        self.assertAlmostEqual(traces["tr1_cu"], np.trace(o1 @ o1 @ o1), places=10)
        self.assertAlmostEqual(traces["tr2_cu"], np.trace(o2 @ o2 @ o2), places=10)
        self.assertAlmostEqual(traces["tr12"], np.trace(o1 @ o2), places=10)
        self.assertAlmostEqual(traces["tr1_sq_2"], np.trace(o1 @ o1 @ o2), places=10)
        self.assertAlmostEqual(traces["tr1_2_sq"], np.trace(o1 @ o2 @ o2), places=10)

    def test_li_chen_components_match_bruteforce(self) -> None:
        rng = np.random.default_rng(321)
        x = rng.standard_normal((5, 3))
        y = rng.standard_normal((6, 3))
        self.assertAlmostEqual(li_chen_a_component(x), brute_li_chen_a(x), places=10)
        self.assertAlmostEqual(li_chen_c_component(x, y), brute_li_chen_c(x, y), places=10)

    def test_li_chen_uses_page7_denominator_directly(self) -> None:
        rng = np.random.default_rng(55)
        x = rng.standard_normal((8, 4))
        y = rng.standard_normal((9, 4))
        result = li_chen_exact(x, y)
        denom = result.diagnostics["denominator_raw"]
        self.assertGreater(denom, 0)
        self.assertAlmostEqual(result.statistic, result.diagnostics["T_raw"] / denom, places=12)

    def test_ours_bootstrap_p_value_is_valid(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.standard_normal((10, 4))
        y = rng.standard_normal((11, 4))
        result = ours_l2_bootstrap(x, y, B=8, alpha=0.05, rng=rng, batch_size=4)
        self.assertTrue(0.0 <= result.p_value <= 1.0)
        self.assertGreater(result.diagnostics["d"], 0)

    def test_smoke_study_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = run_study("study3", profile="smoke", out_dir=tmp, R=1, B=4)
            self.assertFalse(raw.empty)
            self.assertTrue(((raw["p_value"].dropna() >= 0) & (raw["p_value"].dropna() <= 1)).all())
            self.assertTrue((Path(tmp) / "raw" / "study3_smoke_raw.csv").exists())
            self.assertTrue((Path(tmp) / "summary" / "study3_smoke_summary_long.csv").exists())

    def test_smoke_study1_dist_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = run_study("study1_dist", profile="smoke", out_dir=tmp, R=1, B=4)
            self.assertEqual(set(raw["innovation"]), {"gaussian", "chisq1", "t5"})
            self.assertTrue(((raw["p_value"].dropna() >= 0) & (raw["p_value"].dropna() <= 1)).all())
            self.assertTrue((Path(tmp) / "summary" / "study1_dist_smoke_paper_size_wide.csv").exists())

    def test_tcga_barcode_helpers(self) -> None:
        self.assertEqual(tcga_patient_barcode("TCGA-AB-1234-01A-01R"), "TCGA-AB-1234")
        self.assertTrue(is_primary_tumor("TCGA-AB-1234-01"))
        self.assertFalse(is_primary_tumor("TCGA-AB-1234-11"))
        self.assertEqual(normalize_subtype("Basal"), "Basal-like")
        self.assertEqual(normalize_subtype("LumA"), "Luminal A")

    def test_parse_gmt_and_program_dimension_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gmt = Path(tmp) / "hallmark.gmt"
            gmt.write_text(
                "HALLMARK_E2F_TARGETS\turl\tG1\tG2\tG3\n"
                "HALLMARK_G2M_CHECKPOINT\turl\tG2\tG4\n"
                "HALLMARK_MYC_TARGETS_V1\turl\tG5\n"
                "HALLMARK_MYC_TARGETS_V2\turl\tG6\n"
                "HALLMARK_DNA_REPAIR\turl\tG7\n"
                "HALLMARK_MITOTIC_SPINDLE\turl\tG8\n",
                encoding="utf-8",
            )
            gene_sets = parse_gmt(gmt)
            self.assertIn("HALLMARK_E2F_TARGETS", gene_sets)
            expr = np.arange(8 * 6, dtype=float).reshape(8, 6)
            frame = __import__("pandas").DataFrame(
                expr,
                index=[f"G{i}" for i in range(1, 9)],
                columns=[f"S{i}" for i in range(6)],
            )
            clinical = __import__("pandas").DataFrame(
                {
                    "sampleID": [f"S{i}" for i in range(6)],
                    "group": ["Basal-like", "Basal-like", "Basal-like", "Luminal A", "Luminal A", "Luminal A"],
                }
            )
            programs = build_program_data(frame, clinical, gene_sets, min_p=4, max_p=5)
            self.assertEqual(len(programs), 0)
            programs = build_program_data(frame, clinical, gene_sets, min_p=4, max_p=8)
            self.assertEqual(len(programs), 1)
            self.assertGreaterEqual(programs[0].p, 4)

    def test_realdata_candidates_are_biologically_named(self) -> None:
        candidates = default_candidates()
        self.assertGreater(len(candidates), 5)
        for candidate in candidates:
            self.assertIsInstance(candidate, DataCandidate)
            self.assertTrue(candidate.candidate_id)
            self.assertTrue(candidate.group1_label)
            self.assertTrue(candidate.group2_label)
            self.assertNotEqual(candidate.group1_label, candidate.group2_label)
            self.assertTrue(candidate.biological_rationale)

    def test_realdata_method_runner_small_matrix(self) -> None:
        from covtestsim.realdata import ProgramData

        rng = np.random.default_rng(2027)
        program = ProgramData(
            candidate_id="toy_candidate",
            cohort="TOY",
            contrast="toy_contrast",
            group1_label="Group 1",
            group2_label="Group 2",
            program_id="toy",
            label="Toy",
            genes=[f"G{i}" for i in range(4)],
            components=["HALLMARK_E2F_TARGETS"],
            x=rng.standard_normal((10, 4)),
            y=rng.standard_normal((11, 4)),
            n1=10,
            n2=11,
            p=4,
            biological_rationale="Synthetic method-runner test.",
        )
        rows = run_methods_for_program(program, B=4, alpha=0.05, seed=1, batch_size=2)
        self.assertEqual({r["method"] for r in rows}, {"Ours-I", "Ours-C", "Max-bootstrap", "Wang-NR", "Li-Chen"})
        for row in rows:
            self.assertTrue(np.isfinite(row["statistic"]))
            self.assertTrue(0.0 <= row["p_value"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
