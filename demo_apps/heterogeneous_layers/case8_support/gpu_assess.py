"""Summarize fresh case-8 GPU precision, reciprocity, and FEM differences."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from demo_apps.heterogeneous_layers.case8_support.reference_io import load_fem_matrix
from demo_apps.heterogeneous_layers.case8_support.full_case8 import make_case
from heterogeneous_wos.boxes import EPSILON_0



def binomial_interval(count, total, alpha):
    """Clopper-Pearson interval for one hit category."""
    lower = 0.0 if count == 0 else float(beta.ppf(alpha / 2, count, total - count + 1))
    upper = 1.0 if count == total else float(beta.ppf(1 - alpha / 2, count + 1, total - count))
    return lower, upper


def signed_hit_intervals(path_data, rows, matrix, alpha):
    """Conservative intervals for +/- fixed-weight terminal hit payoffs."""
    factor = EPSILON_0 * make_case()["length_unit_m"] * 1e15
    intervals = np.empty((8, 8, 2), dtype=float)
    hit_counts = np.empty((8, 8, 2), dtype=int)
    with np.load(path_data, allow_pickle=False) as paths:
        for i, row in enumerate(rows):
            terminal = row["terminal"]
            labels = paths[f"row{terminal}_labels"]
            weights = paths[f"row{terminal}_weights"]
            n = len(labels)
            if n != row["paths"] or not np.isin(labels, np.arange(9)).all():
                raise ValueError(f"Invalid path labels for terminal {terminal}")
            magnitude = float(np.abs(weights[0]))
            if not np.allclose(np.abs(weights), magnitude, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Nonconstant signed charge weight for terminal {terminal}")
            for j in range(1, 9):
                positive = int(np.count_nonzero((labels == j) & (weights > 0)))
                negative = int(np.count_nonzero((labels == j) & (weights < 0)))
                p_lo, p_hi = binomial_interval(positive, n, alpha / 2)
                m_lo, m_hi = binomial_interval(negative, n, alpha / 2)
                sampled = factor * magnitude * (positive - negative) / n
                exact = matrix[i, j] - sampled
                intervals[i, j - 1] = (exact + factor * magnitude * (p_lo - m_hi),
                                       exact + factor * magnitude * (p_hi - m_lo))
                hit_counts[i, j - 1] = (positive, negative)
    return intervals, hit_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--target-se", type=float, default=0.05,
                        help="Target maximum self and off-diagonal sampling SE in fF")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fem-reference", type=Path,
                        help="Optional fresh FEM JSON or labeled matrix CSV")
    parser.add_argument("--path-data", type=Path,
                        help="Per-path NPZ for exact rare-hit confidence intervals")
    parser.add_argument("--figure", type=Path, help="Save potential, GPU, FEM, and difference heatmaps")
    parser.add_argument("--wos-potential", type=Path, help="GPU terminal-hit potential NPZ for the figure")
    args = parser.parse_args()
    if args.path_data and not args.fem_reference:
        parser.error("--path-data requires --fem-reference")
    if args.figure and not args.fem_reference:
        parser.error("--figure requires --fem-reference")
    if args.figure and not args.wos_potential:
        parser.error("--figure requires --wos-potential")
    report = json.loads(args.report.read_text())
    rows = sorted(report["rows"], key=lambda item: item["terminal"])
    if [item["terminal"] for item in rows] != list(range(1, 9)):
        parser.error("Report must contain all eight non-ground case-8 rows")
    matrix = np.asarray([item["C_fF"] for item in rows])
    se = np.asarray([item["standard_error_fF"] for item in rows])
    fem = load_fem_matrix(args.fem_reference) if args.fem_reference else None
    reference_label = "FEM"
    if args.fem_reference and args.fem_reference.suffix.lower() == ".json":
        fem_record = json.loads(args.fem_reference.read_text())
        if isinstance(fem_record, list):
            fem_record = fem_record[-1]
        if fem_record.get("order") is not None:
            reference_label = f"Order-{fem_record['order']} FEM"
        if fem_record.get("case_sha256") and fem_record["case_sha256"] != report["case_sha256"]:
            parser.error("GPU and FEM reports were generated from different case JSON")
    self_se = np.diag(se[:, 1:])
    mutual_mask = np.ones_like(se, dtype=bool)
    mutual_mask[np.arange(8), np.arange(1, 9)] = False
    reciprocity = matrix[:, 1:] - matrix[:, 1:].T
    reciprocal_se = np.sqrt(se[:, 1:] ** 2 + se[:, 1:].T ** 2)
    offdiag = ~np.eye(8, dtype=bool)
    reciprocal_z = np.abs(reciprocity[offdiag]) / np.where(
        reciprocal_se[offdiag] > 0, reciprocal_se[offdiag], np.inf)
    if fem is not None:
        comparison = fem[1:] if fem.shape == (9, 9) else fem
        gpu_comparison = matrix if fem.shape == (9, 9) else matrix[:, 1:]
        se_comparison = se if fem.shape == (9, 9) else se[:, 1:]
        zero_se = se_comparison == 0
        unresolved_zero_se = zero_se & (gpu_comparison != comparison)
        fem_z = np.abs(gpu_comparison - comparison) / np.where(
            se_comparison > 0, se_comparison, np.inf)
    else:
        fem_z = None
    assessment = {
        "report": str(args.report),
        "paths_per_row": report["parameters"]["paths"],
        "tolerance": report["parameters"]["tolerance"],
        "unfinished": sum(item["unfinished"] for item in rows),
        "max_self_sampling_SE_fF": float(self_se.max()),
        "max_mutual_sampling_SE_fF": float(se[mutual_mask].max()),
        "sampling_precision_target_fF": args.target_se,
        "sampling_precision_met": bool(max(self_se.max(), se[mutual_mask].max()) <= args.target_se),
        "maximum_reciprocity_gap_fF": float(np.max(np.abs(reciprocity[offdiag]))),
        "maximum_reciprocity_gap_over_combined_SE": float(np.max(reciprocal_z)),
        "maximum_FEM_difference_over_sampling_SE": float(np.max(fem_z)) if fem_z is not None else None,
        "FEM_entries_with_zero_GPU_sampling_SE": int(np.count_nonzero(zero_se)) if fem is not None else None,
        "FEM_differences_with_zero_GPU_sampling_SE": int(np.count_nonzero(unresolved_zero_se)) if fem is not None else None,
        "maximum_FEM_difference_fF": float(np.max(np.abs(gpu_comparison - comparison))) if fem is not None else None,
        "self_capacitance_fF": np.diag(matrix[:, 1:]).tolist(),
        "FEM_self_capacitance_fF": np.diag(fem[1:, 1:] if fem.shape == (9, 9) else fem).tolist() if fem is not None else None,
        "mean_steps_by_row": [item["mean_steps"] for item in rows],
    }
    if fem is not None and args.path_data:
        reference = fem[1:, 1:] if fem.shape == (9, 9) else fem
        entry_intervals, hit_counts = signed_hit_intervals(args.path_data, rows, matrix, 0.05)
        family_intervals, _ = signed_hit_intervals(args.path_data, rows, matrix, 0.05 / 64)
        outside_entry = (reference < entry_intervals[:, :, 0]) | (reference > entry_intervals[:, :, 1])
        outside_family = (reference < family_intervals[:, :, 0]) | (reference > family_intervals[:, :, 1])
        assessment.update({
            "FEM_entries_outside_95pct_hit_interval": int(outside_entry.sum()),
            "FEM_entries_outside_95pct_familywise_hit_interval": int(outside_family.sum()),
            "zero_hit_entries": [[int(i + 1), int(j + 1)] for i, j in np.argwhere(hit_counts.sum(axis=2) == 0)],
            "familywise_interval_max_half_width_fF": float(np.max((family_intervals[:, :, 1] - family_intervals[:, :, 0]) / 2)),
            "hit_interval_method": "Two Clopper-Pearson binomial intervals for positive and negative terminal hits; Bonferroni union bound",
        })
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        reference = fem[1:, 1:] if fem.shape == (9, 9) else fem
        estimate = matrix[:, 1:]
        differences = estimate - reference
        with np.load(args.wos_potential, allow_pickle=False) as saved:
            x = saved["x"]
            z = saved["z"]
            voltage = saved["potential"]
            y = float(saved["y"])
            pointwise_se = saved["standard_error"]
            paths_per_pixel = int(saved["paths_per_pixel"])
            if (str(saved["case_sha256"]) != report["case_sha256"] or
                str(saved["gpu_implementation_sha256"]) != report["gpu_implementation_sha256"] or
                int(saved["terminal"]) != 1 or
                voltage.shape != (len(z), len(x)) or pointwise_se.shape != voltage.shape):
                raise ValueError("Saved WoS potential does not match case 8 or the GPU implementation")
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad("#e8eff7")
        potential = axes[0].imshow(voltage, origin="lower",
                                   extent=(x[0], x[-1], z[0], z[-1]),
                                   cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for conductor in make_case()["conductors"]:
            for box in conductor["boxes"]:
                lo, hi = np.asarray(box)
                if lo[1] <= y <= hi[1]:
                    axes[0].add_patch(Rectangle((lo[0], lo[2]), hi[0]-lo[0], hi[2]-lo[2],
                        facecolor="none", edgecolor="white", linewidth=0.7))
        axes[0].set(title=f"WoSX potential · T1 · y = {y:g}\n"
                          f"{paths_per_pixel} paths/pixel · max SE {pointwise_se.max():.3f} V",
                    xlabel="x (0.1 µm)", ylabel="z (0.1 µm)")
        fig.colorbar(potential, ax=axes[0], shrink=0.75, label="Potential (V)")
        panels = ((estimate, "WoSX GPU (fF)", "coolwarm"),
                  (reference, f"{reference_label} (fF)", "coolwarm"),
                  (differences, "GPU − FEM (fF)", "PiYG"))
        common = max(np.abs(estimate).max(), np.abs(reference).max())
        for panel_index, (axis, (values, title, cmap)) in enumerate(zip(axes[1:], panels)):
            limit = np.abs(values).max() if panel_index == 2 else common
            view = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit)
            axis.set_title(title + (f"\ncolor scale enlarged {common / limit:.0f}×" if panel_index == 2 else ""))
            axis.set_xticks(range(8), labels=range(1, 9))
            axis.set_yticks(range(8), labels=range(1, 9))
            axis.set_xlabel("Excitation terminal")
            axis.set_ylabel("Charge terminal")
            fig.colorbar(view, ax=axis, shrink=0.75)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)
        assessment["figure"] = str(args.figure)
        assessment["potential_paths_per_pixel"] = paths_per_pixel
        assessment["potential_max_pointwise_SE_V"] = float(pointwise_se.max())
    print(json.dumps(assessment, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(assessment, indent=2) + "\n")


if __name__ == "__main__":
    main()
