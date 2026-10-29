"""Run paired GPU tolerance levels on fresh case-8 signed-charge samples."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heterogeneous_wos.boxes import EPSILON_0
from demo_apps.heterogeneous_layers.case8_support.full_case8 import make_case
from demo_apps.heterogeneous_layers.case8_support.reference_io import load_fem_matrix
from demo_apps.heterogeneous_layers.case8_support.gpu_wos import implementation_sha256
RUNNER = ROOT / "demo_apps/heterogeneous_layers/case8_support/gpu_wos.py"


def paired_difference(coarse, fine, row, nterm, factor):
    key = f"row{row}"
    a = np.asarray(coarse[f"{key}_labels"], dtype=np.int32)
    b = np.asarray(fine[f"{key}_labels"], dtype=np.int32)
    wa = np.asarray(coarse[f"{key}_weights"])
    wb = np.asarray(fine[f"{key}_weights"])
    starts_equal = np.array_equal(coarse[f"{key}_starts"], fine[f"{key}_starts"])
    weights_equal = np.array_equal(wa, wb)
    if not starts_equal or not weights_equal:
        raise RuntimeError(f"Row {row} is not paired: sampled starts or weights differ")
    difference = np.zeros((len(a), nterm))
    difference[np.arange(len(a)), a] += wa
    difference[np.arange(len(b)), b] -= wb
    mean = difference.mean(axis=0) * factor
    se = difference.std(axis=0, ddof=1) * factor / np.sqrt(len(a))
    return {"row": row, "mean_difference_fF": mean.tolist(),
            "paired_standard_error_fF": se.tolist(),
            "changed_terminal_fraction": float(np.mean(a != b))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--rows", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--tolerances", type=float, nargs="+",
                        default=[2e-4, 1e-4, 5e-5, 2.5e-5])
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--continuations", type=int, default=200)
    parser.add_argument("--resume", action="store_true", help="Reuse completed levels with matching paths and seed")
    parser.add_argument("--fem-reference", type=Path,
                        help="Optional fresh FEM JSON or labeled matrix CSV")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/case8/wos/convergence")
    args = parser.parse_args()
    if args.paths < 2 or len(args.tolerances) < 2 or any(t <= 0 for t in args.tolerances):
        parser.error("Need at least two paths and two positive tolerances")
    if len(set(args.rows)) != len(args.rows) or any(row not in range(1, 9) for row in args.rows):
        parser.error("Rows must be unique members of 1..8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    executable = ROOT / "build/demo_apps/heterogeneous_rectilinear_walk_gpu"
    implementation = implementation_sha256(executable)
    deck_hash = hashlib.sha256((ROOT / "demo_apps/heterogeneous_layers/case8/case8.cap3d").read_bytes()).hexdigest()
    levels = []
    for tolerance in args.tolerances:
        name = "tol-" + f"{tolerance:.1e}".replace(".0e", "e")
        report_path = args.output_dir / f"{name}.json"
        paths_path = args.output_dir / f"{name}-paths.npz"
        command = [sys.executable, str(RUNNER), "--rows", *map(str, args.rows),
                   "--paths", str(args.paths), "--seed", str(args.seed),
                   "--continuations", str(args.continuations),
                   "--tolerance", str(tolerance),
                   "--interface-tolerance", str(tolerance),
                   "--output", str(report_path), "--path-output", str(paths_path)]
        reuse = False
        if args.resume and report_path.is_file() and paths_path.is_file():
            prior = json.loads(report_path.read_text())
            reuse = (prior["parameters"]["paths"] == args.paths and
                     prior["parameters"]["seed"] == args.seed and
                     prior["parameters"]["tolerance"] == tolerance and
                     prior.get("gpu_implementation_sha256") == implementation and
                     prior.get("source_deck_sha256") == deck_hash and
                     [entry["terminal"] for entry in prior["rows"]] == args.rows and
                     all(entry["unfinished"] == 0 for entry in prior["rows"]))
        if reuse:
            print(f"Reusing completed {name}", flush=True)
        else:
            print(f"Running {name}: {args.paths} paths per row", flush=True)
            subprocess.run(command, check=True)
        levels.append((tolerance, json.loads(report_path.read_text()), paths_path))
    reference = load_fem_matrix(args.fem_reference) if args.fem_reference else None
    metadata = make_case()
    factor = EPSILON_0 * metadata["length_unit_m"] * 1e15
    result = {"case": "case8", "paths_per_row": args.paths, "seed": args.seed,
              "rows": args.rows, "levels": [], "paired_differences": []}
    for tolerance, report, paths_path in levels:
        matrix = np.asarray([row["C_fF"] for row in report["rows"]])
        se = np.asarray([row["standard_error_fF"] for row in report["rows"]])
        if reference is not None:
            if reference.shape == (9, 9):
                expected, actual, uncertainty = reference[args.rows], matrix, se
            else:
                expected, actual, uncertainty = reference[np.array(args.rows) - 1], matrix[:, 1:], se[:, 1:]
            z = np.abs(actual - expected) / np.where(uncertainty > 1e-14, uncertainty, np.inf)
        else:
            z = None
        result["levels"].append({"tolerance": tolerance,
            "report": str(args.output_dir / ("tol-" + f"{tolerance:.1e}".replace(".0e", "e") + ".json")),
            "max_standard_error_fF": float(se.max()),
            "max_self_standard_error_fF": float(max(se[i, row] for i, row in enumerate(args.rows))),
            "max_FEM_difference_over_sampling_SE": float(z.max()) if z is not None else None,
            "mean_steps": float(np.mean([row["mean_steps"] for row in report["rows"]])),
            "unfinished": int(sum(row["unfinished"] for row in report["rows"]))})
    for (a_tol, a_report, a_path), (b_tol, b_report, b_path) in zip(levels, levels[1:]):
        with np.load(a_path) as a, np.load(b_path) as b:
            rows = [paired_difference(a, b, row, len(metadata["conductors"]), factor)
                    for row in args.rows]
        max_delta = max(abs(v) for item in rows for v in item["mean_difference_fF"])
        max_z = max((abs(m) / se for item in rows
                     for m, se in zip(item["mean_difference_fF"],
                                      item["paired_standard_error_fF"]) if se > 0),
                    default=0.0)
        result["paired_differences"].append({"coarse": a_tol, "fine": b_tol,
            "max_abs_difference_fF": float(max_delta),
            "max_abs_difference_over_paired_SE": float(max_z), "rows": rows})
    target = args.output_dir / "summary.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved paired study to {target}", flush=True)
    for level in result["levels"]:
        print(f"tol {level['tolerance']:.1e}: max SE {level['max_standard_error_fF']:.4g} fF; "
              f"unfinished {level['unfinished']}")
    for pair in result["paired_differences"]:
        print(f"{pair['coarse']:.1e} -> {pair['fine']:.1e}: "
              f"max paired difference {pair['max_abs_difference_fF']:.4g} fF; "
              f"max paired z {pair['max_abs_difference_over_paired_SE']:.3g}")


if __name__ == "__main__":
    main()
