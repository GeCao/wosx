"""Run fresh case-8 capacitance rows with WoSX's GPU rectilinear walker.

CPU code prepares the exact signed charge measure; WoSX runs the walks. The
result contains raw row estimates and sampling standard errors, with no
archived matrix values used by the calculation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from heterogeneous_wos.rectilinear_case import RectilinearCase
from heterogeneous_wos.gpu_rectilinear import prepare_geometry, walk_batch
from demo_apps.heterogeneous_layers.case8_support.full_case8 import make_case
from heterogeneous_wos.boxes import EPSILON_0


def implementation_sha256(executable):
    digest = hashlib.sha256()
    for path in (ROOT / "demo_apps/heterogeneous_layers/rectilinear_walk_gpu.slang",
                 ROOT / "include/wosx/gpu/point-estimation/rectilinear-queries.slang",
                 ROOT / "include/wosx/gpu/point-estimation/heterogeneous-interface.slang",
                 executable):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1])
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--continuations", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=5,
                        help="Additional resident-GPU batches for rare long walks")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--interface-tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--output", type=Path, default=ROOT / "data/case8/wos/gpu-wos.json")
    parser.add_argument("--path-output", type=Path, help="Save per-path labels, weights and steps for paired studies")
    args = parser.parse_args()
    if args.paths < 2 or args.max_steps < 1 or args.continuations < 0 or args.restarts < 0 or any(
            row not in range(1, 9) for row in args.rows) or len(set(args.rows)) != len(args.rows):
        parser.error("Use at least two paths, positive step budget, and rows 1 through 8")
    if args.tolerance <= 0 or args.interface_tolerance <= 0:
        parser.error("Walk tolerances must be positive")
    executable = ROOT / "build/demo_apps/heterogeneous_rectilinear_walk_gpu"
    if not executable.is_file():
        parser.error("Build heterogeneous_rectilinear_walk_gpu first")
    metadata = make_case()
    case = RectilinearCase(metadata)
    geometry = prepare_geometry(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path_arrays = {}
    factor = EPSILON_0 * metadata["length_unit_m"] * 1e15
    report = {"case_sha256": hashlib.sha256((json.dumps(metadata, indent=2) + "\n").encode()).hexdigest(),
              "source_deck_sha256": metadata["provenance"]["source_sha256"],
              "backend": "WoSX GPU rectilinear walk; CPU signed face measure",
              "gpu_implementation_sha256": implementation_sha256(executable),
              "parameters": {"paths": args.paths, "max_steps_per_pass": args.max_steps,
                             "continuations": args.continuations, "restarts": args.restarts,
                             "tolerance": args.tolerance,
                             "interface_tolerance": args.interface_tolerance,
                             "seed": args.seed},
              "terminals": [item["id"] for item in metadata["conductors"]], "rows": []}
    for row in args.rows:
        row_started = time.perf_counter()
        measure = case.charge_measure(row)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, row]))
        starts, weights = measure.sample(args.paths, rng)
        starts = starts.astype(np.float32)
        if not np.isfinite(starts).all() or not np.isfinite(weights).all():
            raise RuntimeError("Invalid signed charge samples")
        labels = np.full(args.paths, -2, dtype=np.int32)
        steps = np.zeros(args.paths, dtype=np.int64)
        crossings = np.zeros(args.paths, dtype=np.int64)
        reflections = np.zeros(args.paths, dtype=np.int64)
        result, endpoints = walk_batch(
            executable, case, geometry, starts, np.arange(args.paths), row,
            args.output.parent, args.max_steps, args.continuations,
            args.tolerance, args.interface_tolerance, args.seed)
        if np.any(result[:, 0] < -2) or np.any(result[:, 0] >= case.nt):
            raise RuntimeError("GPU returned an invalid terminal label")
        labels = result[:, 0].astype(np.int32)
        steps = result[:, 1].astype(np.int64)
        crossings = result[:, 2].astype(np.int64)
        reflections = result[:, 3].astype(np.int64)
        pending = np.flatnonzero(labels < 0)
        for restart in range(1, args.restarts + 1):
            if not len(pending):
                break
            retried, next_endpoints = walk_batch(
                executable, case, geometry, endpoints[pending], pending, row,
                args.output.parent, args.max_steps, args.continuations,
                args.tolerance, args.interface_tolerance,
                args.seed + restart * 1000003)
            labels[pending] = retried[:, 0].astype(np.int32)
            steps[pending] += retried[:, 1].astype(np.int64)
            crossings[pending] += retried[:, 2].astype(np.int64)
            reflections[pending] += retried[:, 3].astype(np.int64)
            endpoints[pending] = next_endpoints
            pending = np.flatnonzero(labels < 0)
        record = {"terminal": row, "paths": args.paths, "unfinished": len(pending),
                  "mean_steps": float(steps.mean()),
                  "p99_steps": float(np.percentile(steps, 99)),
                  "max_steps": int(steps.max()),
                  "walks_over_100k_steps": int(np.count_nonzero(steps > 100000)),
                  "elapsed_seconds": float(time.perf_counter() - row_started),
                  "mean_crossings": float(crossings.mean()),
                  "mean_reflections": float(reflections.mean()),
                  "signed_measure_audit": float(measure.mass.sum())}
        if len(pending):
            record["unfinished_endpoints"] = endpoints[pending].tolist()
            record["unfinished_steps"] = steps[pending].tolist()
            report["rows"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise RuntimeError(f"Row {row} has {len(pending)} unfinished paths")
        if args.path_output:
            path_arrays[f"row{row}_labels"] = labels
            path_arrays[f"row{row}_weights"] = weights
            path_arrays[f"row{row}_starts"] = starts
            path_arrays[f"row{row}_steps"] = steps
            path_arrays[f"row{row}_crossings"] = crossings
            path_arrays[f"row{row}_reflections"] = reflections
        payoff = np.zeros((args.paths, case.nt))
        payoff[np.arange(args.paths), labels] = weights
        mean = measure.exact + payoff.mean(axis=0)
        se = payoff.std(axis=0, ddof=1) / np.sqrt(args.paths)
        covariance = np.cov(payoff, rowvar=False, ddof=1) * factor * factor / args.paths
        record.update(C_fF=(mean * factor).tolist(),
                      standard_error_fF=(se * factor).tolist(),
                      covariance_of_mean_fF2=covariance.tolist())
        report["rows"].append(record)
        if set(entry["terminal"] for entry in report["rows"]) == set(range(1, case.nt)):
            matrix = np.asarray([entry["C_fF"][1:] for entry in
                                 sorted(report["rows"], key=lambda entry: entry["terminal"])])
            report["reciprocity_gap_max_fF"] = float(np.max(np.abs(matrix - matrix.T)))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"row {row}: {args.paths} completed; self {mean[row] * factor:.6g} "
              f"+/- {se[row] * factor:.3g} fF", flush=True)
    if args.path_output:
        args.path_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.path_output, **path_arrays)
    print(f"Saved fresh GPU results to {args.output}")


if __name__ == "__main__":
    main()
