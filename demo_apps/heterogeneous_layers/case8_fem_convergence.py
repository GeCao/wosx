"""Check case-8 FEM mesh and polynomial-order convergence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from demo_apps.heterogeneous_layers.case8_support.full_case8 import make_case

DEFAULT_OUTPUT = ROOT / "data/case8/fem"
BASE_MESH = ROOT / "assets/case8/fem-focus-hp2-factor02.vol"
REFINED_MESH = ROOT / "assets/case8/fem-focus-hp3-factor02.vol"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    value = json.loads(path.read_text())
    return value[-1] if isinstance(value, list) else value


def ensure_case(path):
    case = make_case()
    if not path.is_file() or json.loads(path.read_text()) != case:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(case, indent=2) + "\n")
    return case


def valid_result(path, case_hash, mesh_hash, targets, order=2):
    if not path.is_file():
        return False
    try:
        result = record(path)
        return (result["case_sha256"] == case_hash and result["mesh_sha256"] == mesh_hash
                and result["order"] == order and result["terminals"] == targets
                and result["bonus_intorder"] == 0
                and (result["variable_order"] is None if order == 2 else
                     result["variable_order"]["far_order"] == 3 and
                     result["variable_order"]["graded_element_max_order"] == 3 and
                     result["variable_order"]["roi"] == [[-4.0, -5.0, -3.0], [13.0, 5.0, 4.0]])
                and result["preconditioner"] == ("bddc" if order == 2 else "multigrid-blocktype-4")
                and result["residual_max"] <= 1e-8)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_solve(case_path, mesh_path, output_path, targets, threads, save_fields=None):
    command = [sys.executable, "-m", "demo_apps.heterogeneous_layers.case8_support.fem_fullcase_iterative",
               "--case", str(case_path), "--mesh", str(mesh_path),
               "--output", str(output_path), "--orders", "2", "--targets",
               *[str(target) for target in targets], "--threads", str(threads)]
    if save_fields is not None:
        command.extend(["--save-fields", str(save_fields)])
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def assess(coarse, refined, coarse_path, refined_path):
    for name, result in (("base", coarse), ("refined", refined)):
        if result["residual_max"] > 1e-8 or result["reciprocity"] > 1e-8:
            raise ValueError(f"{name} FEM solve did not converge or is not reciprocal")
    for key in ("case_sha256", "terminals", "order", "bonus_intorder"):
        if coarse[key] != refined[key]:
            raise ValueError(f"FEM records disagree on {key}")
    if coarse["mesh_sha256"] == refined["mesh_sha256"]:
        raise ValueError("The two FEM records use the same mesh")
    if len(coarse["terminals"]) != 8 or coarse["order"] != 2:
        raise ValueError("Expected complete order-2 case-8 matrices")
    for key in ("material_volumes", "terminal_surface_areas"):
        a, b = coarse[key], refined[key]
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError("Mesh material labels differ")
            a, b = [a[k] for k in sorted(a)], [b[k] for k in sorted(b)]
        if not np.allclose(a, b, rtol=1e-8, atol=1e-10):
            raise ValueError(f"Mesh geometry differs in {key}")
    c = np.asarray(coarse["C_fF"], dtype=float)
    r = np.asarray(refined["C_fF"], dtype=float)
    if c.shape != (8, 8) or r.shape != (8, 8) or not np.isfinite(c).all() or not np.isfinite(r).all():
        raise ValueError("FEM matrices must be finite 8x8 arrays")
    difference = r - c
    offdiagonal = ~np.eye(8, dtype=bool)
    largest = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    return {
        "case_sha256": coarse["case_sha256"],
        "order": coarse["order"],
        "base": {"file": str(coarse_path), "mesh_sha256": coarse["mesh_sha256"],
                 "elements": coarse["elements"], "dofs": coarse["dofs"],
                 "residual_max": coarse["residual_max"]},
        "refined": {"file": str(refined_path), "mesh_sha256": refined["mesh_sha256"],
                    "elements": refined["elements"], "dofs": refined["dofs"],
                    "residual_max": refined["residual_max"]},
        "maximum_absolute_change_fF": float(np.abs(difference).max()),
        "maximum_self_change_fF": float(np.abs(np.diag(difference)).max()),
        "maximum_mutual_change_fF": float(np.abs(difference[offdiagonal]).max()),
        "relative_frobenius_change": float(np.linalg.norm(difference) / np.linalg.norm(r)),
        "largest_change_terminals": [coarse["terminals"][i] for i in largest],
        "largest_change_signed_fF": float(difference[largest]),
        "self_capacitance_base_fF": np.diag(c).tolist(),
        "self_capacitance_refined_fF": np.diag(r).tolist(),
        "difference_refined_minus_base_fF": difference.tolist(),
        "interpretation": "Two-mesh change only; does not by itself bound continuum discretization error",
    }


def assess_enrichment(quadratic, enriched, quadratic_path, enriched_path):
    for key in ("case_sha256", "mesh_sha256", "terminals"):
        if quadratic[key] != enriched[key]:
            raise ValueError(f"Order-2 and order-4 FEM records disagree on {key}")
    if quadratic["order"] != 2 or enriched["order"] != 4:
        raise ValueError("Expected orders 2 and 4")
    if enriched["residual_max"] > 1e-8 or enriched["reciprocity"] > 1e-8:
        raise ValueError("Enriched FEM solve did not converge or is not reciprocal")
    lower = np.asarray(quadratic["C_fF"])
    higher = np.asarray(enriched["C_fF"])
    difference = higher - lower
    return {"case_sha256": quadratic["case_sha256"],
            "mesh_sha256": quadratic["mesh_sha256"],
            "quadratic_file": str(quadratic_path), "enriched_file": str(enriched_path),
            "quadratic_dofs": quadratic["dofs"], "enriched_dofs": enriched["dofs"],
            "maximum_absolute_change_fF": float(np.abs(difference).max()),
            "relative_frobenius_change": float(np.linalg.norm(difference) / np.linalg.norm(higher)),
            "self_capacitance_quadratic_fF": np.diag(lower).tolist(),
            "self_capacitance_enriched_fF": np.diag(higher).tolist(),
            "difference_enriched_minus_quadratic_fF": difference.tolist(),
            "interpretation": "Polynomial enrichment on the same mesh; combine with the mesh check before using FEM as a reference"}


def run_enriched(case_path, mesh_path, output_path, targets, threads):
    command = [sys.executable, "-m", "demo_apps.heterogeneous_layers.case8_support.fem_fullcase_iterative",
               "--case", str(case_path), "--mesh", str(mesh_path),
               "--output", str(output_path), "--orders", "4", "--targets",
               *[str(target) for target in targets], "--far-order", "3",
               "--graded-order", "3", "--roi", "-4", "-5", "-3", "13", "5", "4",
               "--multigrid", "--mg-blocktype", "4", "--threads", str(threads),
               "--tol", "1e-9"]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-mesh", type=Path, default=BASE_MESH)
    parser.add_argument("--refined-mesh", type=Path, default=REFINED_MESH)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--recompute", action="store_true", help="Rerun saved solves")
    parser.add_argument("--enriched", action="store_true", help="Also solve and assess the complete order-4 matrix")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for mesh in (args.base_mesh, args.refined_mesh):
        if not mesh.is_file():
            parser.error(f"Missing FEM mesh: {mesh}")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    case_path = output / "case.json"
    case = ensure_case(case_path)
    case_hash = digest(case_path)
    terminals = [item["id"] for item in case["conductors"][1:]]
    paths = [output / "base-order2.json", output / "refined-order2.json"]
    for mesh, path in zip((args.base_mesh, args.refined_mesh), paths):
        if args.recompute or not valid_result(path, case_hash, digest(mesh), terminals):
            run_solve(case_path, mesh, path, range(1, 9), args.threads)
    result = assess(record(paths[0]), record(paths[1]), *paths)
    target = output / "convergence.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {target}")
    print(f"Maximum matrix change: {result['maximum_absolute_change_fF']:.6f} fF")
    print(f"Maximum mutual change: {result['maximum_mutual_change_fF']:.6f} fF")
    if args.enriched:
        enriched_path = output / "refined-order4.json"
        if args.recompute or not valid_result(enriched_path, case_hash, digest(args.refined_mesh), terminals, order=4):
            run_enriched(case_path, args.refined_mesh, enriched_path, range(1, 9), args.threads)
        enriched = assess_enrichment(record(paths[1]), record(enriched_path), paths[1], enriched_path)
        enrichment_path = output / "enrichment.json"
        enrichment_path.write_text(json.dumps(enriched, indent=2) + "\n")
        print(f"Saved {enrichment_path}")
        print(f"Maximum order-2 to order-4 change: {enriched['maximum_absolute_change_fF']:.6f} fF")


if __name__ == "__main__":
    main()
