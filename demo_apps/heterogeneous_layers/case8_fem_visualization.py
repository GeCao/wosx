"""Solve and visualize case-8 geometry, FEM potential, and capacitance."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
import ngsolve as ng
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from demo_apps.heterogeneous_layers.case8_support.fem_fullcase_iterative import make_space
from demo_apps.heterogeneous_layers.case8_fem_convergence import (
    BASE_MESH, DEFAULT_OUTPUT, digest, ensure_case, record, run_solve, valid_result,
)
COLORS = plt.get_cmap("tab10").colors


def intersects(box, axis, coordinate):
    return box[0][axis] <= coordinate <= box[1][axis]


def add_metal_rectangles(ax, case, normal_axis, coordinate, axes, outline=False):
    for terminal, conductor in enumerate(case["conductors"]):
        for box in conductor["boxes"]:
            if not intersects(box, normal_axis, coordinate):
                continue
            lo, hi = np.asarray(box)
            x, y = lo[axes[0]], lo[axes[1]]
            width, height = hi[axes[0]] - x, hi[axes[1]] - y
            ax.add_patch(Rectangle((x, y), width, height,
                facecolor="none" if outline else COLORS[terminal % len(COLORS)],
                edgecolor="white" if outline else "#142235", linewidth=0.8,
                alpha=1 if outline else 0.9, zorder=3))
    return ax


def potential_slice(mesh, field, case, y, nx, nz):
    xs = np.linspace(-4.1, 4.1, nx)
    zs = np.linspace(-3.0, 3.2, nz)
    values = np.full((nz, nx), np.nan)
    boxes = [np.asarray(box) for conductor in case["conductors"] for box in conductor["boxes"]]
    for row, z in enumerate(zs):
        for col, x in enumerate(xs):
            point = np.array([x, y, z])
            if any(np.all(point >= box[0]) and np.all(point <= box[1]) for box in boxes):
                continue
            try:
                value = float(field(mesh(float(x), float(y), float(z))))
            except Exception:
                continue
            if np.isfinite(value):
                values[row, col] = value
    return xs, zs, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mesh", type=Path, default=BASE_MESH)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--recompute", action="store_true", help="Rerun matrix and field solves")
    parser.add_argument("--grid", type=int, nargs=2, default=(220, 170), metavar=("NX", "NZ"))
    args = parser.parse_args()
    if not args.mesh.is_file() or args.threads < 1 or min(args.grid) < 10:
        parser.error("Need an existing FEM mesh, positive thread count, and grid dimensions >= 10")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    case_path = output / "case.json"
    case = ensure_case(case_path)
    case_hash, mesh_hash = digest(case_path), digest(args.mesh)
    matrix_path = output / "base-order2.json"
    field_report = output / "terminal1-order2.json"
    field_path = output / "terminal-1.sol"
    args.output = output / "overview.png"
    full_labels = [c["id"] for c in case["conductors"][1:]]
    field_labels = full_labels[:1]
    if args.recompute or not valid_result(matrix_path, case_hash, mesh_hash, full_labels):
        run_solve(case_path, args.mesh, matrix_path, range(1, 9), args.threads)
    if args.recompute or not field_path.is_file() or not valid_result(field_report, case_hash, mesh_hash, field_labels):
        run_solve(case_path, args.mesh, field_report, [1], args.threads, output)
    result = record(matrix_path)
    field_result = record(field_report)
    if abs(field_result["C_fF"][0][0] - result["C_fF"][0][0]) > 1e-8:
        raise ValueError("Saved potential field does not match the order-2 FEM matrix")
    mesh = ng.Mesh(str(args.mesh))
    space, _ = make_space(mesh, 2, [f"terminal_{i}" for i in range(9)])
    field = ng.GridFunction(space)
    field.Load(str(field_path))
    xs, zs, voltage = potential_slice(mesh, field, case, -0.5, *args.grid)
    if np.count_nonzero(np.isfinite(voltage)) < voltage.size // 2:
        raise RuntimeError("Too few FEM slice samples were evaluated")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10})
    fig = plt.figure(figsize=(17, 6.5), layout="constrained", facecolor="#f5f7fb")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.2, 1.0])
    ax_geo, ax_u, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]
    fig.suptitle("Case 8  |  order-2 FEM electrostatics", fontsize=18, fontweight="bold")

    ax_geo.set_facecolor("#e8eff7")
    add_metal_rectangles(ax_geo, case, 2, -0.9, (0, 1))
    for terminal, conductor in enumerate(case["conductors"][1:8], 1):
        boxes = [np.asarray(b) for b in conductor["boxes"] if intersects(b, 2, -0.9)]
        if boxes:
            center = np.mean([0.5 * (box[0, :2] + box[1, :2]) for box in boxes], axis=0)
            ax_geo.text(center[0], center[1], str(terminal), color="white", ha="center", va="center",
                        weight="bold", fontsize=9, bbox=dict(boxstyle="circle,pad=0.18", fc="#172036", ec="none", alpha=0.85), zorder=5)
    ax_geo.set(xlim=(-4.1, 4.1), ylim=(-4.0, 3.4), xlabel="x (0.1 µm units)", ylabel="y (0.1 µm units)")
    ax_geo.set_aspect("equal")
    ax_geo.set_title("Conductor geometry  ·  z = −0.9")
    ax_geo.grid(alpha=0.15)

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#e8eff7")
    im = ax_u.imshow(voltage, origin="lower", extent=(xs[0], xs[-1], zs[0], zs[-1]),
                     cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    add_metal_rectangles(ax_u, case, 1, -0.5, (0, 2), outline=True)
    ax_u.set(xlabel="x (0.1 µm units)", ylabel="z (0.1 µm units)")
    ax_u.set_title("FEM potential  ·  y = −0.5  ·  terminal 1 at 1 V")
    fig.colorbar(im, ax=ax_u, shrink=0.82, label="Electric potential (V)")

    C = np.asarray(result["C_fF"])
    labels = result["terminals"]
    image = ax_c.imshow(C, cmap="coolwarm", norm=SymLogNorm(linthresh=0.005, vmin=-0.5, vmax=2.5))
    ax_c.set_xticks(range(8), labels, rotation=55, ha="right")
    ax_c.set_yticks(range(8), labels)
    ax_c.set(xlabel="Excited terminal", ylabel="Measured terminal")
    ax_c.set_title("Capacitance matrix  ·  grounded reference")
    for i in range(8):
        ax_c.text(i, i, f"{C[i, i]:.2f}", ha="center", va="center", fontsize=8, weight="bold", color="white" if C[i, i] > 0.5 else "black")
    fig.colorbar(image, ax=ax_c, shrink=0.82, label="Capacitance (fF)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor=fig.get_facecolor())
    print(f"Saved {args.output}; finite FEM samples: {np.count_nonzero(np.isfinite(voltage))}/{voltage.size}")


if __name__ == "__main__":
    main()
