"""Estimate a case-8 terminal potential slice from WoSX GPU terminal hits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .full_case8 import make_case
from .gpu_wos import ROOT, implementation_sha256
from heterogeneous_wos.gpu_rectilinear import prepare_geometry, walk_batch
from heterogeneous_wos.rectilinear_case import RectilinearCase


def sample_slice(output, nx=80, nz=64, paths_per_pixel=128, seed=73193,
                 tolerance=5e-5, continuations=200, restarts=5):
    """Return terminal-1 hit frequencies on the x-z section at y=-0.5."""
    if min(nx, nz) < 2 or paths_per_pixel < 2 or tolerance <= 0:
        raise ValueError("Need a grid of at least 2x2, two paths per pixel, and positive tolerance")
    executable = ROOT / "build/demo_apps/heterogeneous_rectilinear_walk_gpu"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    metadata = make_case()
    case = RectilinearCase(metadata)
    x = np.linspace(-4.1, 4.1, nx)
    z = np.linspace(-3.0, 3.2, nz)
    xx, zz = np.meshgrid(x, z)
    pixels = np.column_stack((xx.ravel(), np.full(xx.size, -0.5), zz.ravel()))
    metal = np.full(len(pixels), -1, dtype=np.int32)
    for box, tag in zip(case.boxes, case.tags):
        inside = np.all((pixels >= box[0]) & (pixels <= box[1]), axis=1)
        metal[inside] = tag
    dielectric = np.flatnonzero(metal < 0)
    starts = np.repeat(pixels[dielectric], paths_per_pixel, axis=0).astype(np.float32)
    if len(starts) >= 2**24:
        raise ValueError("Potential campaign exceeds the exact float32 path-ID range")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry = prepare_geometry(case)
    labels, endpoints = walk_batch(executable, case, geometry, starts,
                                   np.arange(len(starts)), 1, output.parent, 10000,
                                   continuations, tolerance, tolerance, seed)
    labels = labels[:, 0].astype(np.int32)
    pending = np.flatnonzero(labels < 0)
    for restart in range(1, restarts + 1):
        if not len(pending):
            break
        retried, next_endpoints = walk_batch(executable, case, geometry,
            endpoints[pending], pending, 1, output.parent, 10000,
            continuations, tolerance, tolerance, seed + restart * 1000003)
        labels[pending] = retried[:, 0].astype(np.int32)
        endpoints[pending] = next_endpoints
        pending = np.flatnonzero(labels < 0)
    if len(pending):
        raise RuntimeError(f"{len(pending)} potential walks did not reach a terminal")
    hits = (labels == 1).reshape(len(dielectric), paths_per_pixel)
    mean = np.empty(len(pixels))
    standard_error = np.zeros(len(pixels))
    mean[metal >= 0] = (metal[metal >= 0] == 1).astype(float)
    mean[dielectric] = hits.mean(axis=1)
    p = mean[dielectric]
    standard_error[dielectric] = np.sqrt(p * (1 - p) / (paths_per_pixel - 1))
    case_hash = hashlib.sha256((json.dumps(metadata, indent=2) + "\n").encode()).hexdigest()
    np.savez_compressed(output, x=x, z=z, y=-0.5, terminal=1,
                        potential=mean.reshape(nz, nx),
                        standard_error=standard_error.reshape(nz, nx),
                        metal_mask=(metal >= 0).reshape(nz, nx),
                        paths_per_pixel=paths_per_pixel, seed=seed,
                        tolerance=tolerance, continuations=continuations,
                        case_sha256=case_hash,
                        gpu_implementation_sha256=implementation_sha256(executable))
    print(f"Saved {output}: {len(dielectric)} dielectric pixels x {paths_per_pixel} walks; "
          f"max pointwise SE {standard_error.max():.4f} V", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, nargs=2, default=(80, 64), metavar=("NX", "NZ"))
    parser.add_argument("--paths-per-pixel", type=int, default=128)
    parser.add_argument("--seed", type=int, default=73193)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    parser.add_argument("--continuations", type=int, default=200)
    args = parser.parse_args()
    sample_slice(args.output, *args.grid, args.paths_per_pixel,
                 args.seed, args.tolerance, args.continuations)


if __name__ == "__main__":
    main()
