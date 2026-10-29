"""GPU WoS potential around any PLY/OBJ conductor in a two-layer dielectric."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "build/demo_apps/heterogeneous_triangle_mesh_walk_gpu"
SHADER = ROOT / "demo_apps/heterogeneous_layers/triangle_mesh_walk_gpu.slang"


def mesh_data(mesh):
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "mesh.bin"
        subprocess.run([str(RUNNER), str(mesh), "--dump", str(output)], check=True)
        with output.open("rb") as stream:
            nv, nf = np.fromfile(stream, np.uint32, 2)
            vertices = np.fromfile(stream, np.float32, int(nv) * 3).reshape(-1, 3)
            faces = np.fromfile(stream, np.uint32, int(nf) * 3).reshape(-1, 3)
    if len(vertices) != nv or len(faces) != nf:
        raise ValueError("Incomplete mesh export")
    return vertices, faces


def sample(mesh, output, nx, nz, paths, eps_low, eps_high, interface_z,
           padding, tolerance, seed, max_steps, continuations, reuse):
    vertices, faces = mesh_data(mesh)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    span = float(max(high - low))
    margin = min(.25, .65 * padding) * span
    y_slice = float(low[1] - min(.08, .5 * padding) * span)
    x = np.linspace(low[0] - margin, high[0] + margin, nx)
    z = np.linspace(low[2] - margin, high[2] + margin, nz)
    if padding <= 0 or min(eps_low, eps_high, tolerance) <= 0:
        raise ValueError("Padding, permittivities, and tolerance must be positive")
    if not low[2] - padding * span < interface_z < high[2] + padding * span:
        raise ValueError("Interface must lie inside the grounded outer box")
    if nx < 2 or nz < 2 or paths < 2 or nx * nz * paths >= 2**24:
        raise ValueError("Need a grid of at least 2x2 and 2 paths/pixel, below 2^24 paths")
    digest = hashlib.sha256()
    for path in (mesh, RUNNER, SHADER):
        digest.update(path.read_bytes())
    settings = dict(mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
                    implementation_sha256=digest.hexdigest(), nx=nx, nz=nz,
                    paths_per_pixel=paths, eps_low=eps_low, eps_high=eps_high,
                    interface_z=interface_z, padding=padding, slice_y=y_slice, tolerance=tolerance,
                    seed=seed, max_steps=max_steps, continuations=continuations)
    cache = output / "potential.npz"
    cached = False
    if reuse and cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            cached = str(data["settings"]) == json.dumps(settings, sort_keys=True)
            if cached:
                potential = data["potential"]
                standard_error = data["standard_error"]
                steps = data["mean_steps"]
    if not cached:
        xx, zz = np.meshgrid(x, z)
        points = np.column_stack((xx.ravel(), np.full(xx.size, y_slice), zz.ravel()))
        starts = np.repeat(points, paths, axis=0)
        starts = np.column_stack((starts, np.arange(len(starts)))).astype(np.float32)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            fixture, result = Path(temporary) / "starts.bin", Path(temporary) / "results.bin"
            with fixture.open("wb") as stream:
                np.asarray([len(starts)], np.uint32).tofile(stream)
                starts.tofile(stream)
            command = [str(RUNNER), str(ROOT), str(mesh), str(fixture), str(result),
                       str(eps_low), str(eps_high), str(interface_z), str(padding),
                       str(max_steps), str(seed), str(tolerance), str(continuations)]
            subprocess.run(command, cwd=ROOT, check=True)
            with result.open("rb") as stream:
                count = int(np.fromfile(stream, np.uint32, 1)[0])
                records = np.fromfile(stream, np.float32, count * 4).reshape(-1, 4)
            if count != len(starts) or not np.isin(records[:, 0], [0, 1]).all():
                raise RuntimeError("Some GPU walks did not reach mesh or ground; increase --continuations")
        hits = (records[:, 0] == 1).reshape(nz, nx, paths)
        potential = hits.mean(axis=2)
        standard_error = np.sqrt(potential * (1 - potential) / (paths - 1))
        steps = records[:, 1].reshape(nz, nx, paths).mean(axis=2)
        np.savez_compressed(cache, x=x, z=z, y=y_slice, potential=potential,
                            standard_error=standard_error, mean_steps=steps,
                            settings=json.dumps(settings, sort_keys=True))
    return vertices, faces, x, z, y_slice, potential, standard_error, steps, settings, cached


def figure(output, mesh, vertices, faces, x, z, y_slice, potential,
           standard_error, settings):
    fig = plt.figure(figsize=(13.8, 5.9), layout="constrained", facecolor="#f7f9fc")
    left = fig.add_subplot(121, projection="3d")
    right = fig.add_subplot(122)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    size = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(size[:, None], 1e-12)
    shade = np.clip(.48 + .42 * normals @ np.array([.35, -.25, .9]), .12, .95)
    colors = plt.cm.viridis(shade)
    left.add_collection3d(Poly3DCollection(triangles, facecolors=colors, linewidths=0))
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    left.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
             xlabel="x", ylabel="y", zlabel="z", title=f"Triangle conductor: {mesh.name}\n1 V")
    left.set_box_aspect(high - low)
    left.view_init(elev=22, azim=-65)
    image = right.imshow(potential, extent=(x[0], x[-1], z[0], z[-1]),
                         origin="lower", cmap="magma", vmin=0, vmax=1, aspect="equal")
    right.axhline(settings["interface_z"], color="#56e8e1", lw=2, ls="--",
                  label="dielectric interface")
    right.set(xlabel="x", ylabel="z", title=f"WoSX GPU potential at y={y_slice:.3g}\n"
              f"{settings['paths_per_pixel']} walks/pixel · max SE {standard_error.max():.3f} V")
    right.legend(loc="upper right", facecolor="#172133", labelcolor="white")
    fig.colorbar(image, ax=right, label="Potential (V)", shrink=.85)
    fig.suptitle(f"Triangle-mesh heterogeneous WoS  |  εr={settings['eps_low']:g} / {settings['eps_high']:g}",
                 fontsize=16, fontweight="bold")
    fig.savefig(output / "potential.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, nargs="?", default=ROOT / "assets/spot.ply")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: data/triangle_mesh/<mesh stem>")
    parser.add_argument("--grid", type=int, nargs=2, default=(64, 64), metavar=("NX", "NZ"))
    parser.add_argument("--paths", type=int, default=48)
    parser.add_argument("--eps-low", type=float, default=2)
    parser.add_argument("--eps-high", type=float, default=8)
    parser.add_argument("--interface-z", type=float)
    parser.add_argument("--padding", type=float, default=.5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--continuations", type=int, default=30)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if not RUNNER.is_file():
        parser.error("Build heterogeneous_triangle_mesh_walk_gpu first")
    if not args.mesh.is_file():
        parser.error(f"Mesh not found: {args.mesh}")
    if args.output_dir is None:
        args.output_dir = ROOT / "data/triangle_mesh" / args.mesh.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vertices, _ = mesh_data(args.mesh)
    interface_z = float((vertices[:, 2].min() + vertices[:, 2].max()) / 2
                        if args.interface_z is None else args.interface_z)
    result = sample(args.mesh, args.output_dir, *args.grid, args.paths,
                    args.eps_low, args.eps_high, interface_z, args.padding,
                    args.tolerance, args.seed, args.max_steps,
                    args.continuations, args.reuse)
    vertices, faces, x, z, y_slice, potential, standard_error, steps, settings, cached = result
    figure(args.output_dir, args.mesh, vertices, faces, x, z, y_slice,
           potential, standard_error, settings)
    report = dict(mesh=str(args.mesh), vertices=len(vertices), triangles=len(faces),
                  settings=settings, slice_y=y_slice, paths=int(potential.size * args.paths),
                  potential_min_V=float(potential.min()), potential_max_V=float(potential.max()),
                  mean_steps=float(steps.mean()), max_pointwise_SE_V=float(standard_error.max()),
                  reused=bool(cached), figure=str(args.output_dir / "potential.png"))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
