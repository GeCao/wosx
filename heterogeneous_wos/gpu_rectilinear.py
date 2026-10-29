"""WoSX GPU binary adapter for exact rectilinear geometry and path batches."""

from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def float4(values):
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros((len(values), 4), dtype=np.float32)
    result[:, :values.shape[1]] = values
    return result


def prepare_geometry(case):
    low = float4(case.boxes[:, 0])
    low[:, 3] = case.tags
    return (low, float4(case.boxes[:, 1]), float4(case.interfaces[:, :4]),
            float4(case.interfaces[:, 4:6]),
            *(np.asarray(axis, dtype=np.float32) for axis in case.coordinates),
            case.eps.astype(np.float32))


def walk_batch(executable, case, geometry, points, path_ids, row, output_dir,
               max_steps, continuations, tolerance, interface_tolerance, seed):
    low, high, patch_first, patch_second, axis_x, axis_y, axis_z, epsilon = geometry
    fixture = output_dir / f"row{row:02d}-input.bin"
    output = output_dir / f"row{row:02d}-output.bin"
    starts = float4(points)
    starts[:, 3] = path_ids
    with fixture.open("wb") as stream:
        nx, ny, nz = case.eps.shape
        np.array([len(low), len(patch_first), len(points), nx, ny, nz, 0, 0],
                 dtype=np.uint32).tofile(stream)
        for array in (low, high, patch_first, patch_second, starts,
                      axis_x, axis_y, axis_z, epsilon,
                      np.zeros((len(points), 4), dtype=np.float32),
                      np.zeros((len(points), 4), dtype=np.float32)):
            array.tofile(stream)
    command = [str(executable), str(ROOT), str(fixture), str(output),
               str(max_steps), str(seed + 10007 * row),
               str(tolerance), str(interface_tolerance), str(continuations)]
    subprocess.run(command, check=True)
    with output.open("rb") as stream:
        count = int(np.fromfile(stream, dtype=np.uint32, count=1)[0])
        results = np.fromfile(stream, dtype=np.float32, count=count * 4).reshape(count, 4)
        endpoints = np.fromfile(stream, dtype=np.float32, count=count * 4).reshape(count, 4)
    if count != len(points) or not np.isfinite(results).all() or not np.isfinite(endpoints).all():
        raise RuntimeError("Invalid GPU walk result")
    fixture.unlink()
    output.unlink()
    return results, endpoints[:, :3]


