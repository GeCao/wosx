"""Separate Cartesian finite-volume check for rectilinear capacitance cases.

This grid reference is deliberately separate from the random-walk estimator.
It approximates the same box geometry only when all material and metal faces
align with cell faces. Compare at two grid resolutions to assess grid error.
"""

from .boxes import EPSILON_0


def finite_volume_capacitance(problem, cells=40):
    try:
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import cg
    except ImportError as exc:
        raise RuntimeError("Finite-volume reference needs NumPy and SciPy") from exc
    if getattr(problem, "outer", "grounded") == "open":
        raise ValueError("Finite-box grid reference cannot represent open infinity")
    if cells < 4:
        raise ValueError("Need at least four cells per axis")
    span = [problem.domain.hi[k] - problem.domain.lo[k] for k in range(3)]
    if max(span) - min(span) > 1e-12:
        raise ValueError("Reference currently requires a cubic domain")
    h = span[0] / cells
    interfaces = getattr(problem, "interfaces", ())
    inclusion = getattr(problem, "inclusion", None)
    materials = getattr(problem, "materials", ())
    terminal_boxes = tuple(box for terminal in problem.terminals
                           for box in (terminal if isinstance(terminal, tuple) else (terminal,)))
    material_boxes = tuple(region.box for region in materials)
    if inclusion is not None:
        material_boxes += (inclusion,)
    for face in interfaces:
        grid_position = (face - problem.domain.lo[0]) / h
        if abs(grid_position - round(grid_position)) > 1e-8:
            raise ValueError("Material planes must align with grid cells")
    for box in (*terminal_boxes, *material_boxes):
        for axis in range(3):
            for face in (box.lo[axis], box.hi[axis]):
                grid_position = (face - problem.domain.lo[axis]) / h
                if abs(grid_position - round(grid_position)) > 1e-8:
                    raise ValueError("Metal and inclusion faces must align with grid cells")
    coords = [problem.domain.lo[k] + (np.arange(cells) + 0.5) * h for k in range(3)]
    label = np.full((cells, cells, cells), -1, dtype=int)
    epsilon = np.empty_like(label, dtype=float)
    if hasattr(problem, "background_permittivity"):
        epsilon.fill(problem.background_permittivity)
    else:
        for ix, x in enumerate(coords[0]):
            epsilon[ix, :, :] = problem.permittivities[problem._layer(x)]
    regions = [(region.box, region.permittivity) for region in materials]
    if inclusion is not None:
        regions.append((inclusion, problem.inclusion_permittivity))
    for box, permittivity in regions:
        inside = np.ones_like(label, dtype=bool)
        for k in range(3):
            shape = [1, 1, 1]
            shape[k] = cells
            inside &= ((coords[k] >= box.lo[k]) & (coords[k] <= box.hi[k])).reshape(shape)
        epsilon[inside] = permittivity
    for terminal, boxes in enumerate(problem.terminals):
        boxes = boxes if isinstance(boxes, tuple) else (boxes,)
        terminal_mask = np.zeros_like(label, dtype=bool)
        for box in boxes:
            inside = np.ones_like(label, dtype=bool)
            for k in range(3):
                shape = [1, 1, 1]
                shape[k] = cells
                inside &= ((coords[k] >= box.lo[k]) & (coords[k] <= box.hi[k])).reshape(shape)
            terminal_mask |= inside
        label[terminal_mask] = terminal
    unknown = np.full_like(label, -1)
    unknown[label < 0] = np.arange(np.count_nonzero(label < 0))
    size = int(np.count_nonzero(label < 0))
    rows, cols, data = [], [], []
    diagonal = np.zeros(size)
    rhs = np.zeros((size, len(problem.terminals)))
    contacts = [[] for _ in problem.terminals]
    for ijk in np.ndindex(label.shape):
        a = unknown[ijk]
        if a < 0:
            continue
        eps_a = epsilon[ijk]
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = list(ijk)
                neighbor[axis] += direction
                if neighbor[axis] < 0 or neighbor[axis] >= cells:
                    if getattr(problem, "outer", "grounded") == "grounded":
                        diagonal[a] += 2 * eps_a * h
                    continue
                neighbor = tuple(neighbor)
                terminal = label[neighbor]
                if terminal >= 0:
                    g = 2 * eps_a * h
                    diagonal[a] += g
                    rhs[a, terminal] += g
                    contacts[terminal].append((a, g))
                else:
                    eps_b = epsilon[neighbor]
                    g = 2 * eps_a * eps_b / (eps_a + eps_b) * h
                    diagonal[a] += g
                    rows.append(a)
                    cols.append(unknown[neighbor])
                    data.append(-g)
    rows.extend(range(size))
    cols.extend(range(size))
    data.extend(diagonal)
    matrix = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    response = np.zeros((len(problem.terminals), len(problem.terminals)))
    from inspect import signature
    tolerance_keyword = "rtol" if "rtol" in signature(cg).parameters else "tol"
    for j in range(len(problem.terminals)):
        solution, code = cg(matrix, rhs[:, j], atol=0, maxiter=10000,
                            **{tolerance_keyword: 1e-10})
        if code != 0:
            raise RuntimeError(f"Finite-volume CG did not converge: {code}")
        for i, adjacent in enumerate(contacts):
            response[i, j] = sum(g * ((1.0 if i == j else 0.0) - solution[a])
                                 for a, g in adjacent)
    return (response * EPSILON_0 * problem.length_unit).tolist()
