"""Exact box arrangement and Q1 weak charge measures for rectilinear terminals.

Metal terminals are unions of axis-aligned boxes. Later dielectric boxes take
precedence over earlier boxes. The arrangement retains all original faces and
lift support planes, so its cells are exact geometric boxes, not a voxelized
approximation. This module constructs the signed face measure of Eq. 5.67.
"""

from bisect import bisect_left
from dataclasses import dataclass
from itertools import product
from math import cos, pi, sin, sqrt
from random import Random

from .boxes import EPSILON_0

from .boxes import Box


@dataclass(frozen=True)
class MaterialBox:
    box: Box
    permittivity: float

    def __post_init__(self):
        if self.permittivity <= 0:
            raise ValueError("Material permittivity must be positive")


@dataclass(frozen=True)
class FaceComponent:
    axis: int
    coordinate: float
    bounds: tuple[tuple[float, float], tuple[float, float]]
    corner: tuple[int, int]
    mass: float
    prescribed: int | None = None
    outer: bool = False

    def sample(self, rng):
        """Sample the normalized nonnegative bilinear Bernstein component."""
        p = [0.0] * 3
        p[self.axis] = self.coordinate
        tangential = [k for k in range(3) if k != self.axis]
        for idx, k in enumerate(tangential):
            lo, hi = self.bounds[idx]
            u = sqrt(rng.random())
            if self.corner[idx] == 0:
                u = 1 - u
            p[k] = lo + (hi - lo) * u
        return tuple(p)


@dataclass(frozen=True)
class RectilinearProblem:
    domain: Box
    terminals: tuple[tuple[Box, ...], ...]
    background_permittivity: float
    materials: tuple[MaterialBox, ...] = ()
    support: float = 0.1
    length_unit: float = 1.0
    outer: str = "grounded"

    def __post_init__(self):
        if not self.terminals or any(not boxes for boxes in self.terminals):
            raise ValueError("Need at least one nonempty metal terminal")
        if self.background_permittivity <= 0 or self.support <= 0 or self.length_unit <= 0:
            raise ValueError("Permittivity, support, and length unit must be positive")
        if self.outer not in ("grounded", "reflecting", "open"):
            raise ValueError("Outer condition must be grounded, reflecting, or open")
        if self.outer == "open" and any(
                region.box.lo[k] < self.domain.lo[k] or
                region.box.hi[k] > self.domain.hi[k]
                for region in self.materials for k in range(3)):
            raise ValueError("Open exterior requires all materials inside the envelope")
        for i, boxes in enumerate(self.terminals):
            for box in boxes:
                if any(box.lo[k] <= self.domain.lo[k] or box.hi[k] >= self.domain.hi[k]
                       for k in range(3)):
                    raise ValueError("Metal must lie strictly inside the domain")
                if self.outer in ("grounded", "open") and any(
                        box.lo[k] - self.domain.lo[k] < self.support - 1e-12 or
                        self.domain.hi[k] - box.hi[k] < self.support - 1e-12
                        for k in range(3)):
                    raise ValueError("Grounded/open envelope requires zero lift trace")
                for other_boxes in self.terminals[:i]:
                    for other in other_boxes:
                        if all(box.lo[k] <= other.hi[k] and other.lo[k] <= box.hi[k]
                               for k in range(3)):
                            raise ValueError("Distinct metal terminals touch or overlap")

    def terminal(self, point):
        for i, boxes in enumerate(self.terminals):
            if any(box.contains(point) for box in boxes):
                return i
        return None

    def material_at(self, point):
        eps = self.background_permittivity
        for region in self.materials:
            if region.box.contains(point):
                eps = region.permittivity
        return eps

    @staticmethod
    def _surface_projection(box, point):
        if box.contains(point):
            distance, axis, coordinate = min(
                (abs(point[k] - face), k, face)
                for k in range(3) for face in (box.lo[k], box.hi[k]))
            q = list(point)
            q[axis] = coordinate
            return distance, tuple(q)
        q = tuple(min(max(point[k], box.lo[k]), box.hi[k]) for k in range(3))
        return sqrt(sum((point[k] - q[k]) ** 2 for k in range(3))), q

    @staticmethod
    def _direction(rng):
        normal = 2 * rng.random() - 1
        angle = 2 * pi * rng.random()
        tangent = sqrt(1 - normal * normal)
        return normal, tangent * cos(angle), tangent * sin(angle)

    def _fold(self, point):
        result = []
        for k in range(3):
            lo, hi = self.domain.lo[k], self.domain.hi[k]
            length = hi - lo
            offset = (point[k] - lo) % (2 * length)
            result.append(lo + (offset if offset <= length else 2 * length - offset))
        return tuple(result)

    def _exterior_return(self, point, rng):
        """Escape or sample the enclosing sphere's exterior harmonic measure."""
        center = tuple((self.domain.lo[k] + self.domain.hi[k]) / 2 for k in range(3))
        radius = sqrt(sum(((self.domain.hi[k] - self.domain.lo[k]) / 2) ** 2
                          for k in range(3)))
        relative = tuple(point[k] - center[k] for k in range(3))
        distance = sqrt(sum(v * v for v in relative))
        if distance <= radius:
            return point
        if rng.random() >= radius / distance:
            return None
        # Kelvin inversion turns the conditional exterior hitting law into
        # the interior sphere Poisson kernel at radial fraction a=R/|x|.
        a = radius / distance
        if a < 1e-10:
            cosine = 2 * rng.random() - 1
        else:
            inverse_root = 1 / (1 + a) + 2 * a * rng.random() / (1 - a * a)
            denominator = 1 / (inverse_root * inverse_root)
            cosine = max(-1.0, min(1.0, (1 + a * a - denominator) / (2 * a)))
        direction = tuple(v / distance for v in relative)
        helper = (0.0, 0.0, 1.0) if abs(direction[2]) < .9 else (0.0, 1.0, 0.0)
        cross = (direction[1] * helper[2] - direction[2] * helper[1],
                 direction[2] * helper[0] - direction[0] * helper[2],
                 direction[0] * helper[1] - direction[1] * helper[0])
        cross_norm = sqrt(sum(v * v for v in cross))
        tangent1 = tuple(v / cross_norm for v in cross)
        tangent2 = (direction[1] * tangent1[2] - direction[2] * tangent1[1],
                    direction[2] * tangent1[0] - direction[0] * tangent1[2],
                    direction[0] * tangent1[1] - direction[1] * tangent1[0])
        angle = 2 * pi * rng.random()
        sine = sqrt(max(0.0, 1 - cosine * cosine))
        return tuple(center[k] + radius * (cosine * direction[k] +
                     sine * (cos(angle) * tangent1[k] + sin(angle) * tangent2[k]))
                     for k in range(3))

    def terminal_hit(self, point, rng, tolerance=1e-4, max_steps=100000):
        """Trace homogeneous spheres and weighted material-junction spheres."""
        if tolerance <= 0 or max_steps < 1:
            raise ValueError("Tolerance and maximum walk length must be positive")
        p = tuple(point)
        metal = tuple(box for boxes in self.terminals for box in boxes)
        material = tuple(region.box for region in self.materials)
        eps_max = max((self.background_permittivity,
                       *(region.permittivity for region in self.materials)))
        for _ in range(max_steps):
            if self.outer == "open":
                p = self._exterior_return(p, rng)
                if p is None:
                    return None
            for i, boxes in enumerate(self.terminals):
                if min(box.distance(p) for box in boxes) <= tolerance:
                    return i
            wall_distance = min(min(p[k] - self.domain.lo[k],
                                    self.domain.hi[k] - p[k]) for k in range(3))
            if self.outer == "grounded" and wall_distance <= tolerance:
                return None
            candidates = [self._surface_projection(box, p) for box in material]
            distance, q = min(candidates, default=(float("inf"), None))
            radius = None
            if distance <= tolerance:
                limits = [box.distance(q) for box in metal]
                limits += [q[k] - self.domain.lo[k] for k in range(3)]
                limits += [self.domain.hi[k] - q[k] for k in range(3)]
                limits += [abs(q[k] - face) for box in material for k in range(3)
                           for face in (box.lo[k], box.hi[k]) if abs(q[k] - face) > 1e-12]
                radius = 0.5 * min(limits)
            if radius is not None and radius > 0 and distance <= .01 * radius:
                while True:
                    direction = self._direction(rng)
                    endpoint = tuple(q[k] + radius * direction[k] for k in range(3))
                    if rng.random() * eps_max < self.material_at(endpoint):
                        p = endpoint
                        break
            else:
                radius = min(box.distance(p) for box in metal)
                if candidates:
                    radius = min(radius, distance)
                if self.outer == "grounded":
                    radius = min(radius, wall_distance)
                direction = self._direction(rng)
                endpoint = tuple(p[k] + radius * direction[k] for k in range(3))
                p = self._fold(endpoint) if self.outer == "reflecting" else endpoint
        raise RuntimeError("Walk exceeded max_steps")

    def arrangement(self):
        axes = []
        for k in range(3):
            coords = {self.domain.lo[k], self.domain.hi[k]}
            for boxes in self.terminals:
                for box in boxes:
                    for face in (box.lo[k], box.hi[k]):
                        coords.add(face)
                        coords.add(max(self.domain.lo[k], face - self.support))
                        coords.add(min(self.domain.hi[k], face + self.support))
            for region in self.materials:
                coords.update(face for face in (region.box.lo[k], region.box.hi[k])
                              if self.domain.lo[k] < face < self.domain.hi[k])
            ordered = sorted(coords)
            merged = [ordered[0]]
            threshold = 1e-10 * (self.domain.hi[k] - self.domain.lo[k])
            for coordinate in ordered[1:]:
                if coordinate - merged[-1] > threshold:
                    merged.append(coordinate)
            axes.append(tuple(merged))
        return tuple(axes)

    def _distance_to_terminal(self, point, terminal):
        return min(max(max(box.lo[k] - point[k], 0.0, point[k] - box.hi[k])
                       for k in range(3)) for box in self.terminals[terminal])

    def lift_value(self, point, terminal):
        di = self._distance_to_terminal(point, terminal)
        if len(self.terminals) == 1:
            return max(0.0, 1 - di / self.support)
        do = min(self._distance_to_terminal(point, other)
                 for other in range(len(self.terminals)) if other != terminal)
        return max(0.0, 1 - di / self.support) * do / (di + do)

    def charge_measure(self, terminal):
        """Return Bernstein components and exact constant-annihilation audit.

        The audit includes outer-face trace even for grounded boundaries; its
        potential contribution is zero for a grounded outer wall. Components
        on metal faces are labeled prescribed and can be integrated exactly.
        """
        if not 0 <= terminal < len(self.terminals):
            raise ValueError("Invalid terminal")
        axes = self.arrangement()
        shape = tuple(len(axis) - 1 for axis in axes)
        nodes = {ijk: self.lift_value(tuple(axes[k][ijk[k]] for k in range(3)), terminal)
                 for ijk in product(*(range(len(axis)) for axis in axes))}
        cell_eps = {}
        cell_metal = {}
        for ijk in product(*(range(n) for n in shape)):
            center = tuple((axes[k][ijk[k]] + axes[k][ijk[k] + 1]) / 2 for k in range(3))
            cell_eps[ijk] = self.material_at(center)
            cell_metal[ijk] = self.terminal(center)
        components = []
        signed_total = 0.0
        for axis in range(3):
            tangential = [k for k in range(3) if k != axis]
            for face in range(shape[axis] + 1):
                for t0 in range(shape[tangential[0]]):
                    for t1 in range(shape[tangential[1]]):
                        base = [0, 0, 0]
                        base[axis] = face
                        base[tangential[0]] = t0
                        base[tangential[1]] = t1
                        low_cell = tuple(face - 1 if k == axis else base[k] for k in range(3)) if face else None
                        high_cell = tuple(face if k == axis else base[k] for k in range(3)) if face < shape[axis] else None
                        bounds = ((axes[tangential[0]][t0], axes[tangential[0]][t0 + 1]),
                                  (axes[tangential[1]][t1], axes[tangential[1]][t1 + 1]))
                        area = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
                        prescribed = (cell_metal[low_cell] if low_cell is not None and cell_metal[low_cell] is not None
                                      else cell_metal[high_cell] if high_cell is not None else None)
                        for corner in product((0, 1), repeat=2):
                            index = base.copy()
                            index[tangential[0]] += corner[0]
                            index[tangential[1]] += corner[1]

                            def derivative(cell):
                                if cell is None:
                                    return 0.0
                                lower = index.copy()
                                lower[axis] = cell[axis]
                                upper = lower.copy()
                                upper[axis] += 1
                                return ((nodes[tuple(upper)] - nodes[tuple(lower)]) /
                                        (axes[axis][upper[axis]] - axes[axis][lower[axis]]))

                            if low_cell is None:
                                coefficient = -cell_eps[high_cell] * derivative(high_cell)
                            elif high_cell is None:
                                coefficient = cell_eps[low_cell] * derivative(low_cell)
                            else:
                                coefficient = (cell_eps[low_cell] * derivative(low_cell) -
                                               cell_eps[high_cell] * derivative(high_cell))
                            mass = coefficient * area / 4
                            signed_total += mass
                            if abs(mass) > 1e-15:
                                components.append(FaceComponent(axis, axes[axis][face], bounds,
                                                                corner, mass, prescribed,
                                                                low_cell is None or high_cell is None))
        return components, signed_total


def estimate_rectilinear_capacitance(problem, samples_per_row=1000,
                                      tolerance=1e-4, seed=1,
                                      max_steps=100000):
    """Estimate a raw physical C matrix from signed Q1 Bernstein face measures.

    Sampling is global per row; the full row covariance retains correlations
    between terminal indicators. Metal face terms are integrated exactly.
    """
    if samples_per_row < 2:
        raise ValueError("Need at least two samples per charge row")
    rng = Random(seed)
    n = len(problem.terminals)
    scale = EPSILON_0 * problem.length_unit
    matrix = [[0.0] * n for _ in range(n)]
    covariance = [[[0.0] * n for _ in range(n)] for _ in range(n)]
    diagnostics = []
    for i in range(n):
        components, audit = problem.charge_measure(i)
        if abs(audit) > 1e-8 * max(1.0, sum(abs(c.mass) for c in components)):
            raise RuntimeError("Charge measure fails constant-potential audit")
        free = []
        cumulative = []
        total = 0.0
        for component in components:
            if component.prescribed is not None:
                matrix[i][component.prescribed] += component.mass
            elif component.outer and problem.outer in ("grounded", "open"):
                continue
            else:
                total += abs(component.mass)
                cumulative.append(total)
                free.append(component)
        diagnostics.append({"signed_audit": audit, "free_absolute_mass": total,
                            "component_count": len(components)})
        if not free:
            continue
        sample_sums = [0.0] * n
        sample_products = [[0.0] * n for _ in range(n)]
        for _ in range(samples_per_row):
            component = free[bisect_left(cumulative, rng.random() * total)]
            point = component.sample(rng)
            hit = problem.terminal_hit(point, rng, tolerance, max_steps)
            if hit is not None:
                value = total if component.mass > 0 else -total
                sample_sums[hit] += value
                sample_products[hit][hit] += value * value
        for j in range(n):
            mean = sample_sums[j] / samples_per_row
            matrix[i][j] += mean
            for k in range(n):
                covariance[i][j][k] = ((sample_products[j][k] / samples_per_row -
                                         mean * sample_sums[k] / samples_per_row) /
                                        (samples_per_row - 1))
    return ([[value * scale for value in row] for row in matrix],
            [[[value * scale * scale for value in row] for row in term]
             for term in covariance], diagnostics)
