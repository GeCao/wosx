"""3D capacitance extraction for separated boxes and rectilinear dielectrics.

Implements the Chapter 5 weighted planar-interface exit and weak lift identity.
The physical matrix is epsilon_0 * length_unit times the dimensionless result.
"""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from random import Random

EPSILON_0 = 8.8541878128e-12


@dataclass(frozen=True)
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def __post_init__(self):
        if len(self.lo) != 3 or len(self.hi) != 3 or any(a >= b for a, b in zip(self.lo, self.hi)):
            raise ValueError("Box requires three ordered coordinate intervals")

    def distance(self, p):
        return sqrt(sum(max(a - x, 0.0, x - b) ** 2 for x, a, b in zip(p, self.lo, self.hi)))

    def contains(self, p):
        return all(a <= x <= b for x, a, b in zip(p, self.lo, self.hi))


@dataclass(frozen=True)
class BoxCapacitanceProblem:
    domain: Box
    terminals: tuple[Box, ...]
    interfaces: tuple[float, ...]
    permittivities: tuple[float, ...]
    collar: float
    length_unit: float = 1.0
    inclusion: Box | None = None
    inclusion_permittivity: float | None = None

    def __post_init__(self):
        if not self.terminals or self.collar <= 0 or self.length_unit <= 0:
            raise ValueError("Need terminals and positive collar/length unit")
        if len(self.permittivities) != len(self.interfaces) + 1 or any(e <= 0 for e in self.permittivities):
            raise ValueError("Need one positive permittivity per layer")
        cuts = (self.domain.lo[0], *self.interfaces, self.domain.hi[0])
        if any(a >= b for a, b in zip(cuts, cuts[1:])):
            raise ValueError("Material planes must be ordered inside the domain")
        if (self.inclusion is None) != (self.inclusion_permittivity is None):
            raise ValueError("Inclusion and its permittivity must be specified together")
        if self.inclusion is not None:
            region = self.inclusion
            if self.inclusion_permittivity <= 0:
                raise ValueError("Inclusion permittivity must be positive")
            if any(region.lo[k] <= self.domain.lo[k] or region.hi[k] >= self.domain.hi[k]
                   for k in range(3)):
                raise ValueError("Inclusion must lie strictly inside the domain")
            if not any(a < region.lo[0] and region.hi[0] < b
                       for a, b in zip(cuts, cuts[1:])):
                raise ValueError("Inclusion must lie within one layer")
        for i, box in enumerate(self.terminals):
            if any(box.lo[k] - self.collar <= self.domain.lo[k] or
                   box.hi[k] + self.collar >= self.domain.hi[k] for k in range(3)):
                raise ValueError("Terminal collar must lie inside the grounded domain")
            if not any(a < box.lo[0] - self.collar and box.hi[0] + self.collar < b
                       for a, b in zip(cuts, cuts[1:])):
                raise ValueError("Terminal collar must lie within one material layer")
            if self.inclusion is not None:
                collar_lo = tuple(v - self.collar for v in box.lo)
                collar_hi = tuple(v + self.collar for v in box.hi)
                wholly_inside = all(self.inclusion.lo[k] < collar_lo[k] and
                                     collar_hi[k] < self.inclusion.hi[k] for k in range(3))
                wholly_outside = any(collar_hi[k] < self.inclusion.lo[k] or
                                        collar_lo[k] > self.inclusion.hi[k] for k in range(3))
                if not (wholly_inside or wholly_outside):
                    raise ValueError("Terminal collar must avoid inclusion boundary")
            for other in self.terminals[:i]:
                # Disjoint rectangular collars ensure no lift samples another metal.
                if all(box.lo[k] - self.collar < other.hi[k] + self.collar and
                       other.lo[k] - self.collar < box.hi[k] + self.collar for k in range(3)):
                    raise ValueError("Terminal collars must be disjoint")

    def _terminal(self, p, tolerance):
        for j, box in enumerate(self.terminals):
            if box.distance(p) <= tolerance:
                return j
        return None

    def _layer(self, x):
        return sum(x > s for s in self.interfaces)

    def material_permittivity(self, p):
        if self.inclusion is not None and self.inclusion.contains(p):
            return self.inclusion_permittivity
        return self.permittivities[self._layer(p[0])]

    def _inclusion_projection(self, p):
        box = self.inclusion
        if box is None:
            return None
        if box.contains(p):
            distance, axis, coordinate = min(
                (abs(p[k] - face), k, face)
                for k in range(3) for face in (box.lo[k], box.hi[k]))
            q = list(p)
            q[axis] = coordinate
            return distance, tuple(q)
        q = tuple(min(max(p[k], box.lo[k]), box.hi[k]) for k in range(3))
        return sqrt(sum((p[k] - q[k]) ** 2 for k in range(3))), q

    def _direction(self, rng):
        normal = 2 * rng.random() - 1
        phi = 2 * pi * rng.random()
        tangent = sqrt(1 - normal * normal)
        return (normal, tangent * cos(phi), tangent * sin(phi))

    def sample_material_exit(self, center, radius, rng):
        """Sample the epsilon-weighted exit of a radial-sector interface ball."""
        eps_max = max(*self.permittivities, self.inclusion_permittivity or 0.0)
        while True:
            direction = self._direction(rng)
            endpoint = tuple(center[k] + radius * direction[k] for k in range(3))
            if rng.random() * eps_max < self.material_permittivity(endpoint):
                return endpoint

    def terminal_hit(self, point, rng, tolerance=1e-4, max_steps=100000):
        """Return the absorbing terminal index, or None for grounded enclosure."""
        p = tuple(point)
        cuts = (self.domain.lo[0], *self.interfaces, self.domain.hi[0])
        for _ in range(max_steps):
            hit = self._terminal(p, tolerance)
            if hit is not None:
                return hit
            if min(min(p[k] - self.domain.lo[k], self.domain.hi[k] - p[k]) for k in range(3)) <= tolerance:
                return None
            candidates = [(abs(p[0] - s), (s, p[1], p[2])) for s in self.interfaces]
            inclusion_event = self._inclusion_projection(p)
            if inclusion_event is not None:
                candidates.append(inclusion_event)
            distance, q = min(candidates, default=(float("inf"), None))
            radius = None
            if distance <= tolerance:
                limits = [box.distance(q) for box in self.terminals]
                limits += [q[k] - self.domain.lo[k] for k in range(3)]
                limits += [self.domain.hi[k] - q[k] for k in range(3)]
                limits += [abs(q[0] - s) for s in self.interfaces if abs(q[0] - s) > 1e-12]
                if self.inclusion is not None:
                    limits += [abs(q[k] - face) for k in range(3)
                               for face in (self.inclusion.lo[k], self.inclusion.hi[k])
                               if abs(q[k] - face) > 1e-12]
                radius = 0.5 * min(limits)
            if radius is not None and radius > 0 and distance <= 0.01 * radius:
                # Rejection from a uniform sphere gives density proportional
                # to epsilon on each radial material sector (Eq. 5.3).
                p = self.sample_material_exit(q, radius, rng)
            else:
                layer = self._layer(p[0])
                radius = min(box.distance(p) for box in self.terminals)
                radius = min(radius, p[0] - cuts[layer], cuts[layer + 1] - p[0])
                radius = min(radius, *(p[k] - self.domain.lo[k] for k in (1, 2)),
                             *(self.domain.hi[k] - p[k] for k in (1, 2)))
                if inclusion_event is not None:
                    radius = min(radius, inclusion_event[0])
                direction = self._direction(rng)
                p = tuple(p[k] + radius * direction[k] for k in range(3))
        raise RuntimeError("Walk exceeded max_steps")


def _hat_sample(rng, lo, hi, collar):
    """Sample coordinate with density proportional to a box hat; mass = width+c."""
    mass = hi - lo + collar
    u = rng.random() * mass
    if u < collar / 2:
        return lo - collar + sqrt(2 * collar * u)
    if u < collar / 2 + hi - lo:
        return lo + u - collar / 2
    return hi + collar - sqrt(2 * collar * (mass - u))


def estimate_capacitance(problem, samples_per_face=1000, tolerance=1e-4,
                         seed=1, max_steps=100000, return_covariance=False):
    """Estimate physical terminal C and per-entry standard errors in farads.

    A tensor-product hat equals one on each conductor, falls to zero one collar
    width away, and has a signed flux-jump measure on twelve knot faces. The
    inner-face metal contribution is integrated exactly; dielectric faces are
    sampled independently. The returned matrix is raw, without symmetrization.
    When requested, a third result contains joint row covariance matrices.
    """
    if samples_per_face < 2 or tolerance <= 0:
        raise ValueError("Need at least two face samples and positive tolerance")
    rng = Random(seed)
    n = len(problem.terminals)
    means = [[0.0] * n for _ in range(n)]
    covariance = [[[0.0] * n for _ in range(n)] for _ in range(n)]
    c = problem.collar
    cuts = (problem.domain.lo[0], *problem.interfaces, problem.domain.hi[0])
    for i, box in enumerate(problem.terminals):
        layer = next(k for k, (a, b) in enumerate(zip(cuts, cuts[1:])) if a < box.lo[0] and box.hi[0] < b)
        eps = problem.material_permittivity(tuple((box.lo[k] + box.hi[k]) / 2 for k in range(3)))
        for axis in range(3):
            others = [k for k in range(3) if k != axis]
            area_mass = (box.hi[others[0]] - box.lo[others[0]] + c) * (box.hi[others[1]] - box.lo[others[1]] + c)
            for location, sign in ((box.lo[axis] - c, -1), (box.lo[axis], 1),
                                   (box.hi[axis], 1), (box.hi[axis] + c, -1)):
                mass = sign * eps * area_mass / c
                hits = [0] * n
                inner = location == box.lo[axis] or location == box.hi[axis]
                if inner:
                    core_area = (box.hi[others[0]] - box.lo[others[0]]) * (box.hi[others[1]] - box.lo[others[1]])
                    means[i][i] += sign * eps * core_area / c
                for _ in range(samples_per_face):
                    p = [0.0] * 3
                    p[axis] = location
                    for k in others:
                        p[k] = _hat_sample(rng, box.lo[k], box.hi[k], c)
                    # The core rectangle is prescribed metal. Its exact mass
                    # was added above; zero it in the stochastic residual.
                    if inner and all(box.lo[k] <= p[k] <= box.hi[k] for k in others):
                        continue
                    hit = problem.terminal_hit(p, rng, tolerance, max_steps)
                    if hit is not None:
                        hits[hit] += 1
                fractions = [count / samples_per_face for count in hits]
                for j in range(n):
                    means[i][j] += mass * fractions[j]
                    for k in range(n):
                        centered = (fractions[j] if j == k else 0.0) - fractions[j] * fractions[k]
                        covariance[i][j][k] += mass * mass * centered / (samples_per_face - 1)
    scale = EPSILON_0 * problem.length_unit
    matrix = [[v * scale for v in row] for row in means]
    errors = [[sqrt(covariance[i][j][j]) * scale for j in range(n)] for i in range(n)]
    if return_covariance:
        return matrix, errors, [[[value * scale * scale for value in row]
                                 for row in term] for term in covariance]
    return matrix, errors
