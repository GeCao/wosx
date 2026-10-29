"""Chapter 5 planar-interface WoS kernel in an infinite parallel-plate slab.

The plates are at x=0 and x=width. Interfaces are parallel to them. A sphere
centered on an interface exits with probability proportional to permittivity
in each hemisphere (Eq. 5.3). This geometry makes the interface ball exact;
projection of a nearby walk point to its center is controlled by tolerance.
"""

from bisect import bisect_right
from dataclasses import dataclass
from math import sqrt
from random import Random


@dataclass(frozen=True)
class LayeredDielectric:
    width: float
    interfaces: tuple[float, ...]
    permittivities: tuple[float, ...]
    left_voltage: float = 0.0
    right_voltage: float = 1.0

    def __post_init__(self):
        if self.width <= 0 or len(self.permittivities) != len(self.interfaces) + 1:
            raise ValueError("Need positive width and one permittivity per layer")
        if any(e <= 0 for e in self.permittivities):
            raise ValueError("Permittivities must be positive")
        cuts = (0.0, *self.interfaces, self.width)
        if any(a >= b for a, b in zip(cuts, cuts[1:])):
            raise ValueError("Interfaces must be strictly ordered inside the plates")

    def exact_potential(self, x: float) -> float:
        """One-dimensional transmission solution, used for validation."""
        if not 0 <= x <= self.width:
            raise ValueError("Point lies outside slab")
        cuts = (0.0, *self.interfaces, self.width)
        resistance = sum((b - a) / e for a, b, e in zip(cuts, cuts[1:], self.permittivities))
        to_x = sum(max(0.0, min(x, b) - a) / e
                   for a, b, e in zip(cuts, cuts[1:], self.permittivities))
        return self.left_voltage + (self.right_voltage - self.left_voltage) * to_x / resistance


def estimate_potential(case: LayeredDielectric, x: float, walks: int = 10000,
                       tolerance: float = 1e-4, seed: int = 1,
                       max_steps: int = 100000) -> tuple[float, float]:
    """Return Monte Carlo mean and standard error at an x coordinate.

    In a homogeneous layer the sphere radius is the distance to the closest
    plate or interface. Near an interface the point is projected to that plane;
    the centered sphere excludes all other interfaces and plates. Its exit
    hemisphere is selected with relative-permittivity weights. The shell and
    interface projection induce finite-tolerance bias.
    """
    if not 0 <= x <= case.width or walks < 1 or tolerance <= 0 or max_steps < 1:
        raise ValueError("Invalid point or sampling settings")
    rng = Random(seed)
    boundaries = (0.0, *case.interfaces, case.width)
    total = total_sq = 0.0
    for _ in range(walks):
        pos = x
        for _step in range(max_steps):
            if pos <= tolerance:
                value = case.left_voltage
                break
            if case.width - pos <= tolerance:
                value = case.right_voltage
                break
            layer = min(bisect_right(case.interfaces, pos), len(case.permittivities) - 1)
            # A point within tolerance of a material plane uses the centered
            # transmission kernel. The ball reaches only halfway to neighbors.
            nearest = min(range(len(case.interfaces)),
                          key=lambda i: abs(pos - case.interfaces[i])) if case.interfaces else None
            j = nearest + 1 if nearest is not None else None
            interface_radius = (0.5 * min(boundaries[j] - boundaries[j - 1],
                                          boundaries[j + 1] - boundaries[j])
                                if j is not None else None)
            if (nearest is not None and abs(pos - case.interfaces[nearest]) <= tolerance
                    and abs(pos - case.interfaces[nearest]) <= 0.01 * interface_radius):
                radius = interface_radius
                left_e, right_e = case.permittivities[nearest:nearest + 2]
                right = rng.random() < right_e / (left_e + right_e)
                # Uniform direction conditional on the chosen hemisphere.
                normal = rng.random() * (1 if right else -1)
                pos = case.interfaces[nearest] + radius * normal
            else:
                radius = min(pos - boundaries[layer], boundaries[layer + 1] - pos)
                # x component of a uniform direction on S^2 is uniform [-1,1].
                pos += radius * (2 * rng.random() - 1)
        else:
            raise RuntimeError("Walk exceeded max_steps; reduce tolerance or raise limit")
        total += value
        total_sq += value * value
    mean = total / walks
    variance = max(0.0, (total_sq / walks - mean * mean) * walks / max(1, walks - 1))
    return mean, sqrt(variance / walks)
