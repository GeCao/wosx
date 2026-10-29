"""Exact CPU queries for rectilinear heterogeneous walk geometry.

This is a host-side reference and a geometry export point for the GPU walker.
Metal and interface distances use boxes/rectangles directly, avoiding a mesh
approximation of the imported case.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Hit:
    distance: float
    point: tuple[float, float, float]
    index: int
    terminal: int | None = None


class RectilinearGeometryQueries:
    """Nearest metal/interface and material lookups over a coordinate case."""

    def __init__(self, coordinates, epsilon, boxes, tags, interfaces):
        self.coordinates = tuple(np.asarray(axis, dtype=float) for axis in coordinates)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.boxes = np.asarray(boxes, dtype=float)
        self.tags = np.asarray(tags, dtype=int)
        self.interfaces = np.asarray(interfaces, dtype=float).reshape(-1, 6)
        if (len(self.coordinates) != 3 or any(len(axis) < 2 or np.any(np.diff(axis) <= 0)
                                              for axis in self.coordinates)):
            raise ValueError("Three increasing coordinate axes are required")
        if self.epsilon.shape != tuple(len(axis) - 1 for axis in self.coordinates):
            raise ValueError("Epsilon grid does not match coordinate axes")
        if self.boxes.ndim != 3 or self.boxes.shape[1:] != (2, 3) or len(self.tags) != len(self.boxes):
            raise ValueError("Metal boxes must have shape (n, 2, 3) and one tag each")
        if np.any(self.boxes[:, 0] >= self.boxes[:, 1]) or np.any(self.epsilon <= 0):
            raise ValueError("Boxes and permittivities must be positive")
        if np.any((self.interfaces[:, 0] < 0) | (self.interfaces[:, 0] > 2)):
            raise ValueError("Invalid interface axes")

    @classmethod
    def from_case(cls, case):
        return cls(case.coordinates, case.eps, case.boxes, case.tags, case.interfaces)

    def material_at(self, point):
        p = np.asarray(point, dtype=float)
        if p.shape != (3,):
            raise ValueError("Point must have three coordinates")
        if any(p[k] < axis[0] or p[k] > axis[-1]
               for k, axis in enumerate(self.coordinates)):
            raise ValueError("Point lies outside the extraction window")
        indices = tuple(int(np.clip(np.searchsorted(axis, p[k], side="right") - 1,
                                    0, len(axis) - 2))
                        for k, axis in enumerate(self.coordinates))
        return float(self.epsilon[indices])

    def nearest_metal(self, point):
        p = np.asarray(point, dtype=float)
        if p.shape != (3,):
            raise ValueError("Point must have three coordinates")
        q = np.clip(p, self.boxes[:, 0], self.boxes[:, 1])
        distances = np.linalg.norm(q - p, axis=1)
        index = int(np.argmin(distances))
        return Hit(float(distances[index]), tuple(q[index]), index, int(self.tags[index]))

    def nearest_interface(self, point):
        p = np.asarray(point, dtype=float)
        if p.shape != (3,):
            raise ValueError("Point must have three coordinates")
        if not len(self.interfaces):
            return None
        q = np.broadcast_to(p, (len(self.interfaces), 3)).copy()
        for axis in range(3):
            selected = np.flatnonzero(self.interfaces[:, 0] == axis)
            if not len(selected):
                continue
            tangential = [k for k in range(3) if k != axis]
            patches = self.interfaces[selected]
            q[selected, axis] = patches[:, 1]
            q[selected, tangential[0]] = np.clip(p[tangential[0]], patches[:, 2], patches[:, 3])
            q[selected, tangential[1]] = np.clip(p[tangential[1]], patches[:, 4], patches[:, 5])
        distances = np.linalg.norm(q - p, axis=1)
        index = int(np.argmin(distances))
        return Hit(float(distances[index]), tuple(q[index]), index)

    def terminal_hit(self, point, tolerance):
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        hit = self.nearest_metal(point)
        return hit.terminal if hit.distance <= tolerance else None
