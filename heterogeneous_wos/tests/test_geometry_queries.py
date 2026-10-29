"""Check exact rectilinear queries used by heterogeneous walks."""
import unittest
import numpy as np

from heterogeneous_wos.geometry_queries import RectilinearGeometryQueries


class GeometryQueriesTest(unittest.TestCase):
    def setUp(self):
        self.queries = RectilinearGeometryQueries(
            ([0, 0.5, 1], [0, 1], [0, 1]),
            np.array([[[2]], [[8]]]),
            np.array([[[0.1, 0.2, 0.2], [0.2, 0.4, 0.4]]]),
            [3], [[0, 0.5, 0, 1, 0, 1]])

    def test_metal_hit_and_terminal(self):
        hit = self.queries.nearest_metal((0.25, 0.3, 0.3))
        self.assertAlmostEqual(hit.distance, 0.05)
        self.assertEqual(hit.point, (0.2, 0.3, 0.3))
        self.assertEqual(self.queries.terminal_hit((0.25, 0.3, 0.3), 0.06), 3)
        self.assertIsNone(self.queries.terminal_hit((0.25, 0.3, 0.3), 0.04))

    def test_interface_projection_and_material(self):
        hit = self.queries.nearest_interface((0.4, 0.7, 0.7))
        self.assertAlmostEqual(hit.distance, 0.1)
        self.assertEqual(hit.point, (0.5, 0.7, 0.7))
        self.assertEqual(self.queries.material_at((0.25, 0.7, 0.7)), 2)
        self.assertEqual(self.queries.material_at((0.75, 0.7, 0.7)), 8)


if __name__ == "__main__":
    unittest.main()
