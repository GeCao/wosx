"""Regression checks for the bounded 3D heterogeneous terminal solver."""

import unittest

from heterogeneous_wos import Box, BoxCapacitanceProblem, estimate_capacitance
from heterogeneous_wos.reference import finite_volume_capacitance


class CapacitanceTest(unittest.TestCase):
    def problem(self):
        return BoxCapacitanceProblem(
            Box((0, 0, 0), (1, 1, 1)),
            (Box((.20, .40, .40), (.35, .60, .60)),
             Box((.65, .40, .40), (.80, .60, .60))),
            (.5,), (2., 8.), .05, 1e-6)

    def test_collar_must_avoid_interface(self):
        with self.assertRaises(ValueError):
            BoxCapacitanceProblem(Box((0, 0, 0), (1, 1, 1)),
                                  (Box((.42, .4, .4), (.48, .6, .6)),),
                                  (.5,), (2., 8.), .05)

    def test_covariance_and_grid_reference(self):
        matrix, error, covariance = estimate_capacitance(
            self.problem(), 300, 5e-4, 2026, return_covariance=True)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(error[i][j] ** 2, covariance[i][j][j], delta=1e-34)
                self.assertAlmostEqual(covariance[i][0][1], covariance[i][1][0], delta=1e-34)
        try:
            coarse = finite_volume_capacitance(self.problem(), 20)
            fine = finite_volume_capacitance(self.problem(), 40)
        except RuntimeError:
            self.skipTest("NumPy/SciPy unavailable")
        self.assertLess(abs(fine[0][0] - matrix[0][0]),
                        abs(coarse[0][0] - matrix[0][0]) + 3 * error[0][0])
        self.assertLess(abs(fine[0][1] - fine[1][0]), 1e-24)

    def test_matrix_signs_and_reciprocity(self):
        matrix, error = estimate_capacitance(self.problem(), 1000, 5e-4, 2026)
        self.assertGreater(matrix[0][0], 0)
        self.assertGreater(matrix[1][1], 0)
        self.assertLess(matrix[0][1], 0)
        self.assertLess(matrix[1][0], 0)
        self.assertLess(abs(matrix[0][1] - matrix[1][0]),
                        4 * (error[0][1] ** 2 + error[1][0] ** 2) ** .5)


if __name__ == "__main__":
    unittest.main()
