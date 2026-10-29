"""Geometry and weak-measure checks for the general rectilinear path."""

import unittest
from random import Random
from math import sqrt

from heterogeneous_wos import Box
from heterogeneous_wos.rectilinear import (MaterialBox, RectilinearProblem,
                                            estimate_rectilinear_capacitance)
from heterogeneous_wos.reference import finite_volume_capacitance


class RectilinearTest(unittest.TestCase):
    def problem(self, outer="grounded"):
        return RectilinearProblem(
            Box((0, 0, 0), (1, 1, 1)),
            ((Box((.2, .2, .2), (.4, .4, .4)),
              Box((.4, .2, .2), (.5, .4, .4))),
             (Box((.7, .6, .6), (.8, .8, .8)),)),
            2.0,
            (MaterialBox(Box((.45, .3, .3), (.65, .7, .7)), 8.0),
             MaterialBox(Box((.55, .4, .4), (.6, .6, .6)), 3.0)),
            .1, 1e-6, outer)

    def test_union_and_ordered_materials(self):
        problem = self.problem()
        self.assertEqual(problem.terminal((.45, .3, .3)), 0)
        self.assertEqual(problem.material_at((.575, .5, .5)), 3.0)
        self.assertEqual(problem.material_at((.5, .5, .5)), 8.0)
        self.assertEqual(problem.material_at((.9, .5, .5)), 2.0)
        for row in range(2):
            components, audit = problem.charge_measure(row)
            self.assertLess(abs(audit), 1e-10)
            self.assertTrue(any(c.prescribed == row for c in components))
            self.assertTrue(any(c.prescribed is None for c in components))

    def test_reflected_walk_and_boundary_trace(self):
        problem = RectilinearProblem(
            Box((0, 0, 0), (1, 1, 1)),
            ((Box((.3, .3, .3), (.7, .7, .7)),),),
            2.0, support=.4, outer="reflecting")
        components, audit = problem.charge_measure(0)
        self.assertLess(abs(audit), 1e-10)
        self.assertTrue(any(c.outer and abs(c.mass) > 0 for c in components))
        self.assertEqual(problem.terminal_hit((.01, .1, .1), Random(1), 1e-3), 0)
        try:
            matrix = finite_volume_capacitance(problem, 20)
        except RuntimeError:
            self.skipTest("NumPy/SciPy unavailable")
        self.assertLess(abs(matrix[0][0]), 1e-22)

    def test_open_exterior_kernel(self):
        problem = RectilinearProblem(
            Box((0, 0, 0), (1, 1, 1)),
            ((Box((.4, .4, .4), (.6, .6, .6)),),),
            1.0, support=.1, outer="open")
        radius = sqrt(3) / 2
        point = (.5 + 2 * radius, .5, .5)
        rng = Random(11)
        returns = [problem._exterior_return(point, rng) for _ in range(10000)]
        returned = [p for p in returns if p is not None]
        self.assertLess(abs(len(returned) / len(returns) - .5), .02)
        mean_cosine = sum((p[0] - .5) / radius for p in returned) / len(returned)
        self.assertLess(abs(mean_cosine - .5), .03)
        components, audit = problem.charge_measure(0)
        self.assertLess(abs(audit), 1e-10)
        self.assertTrue(components)

    def test_estimator_returns_joint_covariance(self):
        matrix, covariance, diagnostics = estimate_rectilinear_capacitance(
            self.problem(), samples_per_row=100, tolerance=1e-3, seed=7)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(covariance), 2)
        self.assertLess(abs(diagnostics[0]["signed_audit"]), 1e-10)
        self.assertAlmostEqual(covariance[0][0][1], covariance[0][1][0])
        self.assertGreaterEqual(covariance[0][0][0], 0)


if __name__ == "__main__":
    unittest.main()
