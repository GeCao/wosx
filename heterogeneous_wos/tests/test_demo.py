"""Analytic transmission checks for the planar WoS kernel."""

import unittest

from heterogeneous_wos import LayeredDielectric, estimate_potential


class LayeredWoSTest(unittest.TestCase):
    def test_interface_weight(self):
        case = LayeredDielectric(1.0, (0.5,), (2.0, 8.0))
        self.assertAlmostEqual(case.exact_potential(0.5), 0.8)
        mean, stderr = estimate_potential(case, 0.5, walks=20000, seed=2026)
        self.assertLess(abs(mean - 0.8), 4 * stderr + 0.002)

    def test_multilayer(self):
        case = LayeredDielectric(1.0, (0.25, 0.75), (1.0, 4.0, 2.0))
        mean, stderr = estimate_potential(case, 0.4, walks=20000, seed=32)
        self.assertLess(abs(mean - case.exact_potential(0.4)), 4 * stderr + 0.003)


if __name__ == "__main__":
    unittest.main()
