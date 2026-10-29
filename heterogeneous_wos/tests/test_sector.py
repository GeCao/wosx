"""Analytic face, edge, and corner transmission probabilities."""

import unittest
from random import Random

from heterogeneous_wos import Box, BoxCapacitanceProblem


class SectorTest(unittest.TestCase):
    def test_sector_solid_angles(self):
        problem = BoxCapacitanceProblem(
            Box((0, 0, 0), (1, 1, 1)),
            (Box((.1, .1, .1), (.2, .2, .2)),),
            (), (1.,), .02,
            inclusion=Box((.4, .4, .4), (.6, .6, .6)),
            inclusion_permittivity=8.)
        rng = Random(77)
        for center, other_sectors in (((.4, .5, .5), 1),
                                      ((.4, .4, .5), 3),
                                      ((.4, .4, .4), 7)):
            hits = sum(problem.inclusion.contains(problem.sample_material_exit(center, .05, rng))
                       for _ in range(10000))
            expected = 8 / (8 + other_sectors)
            self.assertLess(abs(hits / 10000 - expected), .018)


if __name__ == "__main__":
    unittest.main()
