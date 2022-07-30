from multiprocessing.sharedctypes import Value
import unittest
import numpy as np
from transition_dynamic_simulator.multi_state_system import (
    MultiStateSystem,
    PopulationDistrubution,
)
from transition_dynamic_simulator.time_constants import TimeConstants


class TestPopulationDistribution(unittest.TestCase):
    def test_popdist(self):
        N = PopulationDistrubution([1, 2, 3])
        with np.testing.assert_raises(ValueError):
            N = PopulationDistrubution(3)
        with np.testing.assert_raises(ValueError):
            N = PopulationDistrubution([3, "2"])


class TestMultiStateSystem(unittest.TestCase):
    def test_instantiation(self):
        tau = TimeConstants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        N = PopulationDistrubution([1, 2, 3])
        MultiStateSystem(tau, N)
        with np.testing.assert_raises(AssertionError):
            tau = TimeConstants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            N = PopulationDistrubution([1, 2])
            MultiStateSystem(tau, N)

    def test_gain_matrix(self):
        tau = TimeConstants(
            [[1 / 1, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6], [1 / 7, 1 / 8, 1 / 9]]
        )
        N = PopulationDistrubution([11, 22, 33])
        system = MultiStateSystem(tau, N)
        np.testing.assert_equal(
            system._gain_matrix(), [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        )

    def test_loss_matrix(self):
        tau = TimeConstants(
            [[1 / 1, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6], [1 / 7, 1 / 8, 1 / 9]]
        )
        N = PopulationDistrubution([11, 22, 33])
        system = MultiStateSystem(tau, N)
        np.testing.assert_equal(
            system._loss_matrix(),
            [[1 + 2 + 3, 0, 0], [0, 4 + 5 + 6, 0], [0, 0, 7 + 8 + 9]],
        )

    def test_rate_matrix(self):
        tau = TimeConstants(
            [[1 / 1, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6], [1 / 7, 1 / 8, 1 / 9]]
        )
        N = PopulationDistrubution([11, 22, 33])
        system = MultiStateSystem(tau, N)
        np.testing.assert_equal(
            system._rate_matrix(),
            [[-5.0, 4.0, 7.0], [2.0, -10.0, 8.0], [3.0, 6.0, -15.0]],
        )

    def test_dNdt(self):
        tau = TimeConstants(
            [[1 / 1, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6], [1 / 7, 1 / 8, 1 / 9]]
        )
        N = PopulationDistrubution([11, 22, 33])
        system = MultiStateSystem(tau, N)

        system.solve_for_equilibrium()
        print(system.dNdt())
        print(system.N)


if __name__ == "__main__":
    unittest.main()
