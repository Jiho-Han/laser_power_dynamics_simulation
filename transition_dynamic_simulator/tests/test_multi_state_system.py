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
        np.testing.assert_equal(
            system.dNdt(),
            [
                -66 + 11 + 88 + 231,
                -(4 + 5 + 6) * 22 + 22 + 110 + (8 * 33),
                -(7 + 8 + 9) * 33 + 33 + (6 * 22) + (9 * 33),
            ],
        )

    def test_rate_sum(self):
        """
        The rate of change for population among each state must sum to 0.
        """
        tau = TimeConstants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        N = PopulationDistrubution([11, 22, 33])
        system = MultiStateSystem(tau, N)
        dNdt = system.dNdt()
        np.testing.assert_almost_equal(sum(dNdt), 0)

    def test_equilibrium(self):
        """When we solve for equilibrium, all the rates should become 0."""
        tau = TimeConstants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        N = PopulationDistrubution([11, 22, 33])
        system = MultiStateSystem(tau, N)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_almost_equal(system.dNdt(), [0, 0, 0])
        system.solve_for_equilibrium()
        np.testing.assert_almost_equal(system.dNdt(), [0, 0, 0])

    def test_numreical_results(self):
        """
        Now let's try with examples that are easier to visualise...
        """

        # static system, so obviously rate is always 0.
        system = MultiStateSystem(
            tau=TimeConstants.get_static_tc(2),
            N=PopulationDistrubution([0.5, 0.5]),
        )
        np.testing.assert_almost_equal(system.dNdt(), [0, 0])
        with np.testing.assert_raises(ValueError):
            system.solve_for_equilibrium()

        # rates are identical for transition in either direction. So dNdt is also 0.
        tau = TimeConstants.get_static_tc(2)
        tau[0][1] = 5
        tau[1][0] = 5
        system = MultiStateSystem(
            tau=tau,
            N=PopulationDistrubution([0.5, 0.5]),
        )
        np.testing.assert_almost_equal(system.dNdt(), [0, 0])
        # And so the population distribution will not change when you solve for equilibrium.
        system.solve_for_equilibrium()
        np.testing.assert_equal(system.N, [0.5, 0.5])

        # tau01 is twice tau10.
        tau = TimeConstants.get_static_tc(2)
        tau[0][1] = 10
        tau[1][0] = 5
        system = MultiStateSystem(
            tau=tau,
            N=PopulationDistrubution([0.5, 0.5]),
        )
        np.testing.assert_almost_equal(system.dNdt(), [0.05, -0.05])
        # tau10 is twice as fast as tau01. So we expect twice the populatoin in state0 as
        # in state 1.
        system.solve_for_equilibrium()
        np.testing.assert_almost_equal(system.N, [2 / 3, 1 / 3])
        np.testing.assert_almost_equal(system.dNdt(), [0, 0])


if __name__ == "__main__":
    unittest.main()
