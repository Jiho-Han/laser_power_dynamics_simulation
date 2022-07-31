import unittest
import numpy as np
from transition_dynamic_simulator.time_constants import TimeConstants


class TestTimeConstants(unittest.TestCase):
    def test_making_time_constants(self):
        tau = TimeConstants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with np.testing.assert_raises(ValueError):
            tau = TimeConstants([[1, 2, 3], [4, 5, 6], [7, 8, "2"]])
        with np.testing.assert_raises(ValueError):
            tau = TimeConstants([[1, 2, 3], [4, 5, 6], 3])

    def test_get_rates(self):
        tau = TimeConstants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        gamma = tau.get_rates()

        np.testing.assert_equal(
            gamma, [[1 / 1, 1 / 2, 1 / 3], [1 / 4, 1 / 5, 1 / 6], [1 / 7, 1 / 8, 1 / 9]]
        )

    def test_get_static_matrix(self):
        tau = TimeConstants.get_static_tc(2)
        np.testing.assert_equal(tau, [[np.inf, np.inf], [np.inf, np.inf]])


if __name__ == "__main__":
    unittest.main()
