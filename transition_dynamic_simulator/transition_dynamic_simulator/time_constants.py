import numpy as np


class TimeConstants(list):
    """Class to represent time constants for the transitinos between different states (electronic energy levels, chemical species etc).

    The species are indexed with numbers starting from 0 as any list. So with

    tau = TimeConstants([....])
    tau[0][3] means time constant for transition from state 0 to 3.

    clearly, tau[3][3], while perhaps mathematically meaningful, is physically meaningless.

    """

    def __init__(self, time_constants):
        self.validate_data(time_constants)
        super().__init__(time_constants)

    @staticmethod
    def validate_data(tau):
        try:
            assert isinstance(tau, list)
            n = len(tau)
            for row in tau:
                assert isinstance(row, list)
                assert len(row) == n
                for value in row:
                    assert isinstance(value, (float, int))
        except AssertionError:
            raise ValueError

    def get_rates(self):
        return np.reciprocal(np.array(self).astype(float))
