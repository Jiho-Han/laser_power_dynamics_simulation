import numpy as np
from transition_dynamic_simulator.time_constants import TimeConstants


class PopulationDistrubution(list):
    """Represent a distribution among different states. Does not care whether it is proportion or absolute number."""

    def __init__(self, N):
        self.validate_data(N)
        super().__init__(N)

    @staticmethod
    def validate_data(N):
        try:
            assert isinstance(N, list)
            for Nn in N:
                assert isinstance(Nn, (float, int))
        except AssertionError:
            raise ValueError


class MultiStateSystem(list):
    def __init__(self, tau: TimeConstants, N: PopulationDistrubution) -> None:
        self.validate_data(tau, N)
        self.tau = tau  # Represents the time constants. tau.get_rates() gives reciprocals, which represents the rates.
        self.N = N

    @staticmethod
    def validate_data(tau, N):
        assert isinstance(tau, TimeConstants)
        assert isinstance(N, PopulationDistrubution)
        try:
            assert len(N) == len(tau)
        except AssertionError:
            raise AssertionError(
                "Make sure the number states in the time constants match the number of states indicated by populations!"
            )

    def _gain_matrix(self):
        "This matrix post multiplied by N would be the gains for each state."
        return np.transpose(self.tau.get_rates())

    def _loss_matrix(self):
        "This matrix post multiplied by N would be the loss for each state."
        return np.diag([sum(row) for row in self.tau.get_rates()])

    def _rate_matrix(self):
        "Post multiplying this Matrix with N will give overall rate matrix."
        return self._gain_matrix() - self._loss_matrix()

    def dNdt(self):
        "rate of change for each state"
        return np.matmul(self._rate_matrix(), self.N)

    def solve_for_equilibrium(self):
        A = self._rate_matrix()
        dNdt = np.zeros(len(A))
        # At equilibrium, clearly rate_matrix * N = dNdt = 0. So it is tempting to solve for N. But this doesn't actually work:
        # The set of equation can of course only give the ratio between the states. The missing piece of information
        # is the total population. So let's just remove the last line, and replace with this information.
        A[-1] = np.ones(len(A))
        B = dNdt
        B[-1] = sum(self.N)

        try:
            N = np.matmul(np.linalg.inv(A), B)
            self.N = N
        except np.linalg.LinAlgError:
            raise ValueError(
                "Currnetly does not handle cases where some states are static"
            )
