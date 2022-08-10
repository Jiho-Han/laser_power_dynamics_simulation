from transition_dynamic_simulator.multi_state_system import (
    MultiStateSystem,
    PopulationDistrubution,
)
from transition_dynamic_simulator.time_constants import TimeConstants
from scipy.constants import lambda2nu, h

#
class OpticallyPumped4LevelSystem(MultiStateSystem):
    def __init__(self, tau21, signal_cross_section, pump_cross_section, signal_wavelength, pump_wavelength) -> None:
        tau = TimeConstants.get_static_tc(4)
        tau[2][1] = tau21
        N = PopulationDistrubution([1, 0, 0, 0])
        super().__init__(tau, N)
        self.signal_cross_section = signal_cross_section
        self.pump_cross_section = pump_cross_section
        self.signal_wavelength = signal_wavelength
        self.pump_wavelength = pump_wavelength
        self._I_pump = 0.0
        self._I_signal = 0.0

    @property
    def I_pump(self):
        return self._I_pump

    @I_pump.setter
    def I_pump(self, I_pump):
        self._I_pump = I_pump
        R = self._rates_from_cross_section(I_pump, self.pump_cross_section, self.pump_wavelength)
        self.tau[0][3] = 1 / R
        self.tau[3][0] = 1 / R

    @property
    def I_signal(self):
        return self._I_signal

    @I_signal.setter
    def I_signal(self, I_signal):
        R = self._rates_from_cross_section(I_signal, self.signal_cross_section, self.signal_wavelength)
        self.tau[1][2] = 1 / R
        self.tau[2][1] = 1 / R

    @staticmethod
    def _rates_from_cross_section(intensity, cross_section, wavelength):
        return cross_section * intensity / (h * lambda2nu(wavelength))
