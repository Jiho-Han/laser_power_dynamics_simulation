import unittest
from transition_dynamic_simulator.four_level_laser_system import (
    OpticallyPumped4LevelSystem,
)


class TestFourLevelLaserSystem(unittest.TestCase):
    def test_sketch_interfaces(self):
        # First let's define the system with the two parameter that doesn't chnage with time.
        tau21 = 2
        sigma_emission = 3
        I_p = 2
        I_s = 3
        dt = 4
        flls = OpticallyPumped4LevelSystem(tau21, 1, 2, 3, 4)
        # Then whatever the method is, we'll set the pump like this
        flls.I_pump = I_p
        print(flls.I_pump)
        # and set the input signal intensity like this.
        # flls.signal(I_s)
        # To take a step in time...
        # flls.time_t(dt)
        # To get the value of the added power at current time we do this.
        # flls.dP_S()


if __name__ == "__main__":
    unittest.main()
