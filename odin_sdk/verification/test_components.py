import unittest
import numpy as np
from physics_engine.fiber_thermal import FiberThermalManagement
from physics_engine.diffraction_grating import DiffractionGrating
from target_tracking.multi_target_tracker import KalmanTracker, MultiTargetTracker
from core_architecture.pid_controller import PIDController
from llm_command_control.nlp_mission_parser import MissionNLPParser

class TestFiberThermalManagement(unittest.TestCase):
    def test_steady_state_profile(self):
        class DummyFiber: core_diameter_um = 10
        class DummyMaterial:
            def thermal_conductivity(self, temp): return 1.4
            def specific_heat(self, temp): return 700
            density = 2200
        fiber = DummyFiber()
        mat = DummyMaterial()
        ftm = FiberThermalManagement(fiber, mat, length_m=1.0)
        heat = np.ones_like(ftm.z) * 10
        temp = ftm.steady_state_profile(heat)
        self.assertTrue(np.all(temp >= ftm.ambient_temp_C))

class TestDiffractionGrating(unittest.TestCase):
    def test_diffraction_angle(self):
        grating = DiffractionGrating(groove_density_lpm=1200, blaze_angle_deg=30, substrate_material='fused_silica')
        angle = grating.diffraction_angle(1064, 1, 0)
        self.assertTrue(-90 < angle < 90)

class TestKalmanTracker(unittest.TestCase):
    def test_predict_update(self):
        F = np.eye(2)
        Q = np.eye(2)*0.1
        H = np.eye(2)
        R = np.eye(2)*0.1
        x0 = np.array([0,0])
        P0 = np.eye(2)
        tracker = KalmanTracker(x0, P0, F, Q, H, R)
        tracker.predict()
        tracker.update(np.array([1,1]))
        state, cov = tracker.get_state()
        self.assertEqual(state.shape, (2,))

class TestPIDController(unittest.TestCase):
    def test_pid_converges(self):
        pid = PIDController(1.0, 0.1, 0.05, setpoint=10.0, output_limits=(0, 100), dt=0.1)
        val = 0.0
        for _ in range(50):
            cmd = pid.step(val)
            val += 0.2 * cmd
        self.assertTrue(abs(val - 10.0) < 1.0)

class TestMissionNLPParser(unittest.TestCase):
    def test_parse(self):
        parser = MissionNLPParser()
        directive = parser.parse("Engage targets Alpha, Bravo with priority high and duration 30s at power 5kW")
        self.assertEqual(directive.action, 'engage')
        self.assertIn('Alpha', directive.targets)
        self.assertIn('Bravo', directive.targets)
        self.assertIn('priority', directive.parameters)
        self.assertIn('duration', directive.parameters)
        self.assertIn('power', directive.parameters)

if __name__ == '__main__':
    unittest.main()
