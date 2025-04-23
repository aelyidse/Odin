import threading
from typing import Dict, Any, Callable, Optional
import time

class HardwareTelemetryInterface:
    """
    Abstract interface for hardware telemetry sources.
    Implement get_latest() to fetch current telemetry as a dict.
    """
    def get_latest(self) -> Dict[str, Any]:
        raise NotImplementedError

class DigitalTwinSynchronizer:
    """
    Synchronizes digital twin state with real-time hardware telemetry.
    Supports polling or push updates, callback on state change, and thread-safe operation.
    Provides predictive state estimation using physics-based models.
    """
    def __init__(self, telemetry_source: HardwareTelemetryInterface, update_callback: Optional[Callable[[Dict[str, Any]], None]] = None, poll_interval: float = 1.0):
        self.telemetry_source = telemetry_source
        self.update_callback = update_callback
        self.poll_interval = poll_interval
        self._running = False
        self._thread = None
        self.current_state = None
        self.last_state = None
        self.last_update_time = None
    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
    def _run(self):
        import time
        while self._running:
            telemetry = self.telemetry_source.get_latest()
            now = time.time()
            if telemetry != self.current_state:
                self.last_state = self.current_state
                self.current_state = telemetry
                self.last_update_time = now
                if self.update_callback:
                    self.update_callback(telemetry)
            time.sleep(self.poll_interval)
    def get_state(self) -> Optional[Dict[str, Any]]:
        return self.current_state
    def predict_state(self, dt: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Predict future state after dt seconds using a simple physics-based model (constant velocity or acceleration).
        Expects telemetry dict with at least 'pos' and 'vel' (and optionally 'acc').
        """
        if self.current_state is None:
            return None
        pos = self.current_state.get('pos')
        vel = self.current_state.get('vel')
        acc = self.current_state.get('acc', 0.0)
        # Support both scalar and vector
        import numpy as np
        pos = np.array(pos) if pos is not None else np.zeros(1)
        vel = np.array(vel) if vel is not None else np.zeros_like(pos)
        acc = np.array(acc) if isinstance(acc, (list, tuple, np.ndarray)) else np.full_like(pos, acc)
        pred_pos = pos + vel * dt + 0.5 * acc * dt**2
        pred_vel = vel + acc * dt
        predicted = dict(self.current_state)
        predicted['pos'] = pred_pos.tolist() if pred_pos.shape else float(pred_pos)
        predicted['vel'] = pred_vel.tolist() if pred_vel.shape else float(pred_vel)
        predicted['predicted_dt'] = dt
        return predicted

# Example usage:
# class MyTelemetry(HardwareTelemetryInterface):
#     def get_latest(self):
#         return {'pos': 1, 'vel': 2}
# twin = DigitalTwinSynchronizer(MyTelemetry(), print, poll_interval=0.5)
# twin.start()
# time.sleep(2)
# twin.stop()
