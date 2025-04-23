import threading
import time
from typing import Dict, Any, Callable, Optional

class DigitalTwin:
    """
    Real-time, high-fidelity digital representation of the ODIN physical system.
    Synchronizes with live telemetry, control, and simulated/physical data streams.
    """
    def __init__(self, update_interval_s: float = 0.01):
        self.state: Dict[str, Any] = {}
        self.history: list = []
        self.update_interval_s = update_interval_s
        self.running = False
        self.lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self.data_sources: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.observers: list = []

    def register_data_source(self, name: str, fetch_fn: Callable[[], Dict[str, Any]]):
        """Register a live data source (sensor, subsystem, simulation, etc.)."""
        self.data_sources[name] = fetch_fn

    def register_observer(self, observer_fn: Callable[[Dict[str, Any]], None]):
        """Register a callback to be notified on every state update."""
        self.observers.append(observer_fn)

    def start(self):
        if not self.running:
            self.running = True
            self._update_thread = threading.Thread(target=self._run, daemon=True)
            self._update_thread.start()

    def stop(self):
        self.running = False
        if self._update_thread:
            self._update_thread.join()
            self._update_thread = None

    def _run(self):
        while self.running:
            self.update()
            time.sleep(self.update_interval_s)

    def update(self):
        """Fetch all data sources, update state, notify observers, and archive history."""
        new_state = {}
        for name, fetch_fn in self.data_sources.items():
            try:
                new_state[name] = fetch_fn()
            except Exception as e:
                new_state[name] = {'error': str(e)}
        with self.lock:
            self.state = new_state
            self.history.append((time.time(), new_state.copy()))
        for obs in self.observers:
            obs(new_state)

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return self.state.copy()

    def get_history(self) -> list:
        with self.lock:
            return list(self.history)

# Example usage:
# twin = DigitalTwin(update_interval_s=0.01)
# twin.register_data_source('laser', lambda: {'power': get_laser_power(), 'temp': get_laser_temp()})
# twin.register_data_source('tracking', lambda: get_tracking_state())
# twin.register_observer(lambda state: print('Digital Twin Update:', state))
# twin.start()
# ...
# twin.stop()
