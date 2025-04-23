import threading
import time
from typing import Callable, List, Dict, Any, Optional

class MultiRateControlLoop:
    """
    Multi-rate control loop manager with hardware timing guarantees.
    Each controller runs at its own rate, with precise timing and thread isolation.
    """
    def __init__(self):
        self.loops = []
        self.threads = []
        self.running = False
    def add_loop(self, control_fn: Callable[[], None], rate_hz: float, name: Optional[str] = None):
        """
        Add a control function to run at a specified rate (Hz).
        control_fn: function to call each cycle
        rate_hz: execution frequency
        name: optional identifier
        """
        loop_info = {'fn': control_fn, 'period': 1.0/rate_hz, 'name': name or f'loop_{len(self.loops)}'}
        self.loops.append(loop_info)
    def _run_loop(self, loop_info):
        period = loop_info['period']
        fn = loop_info['fn']
        name = loop_info['name']
        next_time = time.perf_counter()
        while self.running:
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            next_time += period
            sleep_time = max(0, next_time - time.perf_counter())
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Timing guarantee log
            actual_period = time.perf_counter() - start
            if abs(actual_period - period) > 1e-4:
                print(f"[WARN][{name}] Timing deviation: {actual_period:.6f}s (target {period:.6f}s)")
    def start(self):
        self.running = True
        self.threads = []
        for loop_info in self.loops:
            t = threading.Thread(target=self._run_loop, args=(loop_info,), daemon=True)
            t.start()
            self.threads.append(t)
    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
    def status(self) -> List[Dict[str, Any]]:
        return [{'name': l['name'], 'period': l['period']} for l in self.loops]

# Example usage:
# def fast_control():
#     print("Fast loop")
# def slow_control():
#     print("Slow loop")
# mrc = MultiRateControlLoop()
# mrc.add_loop(fast_control, rate_hz=1000, name="fast")
# mrc.add_loop(slow_control, rate_hz=10, name="slow")
# mrc.start()
# time.sleep(2)
# mrc.stop()
# print(mrc.status())
