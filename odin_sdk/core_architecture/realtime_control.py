import time
import threading
from typing import Callable, List, Dict, Any, Optional

class RealTimeControlTask:
    """Encapsulates a control task with deterministic timing requirements."""
    def __init__(self, name: str, interval_s: float, callback: Callable[[float, Dict[str, Any]], None], context: Optional[Dict[str, Any]] = None, priority: int = 0):
        self.name = name
        self.interval_s = interval_s
        self.callback = callback
        self.context = context or {}
        self.priority = priority
        self.last_run = 0.0
        self.jitter_log = []
        self.deadline_misses = 0

    def run(self, now: float):
        expected = self.last_run + self.interval_s
        jitter = now - expected if self.last_run > 0 else 0.0
        self.jitter_log.append(jitter)
        if jitter > self.interval_s:
            self.deadline_misses += 1
        self.callback(now, self.context)
        self.last_run = now

class RealTimeControlLoop:
    """Manages deterministic scheduling and execution of control tasks with sub-millisecond response times."""
    def __init__(self, min_cycle_us: int = 100):
        self.tasks: List[RealTimeControlTask] = []
        self.running = False
        self.min_cycle_us = min_cycle_us
        self.thread = None

    def register_task(self, task: RealTimeControlTask):
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: -t.priority)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None

    def _run_loop(self):
        last_times = {t.name: 0.0 for t in self.tasks}
        while self.running:
            now = time.perf_counter()
            for task in self.tasks:
                if now - last_times[task.name] >= task.interval_s:
                    task.run(now)
                    last_times[task.name] = now
            # Sleep for minimum cycle time (busy-wait for highest precision)
            next_wakeup = min([last_times[t.name] + t.interval_s for t in self.tasks])
            sleep_time = max(0.0, next_wakeup - time.perf_counter())
            if sleep_time > self.min_cycle_us * 1e-6:
                time.sleep(sleep_time)
            else:
                # Busy-wait for sub-millisecond precision
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < self.min_cycle_us * 1e-6:
                    pass

    def get_task_stats(self) -> List[Dict[str, Any]]:
        return [{
            'name': t.name,
            'interval_s': t.interval_s,
            'priority': t.priority,
            'deadline_misses': t.deadline_misses,
            'avg_jitter_us': 1e6 * np.mean(t.jitter_log) if t.jitter_log else 0.0
        } for t in self.tasks]
