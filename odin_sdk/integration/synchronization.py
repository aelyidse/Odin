import threading
import time
from typing import Optional, Dict, Any, Callable

class DistributedBarrier:
    """Barrier for synchronizing distributed subsystem operations."""
    def __init__(self, parties: int, name: str = "barrier"):
        self.parties = parties
        self.name = name
        self.count = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.generation = 0

    def wait(self, timeout: Optional[float] = None) -> bool:
        with self.lock:
            gen = self.generation
            self.count += 1
            if self.count == self.parties:
                self.generation += 1
                self.count = 0
                self.condition.notify_all()
                return True
            else:
                start = time.time()
                while gen == self.generation:
                    remaining = None if timeout is None else max(0, timeout - (time.time() - start))
                    if remaining == 0:
                        return False
                    self.condition.wait(timeout=remaining)
                return True

class DistributedEvent:
    """Event for signaling between distributed components."""
    def __init__(self, name: str = "event"):
        self.name = name
        self._flag = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def set(self):
        with self._lock:
            self._flag = True
            self._condition.notify_all()

    def clear(self):
        with self._lock:
            self._flag = False

    def wait(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            start = time.time()
            while not self._flag:
                remaining = None if timeout is None else max(0, timeout - (time.time() - start))
                if remaining == 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

class DistributedClock:
    """Provides a synchronized clock reference for distributed components."""
    def __init__(self, time_source: Optional[Callable[[], float]] = None):
        self.time_source = time_source or time.time
        self.offset = 0.0
    def now(self) -> float:
        return self.time_source() + self.offset
    def synchronize(self, reference_time: float):
        local_time = self.time_source()
        self.offset = reference_time - local_time

# Example usage:
# barrier = DistributedBarrier(parties=3)
# event = DistributedEvent()
# clock = DistributedClock()
