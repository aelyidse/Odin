from typing import Callable, List, Dict, Any, Optional
import time

class FaultEvent:
    """Represents a detected fault with metadata and timestamp."""
    def __init__(self, subsystem: str, fault_type: str, details: Optional[Dict[str, Any]] = None):
        self.timestamp = time.time()
        self.subsystem = subsystem
        self.fault_type = fault_type
        self.details = details or {}

class FaultDetectionEngine:
    """Monitors subsystem health, detects faults, and triggers recovery protocols."""
    def __init__(self):
        self.monitors: List[Callable[[], Optional[FaultEvent]]] = []
        self.fault_log: List[FaultEvent] = []
        self.recovery_callbacks: Dict[str, Callable[[FaultEvent], None]] = {}

    def register_monitor(self, monitor_fn: Callable[[], Optional[FaultEvent]]):
        self.monitors.append(monitor_fn)

    def register_recovery(self, fault_type: str, callback: Callable[[FaultEvent], None]):
        self.recovery_callbacks[fault_type] = callback

    def check(self):
        for monitor in self.monitors:
            event = monitor()
            if event:
                self.fault_log.append(event)
                self.trigger_recovery(event)

    def trigger_recovery(self, event: FaultEvent):
        cb = self.recovery_callbacks.get(event.fault_type)
        if cb:
            cb(event)
        else:
            # Default: log and escalate
            print(f"[ALERT] Unhandled fault: {event.fault_type} in {event.subsystem} at {event.timestamp}")

    def get_fault_history(self) -> List[FaultEvent]:
        return self.fault_log

    def clear_faults(self):
        self.fault_log.clear()

# Example monitor and recovery usage
# def temp_monitor():
#     if overtemp_condition:
#         return FaultEvent('thermal', 'overtemp', {'measured': temp})
#     return None
# def thermal_recovery(event):
#     shutdown_cooler()
#     log_event(event)
# engine = FaultDetectionEngine()
# engine.register_monitor(temp_monitor)
# engine.register_recovery('overtemp', thermal_recovery)
