from typing import Dict, Any, List, Optional, Protocol

class SubsystemMessage(Protocol):
    """Standardized message format for subsystem communication."""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubsystemMessage': ...

class LaserStatus:
    def __init__(self, power_w: float, wavelength_nm: float, state: str, timestamp: float):
        self.power_w = power_w
        self.wavelength_nm = wavelength_nm
        self.state = state
        self.timestamp = timestamp
    def to_dict(self) -> Dict[str, Any]:
        return {
            'power_w': self.power_w,
            'wavelength_nm': self.wavelength_nm,
            'state': self.state,
            'timestamp': self.timestamp
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LaserStatus':
        return cls(
            power_w=data['power_w'],
            wavelength_nm=data['wavelength_nm'],
            state=data['state'],
            timestamp=data['timestamp']
        )

class TargetTrack:
    def __init__(self, track_id: int, position_m: List[float], velocity_mps: List[float], type: str, priority: float, timestamp: float):
        self.track_id = track_id
        self.position_m = position_m
        self.velocity_mps = velocity_mps
        self.type = type
        self.priority = priority
        self.timestamp = timestamp
    def to_dict(self) -> Dict[str, Any]:
        return {
            'track_id': self.track_id,
            'position_m': self.position_m,
            'velocity_mps': self.velocity_mps,
            'type': self.type,
            'priority': self.priority,
            'timestamp': self.timestamp
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TargetTrack':
        return cls(
            track_id=data['track_id'],
            position_m=data['position_m'],
            velocity_mps=data['velocity_mps'],
            type=data['type'],
            priority=data['priority'],
            timestamp=data['timestamp']
        )

class BeamCommand:
    def __init__(self, beam_id: int, target_id: int, power_w: float, dwell_time_s: float, timestamp: float):
        self.beam_id = beam_id
        self.target_id = target_id
        self.power_w = power_w
        self.dwell_time_s = dwell_time_s
        self.timestamp = timestamp
    def to_dict(self) -> Dict[str, Any]:
        return {
            'beam_id': self.beam_id,
            'target_id': self.target_id,
            'power_w': self.power_w,
            'dwell_time_s': self.dwell_time_s,
            'timestamp': self.timestamp
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeamCommand':
        return cls(
            beam_id=data['beam_id'],
            target_id=data['target_id'],
            power_w=data['power_w'],
            dwell_time_s=data['dwell_time_s'],
            timestamp=data['timestamp']
        )

class HealthStatus:
    def __init__(self, subsystem: str, status: str, details: Optional[Dict[str, Any]], timestamp: float):
        self.subsystem = subsystem
        self.status = status
        self.details = details or {}
        self.timestamp = timestamp
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subsystem': self.subsystem,
            'status': self.status,
            'details': self.details,
            'timestamp': self.timestamp
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthStatus':
        return cls(
            subsystem=data['subsystem'],
            status=data['status'],
            details=data.get('details', {}),
            timestamp=data['timestamp']
        )

# Example usage: all subsystems should use these classes for data exchange
# msg = LaserStatus(power_w=1000, wavelength_nm=1064, state='ON', timestamp=time.time())
# serialized = msg.to_dict()
# deserialized = LaserStatus.from_dict(serialized)
