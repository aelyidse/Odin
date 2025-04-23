import numpy as np
from typing import List, Dict, Any, Optional

class KalmanTracker:
    """Single-target Kalman filter tracker."""
    def __init__(self, x0: np.ndarray, P0: np.ndarray, F: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray):
        self.x = x0.copy()
        self.P = P0.copy()
        self.F = F.copy()
        self.Q = Q.copy()
        self.H = H.copy()
        self.R = R.copy()
        self.history = []
        self.age = 0
        self.missed = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.missed = 0
        self.history.append(self.x.copy())

    def miss(self):
        self.missed += 1
        self.history.append(self.x.copy())

    def get_state(self):
        return self.x.copy(), self.P.copy()

class MultiTargetTracker:
    """Multi-target tracker with data fusion and track management."""
    def __init__(self, max_missed: int = 5, confirm_age: int = 3, fusion_method: str = 'average'):
        self.trackers: List[KalmanTracker] = []
        self.max_missed = max_missed
        self.confirm_age = confirm_age
        self.fusion_method = fusion_method
        self.track_ids = []
        self.next_id = 1

    def associate(self, measurements: List[np.ndarray]) -> List[Optional[int]]:
        # Simple nearest-neighbor association (can be replaced with JPDA or MHT)
        associations = [None] * len(measurements)
        if not self.trackers:
            return associations
        dists = np.zeros((len(measurements), len(self.trackers)))
        for i, z in enumerate(measurements):
            for j, trk in enumerate(self.trackers):
                x, _ = trk.get_state()
                dists[i, j] = np.linalg.norm(z - trk.H @ x)
        for i in range(len(measurements)):
            j = np.argmin(dists[i])
            if dists[i, j] < 5.0:  # gating threshold
                associations[i] = j
                dists[:, j] = np.inf  # prevent double assignment
        return associations

    def step(self, measurements: List[np.ndarray], sensor_models: List[Dict[str, Any]]):
        # Predict all tracks
        for trk in self.trackers:
            trk.predict()
        # Associate measurements
        associations = self.associate(measurements)
        assigned = set()
        # Update assigned tracks
        for i, j in enumerate(associations):
            if j is not None:
                trk = self.trackers[j]
                trk.update(measurements[i])
                assigned.add(j)
        # Missed tracks
        for idx, trk in enumerate(self.trackers):
            if idx not in assigned:
                trk.miss()
        # Initiate new tracks for unassigned measurements
        for i, j in enumerate(associations):
            if j is None:
                # Use first sensor model for new track
                model = sensor_models[0]
                x0 = np.zeros(model['F'].shape[0])
                x0[:len(measurements[i])] = measurements[i]
                tracker = KalmanTracker(x0, model['P0'], model['F'], model['Q'], model['H'], model['R'])
                self.trackers.append(tracker)
                self.track_ids.append(self.next_id)
                self.next_id += 1
        # Prune dead tracks
        keep = []
        keep_ids = []
        for i, trk in enumerate(self.trackers):
            if trk.missed < self.max_missed:
                keep.append(trk)
                keep_ids.append(self.track_ids[i])
        self.trackers = keep
        self.track_ids = keep_ids

    def get_tracks(self, confirmed_only: bool = True) -> List[Dict[str, Any]]:
        tracks = []
        for i, trk in enumerate(self.trackers):
            if not confirmed_only or trk.age >= self.confirm_age:
                x, P = trk.get_state()
                tracks.append({'id': self.track_ids[i], 'state': x, 'cov': P, 'age': trk.age, 'missed': trk.missed})
        return tracks

    def fuse_tracks(self, sensor_measurements: List[List[np.ndarray]]) -> List[np.ndarray]:
        # Simple fusion: average all sensor measurements for each target
        fused = []
        for i in range(len(sensor_measurements[0])):
            vals = [sensor_measurements[s][i] for s in range(len(sensor_measurements))]
            fused.append(np.mean(vals, axis=0))
        return fused
