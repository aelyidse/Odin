import numpy as np
from typing import List, Dict, Any, Optional, Tuple

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
    def __init__(self, max_missed: int = 5, confirm_age: int = 3, fusion_method: str = 'average', 
                 occlusion_threshold: float = 0.5, appearance_model: bool = True):
        self.trackers: List[KalmanTracker] = []
        self.max_missed = max_missed
        self.confirm_age = confirm_age
        self.fusion_method = fusion_method
        self.track_ids = []
        self.next_id = 1
        
        # Occlusion handling parameters
        self.occlusion_threshold = occlusion_threshold
        self.appearance_model = appearance_model
        self.appearance_features = {}  # Store appearance features for each track
        self.occlusion_status = {}     # Track occlusion status

    def associate(self, measurements: List[np.ndarray], 
                  appearance_features: Optional[List[np.ndarray]] = None) -> List[Optional[int]]:
        # Simple nearest-neighbor association with occlusion awareness
        associations = [None] * len(measurements)
        if not self.trackers:
            return associations
            
        # Calculate distance matrix
        dists = np.zeros((len(measurements), len(self.trackers)))
        for i, z in enumerate(measurements):
            for j, trk in enumerate(self.trackers):
                x, P = trk.get_state()
                
                # Calculate Mahalanobis distance for motion-based association
                innovation = z - trk.H @ x
                S = trk.H @ P @ trk.H.T + trk.R
                try:
                    motion_dist = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
                except np.linalg.LinAlgError:
                    motion_dist = np.linalg.norm(innovation)  # Fallback to Euclidean
                
                # Incorporate appearance similarity if available
                if self.appearance_model and appearance_features is not None:
                    if i < len(appearance_features) and self.track_ids[j] in self.appearance_features:
                        app_dist = 1.0 - self._appearance_similarity(
                            appearance_features[i], 
                            self.appearance_features[self.track_ids[j]]
                        )
                        # Combined distance (weighted sum)
                        dists[i, j] = 0.7 * motion_dist + 0.3 * app_dist
                    else:
                        dists[i, j] = motion_dist
                else:
                    dists[i, j] = motion_dist
                
                # Increase distance for occluded tracks
                if self.track_ids[j] in self.occlusion_status and self.occlusion_status[self.track_ids[j]]:
                    dists[i, j] *= 1.2  # Penalize occluded tracks slightly
        
        # Perform association with gating
        gating_threshold = 5.0  # Base gating threshold
        
        # First pass: associate high-confidence matches
        for i in range(len(measurements)):
            if np.min(dists[i, :]) < gating_threshold * 0.5:
                j = np.argmin(dists[i, :])
                associations[i] = j
                dists[:, j] = np.inf  # Prevent double assignment
        
        # Second pass: associate remaining measurements with relaxed gating
        for i in range(len(measurements)):
            if associations[i] is None and np.min(dists[i, :]) < gating_threshold:
                j = np.argmin(dists[i, :])
                associations[i] = j
                dists[:, j] = np.inf  # Prevent double assignment
                
        return associations

    def _appearance_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Calculate cosine similarity between appearance features"""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(feat1, feat2) / (norm1 * norm2))

    def _detect_occlusions(self, tracks: List[Dict[str, Any]]) -> Dict[int, bool]:
        """Detect potential occlusions between tracks"""
        occlusions = {}
        for i, track1 in enumerate(tracks):
            state1 = track1['state']
            pos1 = state1[:2] if len(state1) >= 2 else state1  # Extract position
            
            # Check for overlap with other tracks
            for j, track2 in enumerate(tracks):
                if i == j:
                    continue
                    
                state2 = track2['state']
                pos2 = state2[:2] if len(state2) >= 2 else state2
                
                # Calculate distance between tracks
                dist = np.linalg.norm(pos1 - pos2)
                
                # If tracks are very close, mark as potential occlusion
                if dist < self.occlusion_threshold:
                    occlusions[track1['id']] = True
                    occlusions[track2['id']] = True
                    
        return occlusions

    def step(self, measurements: List[np.ndarray], sensor_models: List[Dict[str, Any]], 
             appearance_features: Optional[List[np.ndarray]] = None):
        # Predict all tracks
        for trk in self.trackers:
            trk.predict()
            
        # Associate measurements
        associations = self.associate(measurements, appearance_features)
        assigned = set()
        
        # Update assigned tracks
        for i, j in enumerate(associations):
            if j is not None:
                trk = self.trackers[j]
                trk.update(measurements[i])
                assigned.add(j)
                
                # Update appearance model if available
                if self.appearance_model and appearance_features is not None and i < len(appearance_features):
                    self.appearance_features[self.track_ids[j]] = appearance_features[i]
                    
                # Clear occlusion status for this track
                if self.track_ids[j] in self.occlusion_status:
                    self.occlusion_status[self.track_ids[j]] = False
                    
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
                track_id = self.next_id
                self.track_ids.append(track_id)
                self.next_id += 1
                
                # Store appearance feature if available
                if self.appearance_model and appearance_features is not None and i < len(appearance_features):
                    self.appearance_features[track_id] = appearance_features[i]
                    
        # Get current tracks for occlusion detection
        current_tracks = self.get_tracks(confirmed_only=False)
        
        # Detect occlusions
        new_occlusions = self._detect_occlusions(current_tracks)
        self.occlusion_status.update(new_occlusions)
        
        # Prune dead tracks
        keep = []
        keep_ids = []
        for i, trk in enumerate(self.trackers):
            # Keep tracks with few misses or those in occlusion (more tolerance)
            track_id = self.track_ids[i]
            is_occluded = track_id in self.occlusion_status and self.occlusion_status[track_id]
            
            # Increase tolerance for occluded tracks
            max_misses = self.max_missed * 2 if is_occluded else self.max_missed
            
            if trk.missed < max_misses:
                keep.append(trk)
                keep_ids.append(track_id)
            else:
                # Clean up appearance features for deleted tracks
                if track_id in self.appearance_features:
                    del self.appearance_features[track_id]
                if track_id in self.occlusion_status:
                    del self.occlusion_status[track_id]
                    
        self.trackers = keep
        self.track_ids = keep_ids

    def get_tracks(self, confirmed_only: bool = True) -> List[Dict[str, Any]]:
        tracks = []
        for i, trk in enumerate(self.trackers):
            if not confirmed_only or trk.age >= self.confirm_age:
                x, P = trk.get_state()
                track_id = self.track_ids[i]
                is_occluded = track_id in self.occlusion_status and self.occlusion_status[track_id]
                
                tracks.append({
                    'id': track_id, 
                    'state': x, 
                    'cov': P, 
                    'age': trk.age, 
                    'missed': trk.missed,
                    'occluded': is_occluded
                })
        return tracks

    def fuse_tracks(self, sensor_measurements: List[List[np.ndarray]]) -> List[np.ndarray]:
        # Simple fusion: average all sensor measurements for each target
        fused = []
        for i in range(len(sensor_measurements[0])):
            vals = [sensor_measurements[s][i] for s in range(len(sensor_measurements))]
            fused.append(np.mean(vals, axis=0))
        return fused
