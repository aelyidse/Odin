from typing import List, Dict, Any, Callable
import numpy as np

class ThreatPrioritizationEngine:
    """
    Engine to rank/prioritize threats based on configurable parameters, scoring functions, and mission context awareness.
    Mission context can include rules of engagement, asset protection, mission phase, and user-defined context modifiers.
    """
    def __init__(self, config: Dict[str, Any] = None, mission_context: Dict[str, Any] = None):
        # Default weights and rules
        default_config = {
            'proximity_weight': 1.0,
            'velocity_weight': 0.5,
            'type_weight': 1.0,
            'engagement_weight': 0.7,
            'custom_score_fn': None,  # Optional user-defined scoring function
            'type_priority': {'missile': 3, 'aircraft': 2, 'uav': 1, 'unknown': 0},
            'max_range_m': 50000.0,
            'max_velocity_mps': 1500.0,
            'context_modifiers': None  # Optional function or dict for mission context
        }
        self.config = default_config
        if config:
            self.config.update(config)
        self.mission_context = mission_context or {}

    def score_threat(self, track: Dict[str, Any]) -> float:
        cfg = self.config
        ctx = self.mission_context
        # Proximity: inverse of normalized range
        range_score = 1.0 - min(track.get('range_m', cfg['max_range_m']) / cfg['max_range_m'], 1.0)
        # Velocity: normalized closing rate
        velocity_score = min(abs(track.get('closing_velocity_mps', 0)) / cfg['max_velocity_mps'], 1.0)
        # Type: mapped to priority
        ttype = track.get('type', 'unknown').lower()
        type_score = cfg['type_priority'].get(ttype, 0) / max(cfg['type_priority'].values())
        # Engagement status: 1 if engaged, 0 otherwise
        engagement_score = 1.0 if track.get('engaged', False) else 0.0
        # Weighted sum
        score = (
            cfg['proximity_weight'] * range_score +
            cfg['velocity_weight'] * velocity_score +
            cfg['type_weight'] * type_score +
            cfg['engagement_weight'] * engagement_score
        )
        # Mission context awareness
        # Example: escalate threats near high-value asset, or during critical mission phase
        if 'asset_priority' in ctx and 'asset' in track:
            asset = track['asset']
            score *= ctx['asset_priority'].get(asset, 1.0)
        if 'mission_phase' in ctx and 'phase_modifiers' in ctx:
            phase = ctx['mission_phase']
            score *= ctx['phase_modifiers'].get(phase, 1.0)
        # User-defined context modifiers (function or dict)
        if cfg.get('context_modifiers'):
            if callable(cfg['context_modifiers']):
                score = cfg['context_modifiers'](track, score, ctx)
            elif isinstance(cfg['context_modifiers'], dict):
                for k, v in cfg['context_modifiers'].items():
                    if track.get(k) in v:
                        score *= v[track[k]]
        # Custom scoring function (if provided)
        if cfg['custom_score_fn']:
            score = cfg['custom_score_fn'](track, score, ctx)
        return float(score)

    def prioritize(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Score all threats
        for t in tracks:
            t['priority_score'] = self.score_threat(t)
        # Sort by descending score
        return sorted(tracks, key=lambda x: x['priority_score'], reverse=True)

    def update_config(self, new_config: Dict[str, Any]):
        self.config.update(new_config)

    def update_mission_context(self, new_context: Dict[str, Any]):
        self.mission_context.update(new_context)
