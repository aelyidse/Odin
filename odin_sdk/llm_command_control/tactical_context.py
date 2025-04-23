from typing import Dict, Any, Optional

class TacticalContextModel:
    """
    Models tactical context for mission directive interpretation.
    Incorporates mission phase, ROE, asset status, threat posture, and user-defined factors.
    Provides context-aware scoring and directive adaptation.
    """
    def __init__(self, mission_phase: str = 'default', roe: Optional[str] = None, asset_status: Optional[Dict[str, Any]] = None, threat_posture: Optional[str] = None, custom_factors: Optional[Dict[str, Any]] = None):
        self.mission_phase = mission_phase
        self.roe = roe
        self.asset_status = asset_status or {}
        self.threat_posture = threat_posture
        self.custom_factors = custom_factors or {}
    def as_dict(self) -> Dict[str, Any]:
        return {
            'mission_phase': self.mission_phase,
            'roe': self.roe,
            'asset_status': self.asset_status,
            'threat_posture': self.threat_posture,
            'custom_factors': self.custom_factors
        }
    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def score_directive(self, directive: Dict[str, Any]) -> float:
        """
        Score or adapt a mission directive based on the current tactical context.
        Returns a context-weighted score (higher = more aligned).
        """
        score = 1.0
        # Example: escalate priority during critical phases
        if self.mission_phase in ['engagement', 'egress']:
            score *= 1.2
        if self.roe == 'tight' and directive.get('action') in ['engage', 'destroy']:
            score *= 0.8  # Restrict aggressive actions
        if self.asset_status.get('critical', False):
            score *= 1.5
        # Custom user factors
        for k, v in self.custom_factors.items():
            if k in directive and directive[k] == v:
                score *= 1.1
        return score
    def adapt_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a parsed mission directive for tactical context (e.g., adjust priority, restrict actions).
        Returns a modified directive.
        """
        new_dir = directive.copy()
        # Example: automatically downgrade priority if ROE is tight
        if self.roe == 'tight' and new_dir.get('parameters', {}).get('priority') == 'high':
            new_dir['parameters']['priority'] = 'medium'
        # Example: escalate if asset is critical
        if self.asset_status.get('critical', False):
            new_dir['parameters']['priority'] = 'urgent'
        # Allow user-defined adaptation
        for k, v in self.custom_factors.items():
            if k in new_dir['parameters']:
                new_dir['parameters'][k] = v
        return new_dir

# Example usage:
# ctx = TacticalContextModel(mission_phase='engagement', roe='tight', asset_status={'critical': True})
# directive = {'action': 'engage', 'parameters': {'priority': 'high'}}
# print(ctx.score_directive(directive))
# print(ctx.adapt_directive(directive))
