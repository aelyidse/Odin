import numpy as np
from typing import Dict, Any, List, Optional

class PerformanceRecommendation:
    """Structured recommendation for system performance optimization."""
    def __init__(self, subsystem: str, recommendation: str, impact: str, confidence: float, details: Optional[Dict[str, Any]] = None):
        self.subsystem = subsystem
        self.recommendation = recommendation
        self.impact = impact  # e.g., 'increase power output', 'reduce jitter', 'improve thermal margin'
        self.confidence = confidence  # 0.0-1.0
        self.details = details or {}
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subsystem': self.subsystem,
            'recommendation': self.recommendation,
            'impact': self.impact,
            'confidence': self.confidence,
            'details': self.details
        }

class PerformanceOptimizationAdvisor:
    """Analyzes metrics and history to generate actionable optimization recommendations."""
    def __init__(self):
        self.recommendations: List[PerformanceRecommendation] = []

    def analyze(self, metrics: Dict[str, Any], twin_history: List[Any]):
        # Example: Check for thermal margin issues
        if 'thermal_margin' in metrics and metrics['thermal_margin']['mean'] < 10:
            self.recommendations.append(
                PerformanceRecommendation(
                    subsystem='thermal',
                    recommendation='Increase cooling system setpoint or reduce laser duty cycle.',
                    impact='Improve thermal margin',
                    confidence=0.9,
                    details={'mean_margin': metrics['thermal_margin']['mean']}
                )
            )
        # Example: Beam combining efficiency
        if 'beam_combining_efficiency' in metrics and metrics['beam_combining_efficiency']['mean'] < 0.85:
            self.recommendations.append(
                PerformanceRecommendation(
                    subsystem='beam_combining',
                    recommendation='Check alignment and cleanliness of optics; recalibrate spectral channels.',
                    impact='Increase beam combining efficiency',
                    confidence=0.8,
                    details={'mean_efficiency': metrics['beam_combining_efficiency']['mean']}
                )
            )
        # Example: Control loop jitter
        if 'control_loop_jitter' in metrics and metrics['control_loop_jitter']['max'] > 0.002:
            self.recommendations.append(
                PerformanceRecommendation(
                    subsystem='control',
                    recommendation='Optimize real-time task scheduling or upgrade controller hardware.',
                    impact='Reduce control loop jitter',
                    confidence=0.85,
                    details={'max_jitter': metrics['control_loop_jitter']['max']}
                )
            )
        # Add more rules or ML-based advisors as needed

    def get_recommendations(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.recommendations]

# Example usage:
# advisor = PerformanceOptimizationAdvisor()
# advisor.analyze(metrics, twin.get_history())
# print(advisor.get_recommendations())
