"""Core architecture package for ODIN SDK.

This package provides the foundational framework for building modular components
with typed interfaces for the ODIN directed energy weapon system SDK.
"""

from .component import Component
from .registry import ComponentRegistry
from .system import System
from .subsystems import LaserUnit, BeamCombining, ControlSystem
from .di import DependencyContainer

__all__ = ['Component', 'ComponentRegistry', 'System', 'LaserUnit', 'BeamCombining', 'ControlSystem', 'DependencyContainer']
