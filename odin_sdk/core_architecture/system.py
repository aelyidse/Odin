from typing import Dict, List, Optional
from .registry import ComponentRegistry
from .component import Component

class System:
    """Main system class for managing components and their interactions in ODIN SDK."""
    
    def __init__(self, name: str):
        self._name = name
        self._registry = ComponentRegistry()
        self._dependencies: Dict[str, List[str]] = {}
        self._initialized = False
        
    @property
    def name(self) -> str:
        """Get the system's name."""
        return self._name
        
    @property
    def registry(self) -> ComponentRegistry:
        """Get the component registry."""
        return self._registry
        
    def add_dependency(self, component_name: str, dependency_names: List[str]) -> None:
        """Add dependencies for a component.
        
        Args:
            component_name: Name of the component
            dependency_names: List of component instance names this component depends on
        """
        self._dependencies[component_name] = dependency_names
        
    def initialize(self) -> None:
        """Initialize all components in dependency order."""
        if self._initialized:
            return
            
        initialized = set()
        to_initialize = list(self._dependencies.keys())
        
        while to_initialize:
            component_name = to_initialize[0]
            deps = self._dependencies.get(component_name, [])
            
            if all(dep in initialized for dep in deps):
                component = self._registry.get_component(component_name)
                if component and not component.is_initialized():
                    component.initialize()
                initialized.add(component_name)
                to_initialize.pop(0)
            else:
                to_initialize.append(to_initialize.pop(0))
                
        self._initialized = True
        
    def shutdown(self) -> None:
        """Shutdown all components in reverse dependency order."""
        if not self._initialized:
            return
            
        shutdown_order = list(reversed(list(self._dependencies.keys())))
        for component_name in shutdown_order:
            component = self._registry.get_component(component_name)
            if component:
                component.shutdown()
                
        self._initialized = False
