from typing import Dict, Type, Optional
from .component import Component

class ComponentRegistry:
    """Registry for managing component types and instances in ODIN SDK."""
    
    def __init__(self):
        self._component_types: Dict[str, Type[Component]] = {}
        self._instances: Dict[str, Component] = {}
        
    def register_component_type(self, component_type: Type[Component], name: Optional[str] = None) -> None:
        """Register a component type with an optional custom name.
        
        Args:
            component_type: The component class to register
            name: Optional name to register the component under, defaults to class name
        """
        component_name = name or component_type.__name__
        self._component_types[component_name] = component_type
        
    def create_component(self, component_name: str, instance_name: str, config: Optional[dict] = None) -> Component:
        """Create a component instance from a registered type.
        
        Args:
            component_name: Name of the registered component type
            instance_name: Unique name for this instance
            config: Optional configuration dictionary
        Returns:
            Component instance
        Raises:
            KeyError: If component_name is not registered
        """
        if component_name not in self._component_types:
            raise KeyError(f"Component type {component_name} not registered")
            
        component = self._component_types[component_name](instance_name, config)
        self._instances[instance_name] = component
        return component
        
    def get_component(self, instance_name: str) -> Optional[Component]:
        """Get a component instance by name.
        
        Args:
            instance_name: Name of the component instance
        Returns:
            Component instance if found, None otherwise
        """
        return self._instances.get(instance_name)
        
    def get_registered_types(self) -> Dict[str, Type[Component]]:
        """Get all registered component types."""
        return self._component_types.copy()
