from typing import Any, Dict, Type, Optional, Callable
from .component import Component

class DependencyContainer:
    """Dependency injection container for ODIN SDK components."""
    def __init__(self):
        self._providers: Dict[str, Callable[..., Component]] = {}
        self._instances: Dict[str, Component] = {}
        self._dependency_graph: Dict[str, list] = {}

    def register(self, name: str, provider: Callable[..., Component], dependencies: Optional[list] = None) -> None:
        """Register a component provider with optional dependencies.
        Args:
            name: Unique name for the component
            provider: Callable that returns a component instance
            dependencies: List of dependency names
        """
        self._providers[name] = provider
        self._dependency_graph[name] = dependencies or []

    def resolve(self, name: str) -> Component:
        """Resolve and instantiate a component, injecting dependencies recursively."""
        if name in self._instances:
            return self._instances[name]
        if name not in self._providers:
            raise ValueError(f"No provider registered for component '{name}'")
        # Resolve dependencies first
        deps = [self.resolve(dep_name) for dep_name in self._dependency_graph.get(name, [])]
        instance = self._providers[name](*deps)
        self._instances[name] = instance
        return instance

    def inject_setters(self, name: str, setters: Dict[str, str]) -> None:
        """Inject dependencies via setter methods after construction.
        Args:
            name: Name of the component instance
            setters: Dict mapping setter method names to registered dependency names
        """
        instance = self.resolve(name)
        for setter_method, dep_name in setters.items():
            dep_instance = self.resolve(dep_name)
            setter = getattr(instance, setter_method, None)
            if callable(setter):
                setter(dep_instance)
            else:
                raise AttributeError(f"{instance.__class__.__name__} has no method {setter_method}")

    def clear(self) -> None:
        """Clear all registered providers and instances."""
        self._providers.clear()
        self._instances.clear()
        self._dependency_graph.clear()
