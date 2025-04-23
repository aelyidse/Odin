from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

T = TypeVar('T')

class Component(ABC, Generic[T]):
    """Abstract base class for all ODIN SDK components with typed interfaces."""
    
    def __init__(self, name: str, config: Optional[dict] = None):
        """Initialize a component with a name and optional configuration.
        
        Args:
            name: Unique identifier for the component
            config: Optional configuration dictionary
        """
        self._name = name
        self._config = config or {}
        self._is_initialized = False
        
    @property
    def name(self) -> str:
        """Get the component's name."""
        return self._name
        
    @property
    def config(self) -> dict:
        """Get the component's configuration."""
        return self._config
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component's resources and state."""
        pass
        
    @abstractmethod
    def process(self, data: T) -> T:
        """Process input data and return processed output.
        
        Args:
            data: Input data of generic type T
        Returns:
            Processed data of generic type T
        """
        pass
        
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources and shutdown the component."""
        pass
        
    def is_initialized(self) -> bool:
        """Check if the component has been initialized."""
        return self._is_initialized
