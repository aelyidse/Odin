import json
from typing import Dict, Any, Optional, List

class ConfigValidationError(Exception):
    pass

class ConfigSchema:
    """Defines required fields, types, and constraints for configuration validation."""
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate(self, config: Dict[str, Any], prefix: str = ""):
        for key, rules in self.schema.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if rules.get('required', True) and key not in config:
                raise ConfigValidationError(f"Missing required config key: {full_key}")
            if key in config:
                value = config[key]
                expected_type = rules.get('type')
                if expected_type and not isinstance(value, expected_type):
                    raise ConfigValidationError(f"Config key {full_key} must be of type {expected_type.__name__}")
                if 'enum' in rules and value not in rules['enum']:
                    raise ConfigValidationError(f"Config key {full_key} must be one of {rules['enum']}")
                if 'min' in rules and value < rules['min']:
                    raise ConfigValidationError(f"Config key {full_key} below minimum {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    raise ConfigValidationError(f"Config key {full_key} above maximum {rules['max']}")
                if 'schema' in rules:
                    self.validate(value, prefix=full_key)

class ConfigManager:
    """Manages system-wide configuration with validation and runtime access."""
    def __init__(self, schema: Optional[ConfigSchema] = None):
        self._config: Dict[str, Any] = {}
        self._schema = schema

    def load(self, path: str):
        with open(path, 'r') as f:
            cfg = json.load(f)
        if self._schema:
            self._schema.validate(cfg)
        self._config = cfg

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        self._config[key] = value
        if self._schema:
            self._schema.validate(self._config)

    def all(self) -> Dict[str, Any]:
        return dict(self._config)

# Example schema and usage:
# schema = ConfigSchema({
#     'laser': {'type': dict, 'required': True, 'schema': {
#         'power_w': {'type': float, 'min': 0, 'max': 10000},
#         'wavelength_nm': {'type': float, 'min': 900, 'max': 1100}
#     }},
#     'engagement_mode': {'type': str, 'enum': ['auto', 'manual']}
# })
# mgr = ConfigManager(schema)
# mgr.load('odin_config.json')
# mgr.set('laser', {'power_w': 1200, 'wavelength_nm': 1064})
