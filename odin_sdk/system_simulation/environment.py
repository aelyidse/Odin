import numpy as np
from typing import Dict, Any, Optional, Tuple

class WeatherModel:
    """Models weather conditions for system simulation."""
    def __init__(self, visibility_km: float = 20.0, humidity_pct: float = 50.0, temperature_C: float = 20.0,
                 wind_speed_mps: float = 0.0, wind_dir_deg: float = 0.0, precipitation: str = 'none'):
        self.visibility_km = visibility_km
        self.humidity_pct = humidity_pct
        self.temperature_C = temperature_C
        self.wind_speed_mps = wind_speed_mps
        self.wind_dir_deg = wind_dir_deg
        self.precipitation = precipitation  # 'none', 'rain', 'snow', etc.
    def as_dict(self) -> Dict[str, Any]:
        return {
            'visibility_km': self.visibility_km,
            'humidity_pct': self.humidity_pct,
            'temperature_C': self.temperature_C,
            'wind_speed_mps': self.wind_speed_mps,
            'wind_dir_deg': self.wind_dir_deg,
            'precipitation': self.precipitation
        }

class TerrainModel:
    """Models terrain for system simulation (elevation, cover, reflectivity)."""
    def __init__(self, grid: np.ndarray, grid_spacing_m: float, origin: Tuple[float, float] = (0.0, 0.0),
                 reflectivity: Optional[np.ndarray] = None, cover_map: Optional[np.ndarray] = None):
        self.grid = grid  # 2D elevation map (meters)
        self.grid_spacing_m = grid_spacing_m
        self.origin = origin  # (lat, lon) or (x, y)
        self.reflectivity = reflectivity if reflectivity is not None else np.ones_like(grid)
        self.cover_map = cover_map if cover_map is not None else np.zeros_like(grid)
    def elevation(self, x_idx: int, y_idx: int) -> float:
        return float(self.grid[y_idx, x_idx])
    def get_patch(self, x0: int, y0: int, dx: int, dy: int) -> np.ndarray:
        return self.grid[y0:y0+dy, x0:x0+dx]
    def as_dict(self) -> Dict[str, Any]:
        return {
            'grid': self.grid.tolist(),
            'grid_spacing_m': self.grid_spacing_m,
            'origin': self.origin,
            'reflectivity': self.reflectivity.tolist(),
            'cover_map': self.cover_map.tolist()
        }

class EnvironmentalConditionModel:
    """Combines weather and terrain for full environmental context."""
    def __init__(self, weather: WeatherModel, terrain: TerrainModel):
        self.weather = weather
        self.terrain = terrain
    def as_dict(self) -> Dict[str, Any]:
        return {
            'weather': self.weather.as_dict(),
            'terrain': self.terrain.as_dict()
        }

# Example usage:
# grid = np.random.uniform(0, 100, (100, 100))
# terrain = TerrainModel(grid, grid_spacing_m=10.0)
# weather = WeatherModel(visibility_km=15, humidity_pct=60, temperature_C=25, wind_speed_mps=5, precipitation='rain')
# env = EnvironmentalConditionModel(weather, terrain)
# print(env.as_dict())
