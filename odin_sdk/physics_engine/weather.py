import numpy as np
from typing import Dict, Any, Optional

class WeatherState:
    """
    Represents the physical state of the weather at a given time and location.
    Includes temperature, humidity, pressure, wind, and precipitation.
    """
    def __init__(self, temperature_K: float, humidity_pct: float, pressure_Pa: float, wind_mps: float, wind_dir_deg: float, precipitation: Optional[str] = None):
        self.temperature_K = temperature_K
        self.humidity_pct = humidity_pct
        self.pressure_Pa = pressure_Pa
        self.wind_mps = wind_mps
        self.wind_dir_deg = wind_dir_deg
        self.precipitation = precipitation  # e.g., 'none', 'rain', 'snow'
    def as_dict(self) -> Dict[str, Any]:
        return {
            'temperature_K': self.temperature_K,
            'humidity_pct': self.humidity_pct,
            'pressure_Pa': self.pressure_Pa,
            'wind_mps': self.wind_mps,
            'wind_dir_deg': self.wind_dir_deg,
            'precipitation': self.precipitation
        }

class WeatherPattern:
    """
    Evolves weather state over time with physical consistency (conservation, advection, diurnal forcing).
    Supports simple 2D grid or single-point evolution.
    """
    def __init__(self, grid_shape=(1,1), initial_state: Optional[WeatherState] = None):
        self.grid_shape = grid_shape
        self.state_grid = np.empty(grid_shape, dtype=object)
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                self.state_grid[i,j] = initial_state or WeatherState(288.15, 50.0, 101325.0, 2.0, 90.0, 'none')
    def step(self, dt: float = 3600.0, solar_hour: Optional[float] = None):
        """
        Advance weather by dt seconds with physical consistency.
        Includes diurnal cycle, wind advection, and precipitation triggers.
        """
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                state: WeatherState = self.state_grid[i,j]
                # Diurnal temperature forcing
                if solar_hour is not None:
                    amp = 8.0
                    hour_angle = ((solar_hour - 15) % 24) / 24.0 * 2 * np.pi
                    temp_offset = -amp * np.cos(hour_angle)
                    state.temperature_K += temp_offset * (dt/86400.0)
                    # Humidity inverse to temp
                    state.humidity_pct = np.clip(state.humidity_pct + amp * np.cos(hour_angle) * (dt/86400.0), 0, 100)
                # Wind advection (simple: move state downwind)
                if self.grid_shape != (1,1):
                    di = int(np.round(np.sin(np.deg2rad(state.wind_dir_deg))))
                    dj = int(np.round(np.cos(np.deg2rad(state.wind_dir_deg))))
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.grid_shape[0] and 0 <= nj < self.grid_shape[1]:
                        neighbor: WeatherState = self.state_grid[ni, nj]
                        # Exchange air mass (simple average)
                        for attr in ['temperature_K','humidity_pct','pressure_Pa']:
                            avg = 0.5 * (getattr(state, attr) + getattr(neighbor, attr))
                            setattr(state, attr, avg)
                            setattr(neighbor, attr, avg)
                # Precipitation trigger
                if state.humidity_pct > 95 and state.temperature_K < 273.15:
                    state.precipitation = 'snow'
                elif state.humidity_pct > 95:
                    state.precipitation = 'rain'
                else:
                    state.precipitation = 'none'
    def get_state(self, i=0, j=0) -> WeatherState:
        return self.state_grid[i, j]
    def to_dict(self) -> Dict[str, Any]:
        return {
            'grid_shape': self.grid_shape,
            'states': [[self.state_grid[i,j].as_dict() for j in range(self.grid_shape[1])] for i in range(self.grid_shape[0])]
        }

# Example usage:
# pattern = WeatherPattern(grid_shape=(10,10))
# for t in range(24):
#     pattern.step(dt=3600, solar_hour=t)
# print(pattern.to_dict())
