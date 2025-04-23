import numpy as np
from typing import List, Dict, Any, Optional

class AtmosphericLayer:
    """Represents a single atmospheric layer with variable refractive index and time-of-day effects."""
    def __init__(self, z_min: float, z_max: float, temperature_K: float, pressure_Pa: float, humidity_pct: float, wavelength_nm: float, hour: Optional[float] = None):
        self.z_min = z_min  # Lower altitude [m]
        self.z_max = z_max  # Upper altitude [m]
        self.temperature_K = temperature_K
        self.pressure_Pa = pressure_Pa
        self.humidity_pct = humidity_pct
        self.wavelength_nm = wavelength_nm
        self.hour = hour
        self.apply_time_of_day_effects()
        self.n = self.calculate_refractive_index()
    def apply_time_of_day_effects(self):
        """
        Adjust temperature and humidity for time-of-day (hour in 0-24).
        Daytime heats surface, nighttime cools. Humidity may rise at night.
        """
        if self.hour is None or self.z_min > 2000:  # Only apply to lower layers
            return
        # Simple diurnal cycle: max temp at 15:00, min at 05:00
        # Amplitude decreases with altitude
        base_amp = 8.0 * np.exp(-self.z_min / 2000.0)  # K
        hour_angle = ((self.hour - 15) % 24) / 24.0 * 2 * np.pi
        temp_offset = -base_amp * np.cos(hour_angle)
        self.temperature_K += temp_offset
        # Humidity: higher at night, lower during day
        hum_amp = 15.0 * np.exp(-self.z_min / 2000.0)
        hum_offset = hum_amp * np.cos(hour_angle)
        self.humidity_pct = np.clip(self.humidity_pct + hum_offset, 0, 100)

    def calculate_refractive_index(self) -> float:
        # Edlén's formula for standard air, extended for humidity
        # n - 1 = (k1*P/T)*(1 + k2*P/T) - (k3*e/T)
        # P: pressure [Pa], T: temperature [K], e: vapor pressure [Pa]
        # Constants for visible/near-IR (approximate)
        k1 = 0.0000834254
        k2 = 0.000000240614
        k3 = 0.00000001358
        P = self.pressure_Pa / 100.0  # Convert Pa to hPa
        T = self.temperature_K
        # Approximate vapor pressure from humidity (Magnus formula)
        T_C = T - 273.15
        e_s = 6.1094 * np.exp(17.625 * T_C / (T_C + 243.04))  # hPa
        e = self.humidity_pct / 100.0 * e_s
        n_minus_1 = (k1 * P / T) * (1 + k2 * P / T) - (k3 * e / T)
        return 1.0 + n_minus_1
    def as_dict(self) -> Dict[str, Any]:
        return {
            'z_min': self.z_min,
            'z_max': self.z_max,
            'temperature_K': self.temperature_K,
            'pressure_Pa': self.pressure_Pa,
            'humidity_pct': self.humidity_pct,
            'wavelength_nm': self.wavelength_nm,
            'refractive_index': self.n
        }

class MultiLayerAtmosphere:
    """Models a multi-layered atmosphere with variable refractive indices."""
    def __init__(self, layers: List[AtmosphericLayer]):
        self.layers = layers
    @classmethod
    def standard_profile(cls, n_layers: int = 10, z_max: float = 20000.0, wavelength_nm: float = 1064.0, hour: Optional[float] = None) -> 'MultiLayerAtmosphere':
        # Standard atmosphere: temperature/pressure/humidity decrease with altitude, modulated by time-of-day
        z_edges = np.linspace(0, z_max, n_layers+1)
        layers = []
        for i in range(n_layers):
            z0, z1 = z_edges[i], z_edges[i+1]
            # ISA lapse rate, approximate
            T0 = 288.15 - 0.0065 * ((z0 + z1)/2)
            P0 = 101325 * (T0/288.15)**5.2561
            H0 = max(0, 80 - 0.003 * ((z0 + z1)/2))
            layers.append(AtmosphericLayer(z0, z1, T0, P0, H0, wavelength_nm, hour=hour))
        return cls(layers)

    def refractive_index_at(self, z: float) -> float:
        for layer in self.layers:
            if layer.z_min <= z < layer.z_max:
                return layer.n
        return self.layers[-1].n  # Above highest layer
    def as_dict(self) -> List[Dict[str, Any]]:
        return [layer.as_dict() for layer in self.layers]

# Example usage:
# atmosphere = MultiLayerAtmosphere.standard_profile(n_layers=20, z_max=20000, wavelength_nm=1064.0)
# print(atmosphere.as_dict())
# print(atmosphere.refractive_index_at(5000.0))
