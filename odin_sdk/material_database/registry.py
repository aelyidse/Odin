from typing import Dict, Optional, Union
import json
import numpy as np
from .material import OpticalMaterial, TabulatedOpticalMaterial

class MaterialRegistry:
    """Concrete registry for custom material definition and importation."""
    def __init__(self):
        self._materials: Dict[str, OpticalMaterial] = {}

    def register(self, material: OpticalMaterial) -> None:
        """Register a new material object."""
        if material.name in self._materials:
            raise ValueError(f"Material '{material.name}' already registered.")
        self._materials[material.name] = material

    def get(self, name: str) -> Optional[OpticalMaterial]:
        """Retrieve a material by name."""
        return self._materials.get(name)

    def list_materials(self) -> Dict[str, OpticalMaterial]:
        """List all registered materials."""
        return dict(self._materials)

    def import_from_json(self, filepath: str) -> None:
        """Import material(s) from a JSON file (analytic or tabulated)."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'name' in data:
            self.register(self._from_dict(data))
        elif isinstance(data, list):
            for entry in data:
                self.register(self._from_dict(entry))
        else:
            raise ValueError('Invalid material JSON format.')

    def import_from_csv(self, filepath: str) -> None:
        """Import tabulated material from CSV file (wavelength, temp, n, alpha, k, c)."""
        arr = np.genfromtxt(filepath, delimiter=',', names=True)
        wavelengths = np.unique(arr['wavelength'])
        temps = np.unique(arr['temp'])
        n_table = arr['n'].reshape((len(wavelengths), len(temps)))
        a_table = arr['alpha'].reshape((len(wavelengths), len(temps)))
        k_table = arr['k'].reshape((len(temps),))
        c_table = arr['c'].reshape((len(temps),))
        mat = TabulatedOpticalMaterial(
            name=filepath,
            refractive_index_table=n_table,
            absorption_coeff_table=a_table,
            wavelengths=wavelengths,
            temperatures=temps,
            thermal_conductivity_table=k_table,
            specific_heat_table=c_table,
            density=float(np.mean(arr['density'])),
            metadata={'source': filepath}
        )
        self.register(mat)

    def export_to_json(self, name: str, filepath: str) -> None:
        """Export a registered material to JSON."""
        mat = self.get(name)
        if mat is None:
            raise ValueError(f"Material '{name}' not found.")
        # Only analytic materials are supported for export here
        data = {
            'name': mat.name,
            'wavelength_range': mat.wavelength_range,
            'temp_range': mat.temp_range,
            'density': mat.density,
            'metadata': mat.metadata
            # Note: Function serialization is not supported; user must provide analytic forms
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _from_dict(self, data: dict) -> OpticalMaterial:
        # User must provide analytic forms as Python expressions or via code (not serializable)
        # This method is a placeholder for more advanced parsing or code evaluation
        raise NotImplementedError('Analytic material import requires user-provided functions.')
