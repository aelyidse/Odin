from typing import Dict, Any, Optional
import numpy as np

class MaterialReflectanceModel:
    """
    Holds reflectance properties for different materials (spectral, angular, polarization, etc.).
    """
    def __init__(self, reflectance_db: Optional[Dict[str, Dict[str, Any]]] = None):
        # Example: {'concrete': {'albedo': 0.6, 'specular': 0.1, ...}, ...}
        self.reflectance_db = reflectance_db or {
            'concrete': {'albedo': 0.6, 'specular': 0.1},
            'asphalt': {'albedo': 0.12, 'specular': 0.05},
            'soil': {'albedo': 0.18, 'specular': 0.03},
            'grass': {'albedo': 0.25, 'specular': 0.02},
            'water': {'albedo': 0.08, 'specular': 0.9},
            'metal': {'albedo': 0.5, 'specular': 0.95},
            'sand': {'albedo': 0.35, 'specular': 0.02},
        }
    def get(self, material: str) -> Dict[str, Any]:
        return self.reflectance_db.get(material.lower(), {'albedo': 0.2, 'specular': 0.05})
    def set(self, material: str, props: Dict[str, Any]):
        self.reflectance_db[material.lower()] = props

class TerrainCell:
    """
    Represents a cell in the terrain grid, with material and reflectance properties.
    """
    def __init__(self, material: str, elevation: float = 0.0, reflectance_model: Optional[MaterialReflectanceModel] = None):
        self.material = material
        self.elevation = elevation
        self.reflectance = (reflectance_model or MaterialReflectanceModel()).get(material)

class TerrainModel:
    """
    Terrain grid with material-specific reflectance properties for each cell.
    """
    def __init__(self, grid_shape: tuple, elevation_map: Optional[np.ndarray] = None, material_map: Optional[np.ndarray] = None, reflectance_model: Optional[MaterialReflectanceModel] = None):
        self.grid_shape = grid_shape
        self.elevation_map = elevation_map if elevation_map is not None else np.zeros(grid_shape)
        self.material_map = material_map if material_map is not None else np.full(grid_shape, 'soil', dtype=object)
        self.reflectance_model = reflectance_model or MaterialReflectanceModel()
        self.grid = self._build_grid()
    def _build_grid(self):
        grid = np.empty(self.grid_shape, dtype=object)
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                mat = self.material_map[i, j]
                elev = self.elevation_map[i, j]
                grid[i, j] = TerrainCell(mat, elev, self.reflectance_model)
        return grid
    def get_reflectance(self, i: int, j: int) -> Dict[str, Any]:
        return self.grid[i, j].reflectance
    def set_material(self, i: int, j: int, material: str):
        self.material_map[i, j] = material
        self.grid[i, j] = TerrainCell(material, self.elevation_map[i, j], self.reflectance_model)
    def to_dict(self) -> Dict[str, Any]:
        return {
            'shape': self.grid_shape,
            'cells': [[self.grid[i, j].reflectance for j in range(self.grid_shape[1])] for i in range(self.grid_shape[0])]
        }

# Example usage:
# terrain = TerrainModel((100,100))
# print(terrain.get_reflectance(0,0))
# terrain.set_material(0,0,'metal')
# print(terrain.get_reflectance(0,0))
