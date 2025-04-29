import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class HyperspectralMaterialIdentifier:
    """
    Identifies target materials from hyperspectral data using spectral signature matching
    and machine learning classification techniques.
    """
    def __init__(self, 
                 spectral_library: Optional[Dict[str, np.ndarray]] = None,
                 wavelengths_nm: Optional[np.ndarray] = None,
                 confidence_threshold: float = 0.85):
        """
        Initialize hyperspectral material identifier.
        
        Args:
            spectral_library: Dictionary of {material_name: spectral_signature}
            wavelengths_nm: Array of wavelengths corresponding to spectral signatures
            confidence_threshold: Minimum confidence for positive identification
        """
        self.spectral_library = spectral_library or {}
        self.wavelengths_nm = wavelengths_nm
        self.confidence_threshold = confidence_threshold
        self.classifier = None
        self.pca = None
        
        # Initialize if library provided
        if spectral_library and wavelengths_nm is not None:
            self.train_classifier()
    
    def add_material_signature(self, material_name: str, signature: np.ndarray):
        """Add a material signature to the spectral library."""
        self.spectral_library[material_name] = signature
    
    def train_classifier(self, n_components: int = 10):
        """Train the material classifier using the spectral library."""
        if not self.spectral_library:
            return
            
        # Extract signatures and labels
        signatures = []
        labels = []
        
        for material, signature in self.spectral_library.items():
            signatures.append(signature)
            labels.append(material)
            
        # Convert to numpy arrays
        X = np.array(signatures)
        y = np.array(labels)
        
        # Apply dimensionality reduction
        self.pca = PCA(n_components=min(n_components, X.shape[1]))
        X_reduced = self.pca.fit_transform(X)
        
        # Train classifier
        self.classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
        self.classifier.fit(X_reduced, y)
    
    def identify_material(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """
        Identify material from hyperspectral signature.
        
        Args:
            spectrum: Spectral signature to identify
            
        Returns:
            Dictionary with identification results
        """
        if not self.classifier or not self.pca:
            return {'identified': False, 'error': 'Classifier not trained'}
            
        # Apply dimensionality reduction
        spectrum_reduced = self.pca.transform(spectrum.reshape(1, -1))
        
        # Get predictions and probabilities
        material = self.classifier.predict(spectrum_reduced)[0]
        probabilities = self.classifier.predict_proba(spectrum_reduced)[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        # Calculate spectral angle mapper scores for verification
        sam_scores = {}
        for name, reference in self.spectral_library.items():
            sam_scores[name] = self._spectral_angle_mapper(spectrum, reference)
        
        # Determine if identification is confident
        is_identified = confidence >= self.confidence_threshold
        
        return {
            'identified': is_identified,
            'material': material if is_identified else None,
            'confidence': float(confidence),
            'sam_scores': sam_scores,
            'all_probabilities': {m: float(p) for m, p in zip(self.classifier.classes_, probabilities)}
        }
    
    def _spectral_angle_mapper(self, spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
        """
        Calculate spectral angle between two spectral signatures.
        
        Args:
            spectrum1: First spectral signature
            spectrum2: Second spectral signature
            
        Returns:
            Spectral angle in radians
        """
        # Ensure spectra are normalized
        norm1 = spectrum1 / np.linalg.norm(spectrum1)
        norm2 = spectrum2 / np.linalg.norm(spectrum2)
        
        # Calculate dot product
        dot_product = np.sum(norm1 * norm2)
        
        # Clamp to valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Return angle in radians
        return float(np.arccos(dot_product))
    
    def analyze_image(self, hyperspectral_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze hyperspectral image to identify materials.
        
        Args:
            hyperspectral_image: 3D array (height, width, bands)
            
        Returns:
            Dictionary with analysis results
        """
        height, width, _ = hyperspectral_image.shape
        
        # Initialize results
        material_map = np.zeros((height, width), dtype=object)
        confidence_map = np.zeros((height, width))
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                spectrum = hyperspectral_image[y, x, :]
                result = self.identify_material(spectrum)
                
                if result['identified']:
                    material_map[y, x] = result['material']
                    confidence_map[y, x] = result['confidence']
        
        # Count materials
        unique_materials = {}
        for material in np.unique(material_map):
            if material:  # Skip None values
                count = np.sum(material_map == material)
                unique_materials[material] = int(count)
        
        return {
            'material_map': material_map,
            'confidence_map': confidence_map,
            'material_counts': unique_materials,
            'average_confidence': float(np.mean(confidence_map[confidence_map > 0]))
        }

# Example usage:
# # Create spectral library
# wavelengths = np.linspace(400, 2500, 100)  # 400-2500nm with 100 bands
# library = {
#     'aluminum': np.random.rand(100) * 0.2 + 0.7,  # High reflectance
#     'vegetation': np.concatenate([np.random.rand(50) * 0.3, np.random.rand(50) * 0.7 + 0.3]),
#     'concrete': np.random.rand(100) * 0.4 + 0.3
# }
# 
# # Initialize identifier
# identifier = HyperspectralMaterialIdentifier(library, wavelengths)
# 
# # Identify a sample spectrum
# sample = library['aluminum'] + np.random.normal(0, 0.05, 100)  # Aluminum with noise
# result = identifier.identify_material(sample)
# print(result)