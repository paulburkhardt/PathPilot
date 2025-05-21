import numpy as np
from typing import Dict, Any, List, Tuple


class DistanceCalculator:
    def __init__(self, config: Dict[str, Any]):
        self.warning_threshold = config['warning_threshold']
        self.update_frequency = config['update_frequency']

    def calculate_distances(self, 
                          point_cloud: np.ndarray, 
                          segmented_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate distances to segmented objects using point cloud data.
        
        Args:
            point_cloud: Nx3 array of 3D points
            segmented_objects: List of segmented objects with masks
            
        Returns:
            List of dictionaries containing object distances and metadata
        """
        distances = []
        
        for obj in segmented_objects:
            # Get points corresponding to the object mask
            mask = obj['mask']
            object_points = point_cloud[mask]
            
            if len(object_points) == 0:
                continue
                
            # Calculate minimum distance to object
            min_distance = np.min(np.linalg.norm(object_points, axis=1))
            
            distances.append({
                'distance': min_distance,
                'confidence': obj['confidence'],
                'bbox': obj['bbox']
            })
            
        return distances

    def get_closest_object(self, distances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get the closest object from the list of distances."""
        if not distances:
            return None
            
        return min(distances, key=lambda x: x['distance']) 