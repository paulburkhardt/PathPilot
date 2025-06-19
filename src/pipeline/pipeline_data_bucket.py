from typing import Dict, List, Any

class PipelineDataBucket:
    """
    A container class for passing data between pipeline components.
    
    Args:
        -
    Returns:
        -
    
    Raises:
        KeyError: When trying to store data with a key that is not in __available_data_entities
    """
    
    __available_data_entities: List[str] = [
        "step_nr",
        
        "image",
        "image_height",
        "image_width",
        "image_size",

        "point_cloud",
        "camera_pose",
        "timestamp",
        "calibration_K",
        
        # Floor detection and coordinate system data entities
        "floor_normal",
        "floor_offset", 
        "floor_threshold",
        "floor_points",
        "floor_grid",
        
        # Closest point analysis data entities
        "closest_point_3d",
        "closest_point_index",
        "distance_3d",
        "closest_point_floor",
        "distance_floor",
        "projected_point",
        
        # Camera trajectory data entities
        "camera_positions",
        "camera_quaternions",
        "timestamps",
        
        # Analysis results
        "view_cone_mask",
        "distances_array",
        "floor_distances_array",
        "trajectory_summary",
        
        # Intermediate file paths for stage coordination
        "point_cloud_path",
        "trajectory_path",
        "floor_data_path",
        "closest_points_path",
        "output_directory"
    ]
    
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        
    def put(self, data: Dict[str, Any]) -> None:
        """
        Store data in the bucket.
        
        Args:
            data: Dictionary containing data to store. Keys must be in __available_data_entities.
        Returns:
            -
        Raises:
            KeyError: If a key in data is not in __available_data_entities
        """
        invalid_keys = set(data.keys()) - set(self.__available_data_entities)
        if invalid_keys:
            raise KeyError(f"The following keys are not valid data entities: {invalid_keys}")
            
        self._data.update(data)
        
    def get(self, *keys: str) -> Dict[str, Any]:
        """
        Retrieve data from the bucket.
        
        Args:
            *keys: Variable number of keys to retrieve data for.
        Returns:
            Dictionary containing the requested data.
        Raises:
            KeyError: If a requested key doesn't exist in the bucket.
        """
        return {key: self._data[key] for key in keys}
    
    def get_optional(self, *keys: str) -> Dict[str, Any]:
        """
        Retrieve data from the bucket, including only keys that exist.
        
        Args:
            *keys: Variable number of keys to retrieve data for.
        Returns:
            Dictionary containing the requested data for keys that exist.
        """
        return {key: self._data[key] for key in keys if key in self._data}
    
    def get_with_optional(self, required_keys: List[str], optional_keys: List[str]) -> Dict[str, Any]:
        """
        Retrieve data from the bucket with both required and optional keys.
        
        Args:
            required_keys: Keys that must be present.
            optional_keys: Keys that are optional.
        Returns:
            Dictionary containing all available data.
        Raises:
            KeyError: If a required key doesn't exist in the bucket.
        """
        result = {key: self._data[key] for key in required_keys}
        result.update({key: self._data[key] for key in optional_keys if key in self._data})
        return result
