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
        "total_steps",
        
        "image",
        "image_height",
        "image_width",
        "image_size",

        "point_cloud",
        "camera_pose",
        "timestamp",
        "calibration_K",

        "floor_normal", 
        "floor_offset", 
        "floor_threshold", 
        "floor_points",

        "n_closest_points_3d", 
        "n_closest_points_index", 
        "n_closest_points_distance_2d",
        "view_cone_mask",
        "floor_filtered_mask"
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
