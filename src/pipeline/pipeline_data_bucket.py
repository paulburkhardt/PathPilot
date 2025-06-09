from typing import Dict, List, Any

class PipelineDataBucket:
    """
    A container class for passing data between pipeline components.
    
    Args:
        step_nr: The current step number in the pipeline.
    
    Returns:
        -
    
    Raises:
        KeyError: When trying to store data with a key that is not in __available_data_entities
    """
    
    __available_data_entities: List[str] = [
        "step_nr",
        "rgb_image",
        "depth_image",
        "point_cloud",
        "segmentation_mask",
        "object_data",
        "slam_data",
        "camera_pose",
        "timestamp",
        "img_height",
        "img_width",
        "img_size"
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
