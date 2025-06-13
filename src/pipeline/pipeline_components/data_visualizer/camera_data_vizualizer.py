from typing import List, Dict, Any
from .abstract_rerun_data_vizualizer import AbstractRerunDataVisualizer
from scipy.spatial.transform import rotation as R
import rerun as rr
import numpy as np

class CameraDataVisualizer(AbstractRerunDataVisualizer):
    """
    Data visualizer component for camera poses.
    
    Args:
        -
    Returns:
        -
    Raises:
        NotImplementedError: As this is currently a placeholder
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._point_cloud_visualizer: Dict[str, Any] = {}
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires camera pose data as input."""
        return ["camera_pose"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs visualizations."""
        return []
    
    def _run(self, camera_pose, **kwargs: Any) -> Dict[str, Any]:
        """
        Visualize a camera pose.
        
        Args:
            camera_pose: The input camera pose to visualize
            **kwargs: Additional unused arguments
        """

        return {}