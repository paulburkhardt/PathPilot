from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from ..abstract_pipeline_component import AbstractPipelineComponent


class IncrementalClosestPointFinderComponent(AbstractPipelineComponent):
    """
    Component for finding closest points to the current camera position in accumulated point cloud.
    Works incrementally and waits for floor detection to be available before calculating floor distances.
    
    Args:
        use_view_cone: Enable view cone filtering (default: False)
        cone_angle_deg: Half-angle of view cone in degrees (default: 90.0)
        max_view_distance: Maximum distance for view cone filtering (default: 10.0)
        use_floor_distance: Calculate horizontal distances on floor plane (default: True)
        wait_for_floor: Wait for floor detection before starting analysis (default: True)
    
    Returns:
        Dictionary containing closest point analysis results for current position
    
    Raises:
        ValueError: If invalid configuration or insufficient data
    """
    
    def __init__(self, 
                 use_view_cone: bool = False, 
                 cone_angle_deg: float = 90.0,
                 wait_for_floor: bool = True) -> None:
        super().__init__()
        self.use_view_cone = use_view_cone
        self.cone_angle_deg = cone_angle_deg
        self.wait_for_floor = wait_for_floor
        
        # State tracking
        self.floor_available = False

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point cloud and current camera data."""

        return ["point_cloud", "camera_pose", "step_nr", "floor_normal", "floor_offset"]
    

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs closest point analysis results for current position."""
        outputs = [
            "n_closest_points_3d", 
            "n_closest_points_index", 
            "n_closest_points_distance_2d", 
        ]

            
        if self.use_view_cone:
            outputs.append("view_cone_mask")
            
        return outputs

    def _run(self, 
             point_cloud, 
             camera_pose, 
             step_nr: int,
             floor_normal: Optional[np.ndarray] = None,
             floor_offset: Optional[float] = None,
             **kwargs: Any) -> Dict[str, Any]:
        """
        Find closest point for the current camera position.
        
        Args:
            point_cloud: Point cloud data entity (accumulated)
            camera_pose: Current camera pose object
            step_nr: Current step number
            floor_normal: Floor plane normal (if floor distance enabled)
            floor_offset: Floor plane offset (if floor distance enabled)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing closest point analysis results for current position
        """

        