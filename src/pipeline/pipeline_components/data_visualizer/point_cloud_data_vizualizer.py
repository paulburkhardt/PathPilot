from typing import List, Dict, Any
from .abstract_rerun_data_vizualizer import AbstractRerunDataVisualizer
from scipy.spatial.transform import rotation as R
import rerun as rr
import numpy as np

class PointCloudDataVisualizer(AbstractRerunDataVisualizer):
    """
    Data visualizer component for point clouds.
    
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
        """This component requires point cloud data as input."""
        return ["point_cloud"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs visualizations."""
        return []
    
    def _run(self, point_cloud, **kwargs: Any) -> Dict[str, Any]:
        """
        Visualize a point cloud.
        
        Args:
            point_cloud: The input point cloud to visualize
            **kwargs: Additional unused arguments
        """

        points = point_cloud.point_cloud_numpy
        colors = point_cloud.rgb_numpy
        confidence = point_cloud.confidence_scores_numpy

        self.log_point_cloud(points, colors)
        self.log_pointcloud_floor(points, colors,)
        return {}
    
 

    def log_point_cloud(self, points, colors,confidence):

        # TODO Add mode for vizualizing confidence
        
        if colors is not None:
            rr.log("world/pointcloud", rr.Points3D(points, colors=colors), static=True)
        else:
            rr.log("world/pointcloud", rr.Points3D(points, colors=[128, 128, 128]), static=True)


    def log_pointcloud_floor(self, points, colors,config):
        
        floor_offset    = config['floor_offset']
        floor_normal    = config['floor_normal']
        floor_threshold = config['floor_threshold']

        if floor_normal is  None or floor_offset is  None:
            return
        
        distances_to_floor = np.abs(np.dot(points, floor_normal) - floor_offset)
        floor_mask = distances_to_floor < floor_threshold
        floor_points = points[floor_mask]
        if len(floor_points) > 0:
            rr.log("world/floor_points", rr.Points3D(
                floor_points, 
                colors=[0, 255, 0],  # Bright green
                radii=[0.01]
            ), static=True)

        floor_center = np.mean(points, axis=0)
        floor_center_on_plane = floor_center - np.dot(floor_center - floor_offset * floor_normal, floor_normal) * floor_normal
        
        # Create a grid to visualize the floor plane
        grid_size = 2.0  # 2x2 meter grid
        grid_points = []
        
        # Find two orthogonal vectors in the floor plane
        if abs(floor_normal[0]) < 0.9:
            u = np.cross(floor_normal, [1, 0, 0])
        else:
            u = np.cross(floor_normal, [0, 1, 0])

        u = u / np.linalg.norm(u)
        v = np.cross(floor_normal, u)
        v = v / np.linalg.norm(v)
        
        # Create a grid to visualize the floor plane as lines
        grid_lines = []
        
        # Create horizontal lines
        for i in range(-5, 6):
            line_start = floor_center_on_plane + (i * grid_size/5) * u + (-grid_size) * v
            line_end = floor_center_on_plane + (i * grid_size/5) * u + (grid_size) * v
            grid_lines.append(np.array([line_start, line_end]))
        
        # Create vertical lines  
        for j in range(-5, 6):
            line_start = floor_center_on_plane + (-grid_size) * u + (j * grid_size/5) * v
            line_end = floor_center_on_plane + (grid_size) * u + (j * grid_size/5) * v
            grid_lines.append(np.array([line_start, line_end]))
        
        if len(grid_lines) > 0:
            rr.log("world/floor_grid", rr.LineStrips3D(
                strips=grid_lines,
                colors=[255, 255, 0],  # Yellow for floor grid
                radii=[0.002]
            ), static=True)



