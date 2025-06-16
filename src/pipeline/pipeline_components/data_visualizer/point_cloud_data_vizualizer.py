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
    
    def __init__(self,use_rgb_color: bool,use_confidence: bool) -> None:
        super().__init__()
        self.use_rgb_color = use_rgb_color
        self.use_confidence = use_confidence
        
    
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

        colors, radii = self.adjust_point_visuals(colors, confidence)

        self.log_point_cloud(points, colors, radii)
        #self.log_pointcloud_floor(points, colors,)
        return {}
    

    def adjust_point_visuals(self,colors, confidence):
        radii = 0.01
        if self.use_confidence:
            radii = (confidence * radii)

        point_colors = [128, 128, 128] #Gray
        if colors is not None and self.use_colors is True:
            point_colors = colors
        return point_colors, radii


    def log_point_cloud(self, points, colors, radii):

        rr.log("world/pointcloud", rr.Points3D(points, colors=colors, radii = radii), static=True)


    def log_pointcloud_floor(self, points, colors,config, radii):
        
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
                radii= radii
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



    def find_floor(self,point_cloud):
        z_coordinates = point_cloud[:,2]
        threshold = np.percentile(z_coordinates,15)
        floor_points = point_cloud[z_coordinates < threshold]

        center = np.mean(floor_points, axis=0)
        centered_floor_points= floor_points -center

        Covariance = centered_floor_points.T @ centered_floor_points / len(floor_points)
        _, eigen_vec = np.linalg.eigh(Covariance)
        floor_normal =  eigen_vec[:,0]
        floor_normal/= np.linalg.norm(floor_normal)

        
