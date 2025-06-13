from typing import List, Dict, Any, Optional
from .abstract_rerun_data_vizualizer import AbstractRerunDataVisualizer
import rerun as rr
import numpy as np

class PointCloudDataVisualizer(AbstractRerunDataVisualizer):
    """
    Enhanced data visualizer component for point clouds with floor detection and highlighting.
    
    Args:
        enable_floor_visualization: Enable floor plane visualization (default: True)
        floor_threshold: Distance threshold for floor point identification (default: 0.05)
        grid_size: Size of floor grid visualization in meters (default: 2.0)
        highlight_floor_points: Highlight floor points in different color (default: True)
    
    Returns:
        Empty dictionary as this is a visualization component
    
    Raises:
        ValueError: If required data is missing
    """
    
    def __init__(self, enable_floor_visualization: bool = True, 
                 floor_threshold: float = 0.05, grid_size: float = 2.0,
                 highlight_floor_points: bool = True) -> None:
        super().__init__()
        self.enable_floor_visualization = enable_floor_visualization
        self.floor_threshold = floor_threshold
        self.grid_size = grid_size
        self.highlight_floor_points = highlight_floor_points
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point cloud data and optionally floor data."""
        inputs = ["point_cloud"]
        if self.enable_floor_visualization:
            inputs.extend(["floor_normal", "floor_offset"])
        return inputs
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs visualizations only."""
        return []
    
    def _run(self, point_cloud, floor_normal: Optional[np.ndarray] = None,
             floor_offset: Optional[float] = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Visualize a point cloud with optional floor highlighting.
        
        Args:
            point_cloud: The input point cloud to visualize
            floor_normal: Floor plane normal vector (optional)
            floor_offset: Floor plane offset (optional)
            **kwargs: Additional unused arguments
        """
        # Extract point cloud data
        if hasattr(point_cloud, 'point_cloud_numpy'):
            points = point_cloud.point_cloud_numpy
        else:
            points = point_cloud
            
        # Extract colors if available
        colors = None
        if hasattr(point_cloud, 'rgb_numpy'):
            colors = point_cloud.rgb_numpy
        
        # Extract confidence if available
        confidence = None
        if hasattr(point_cloud, 'confidence_scores_numpy'):
            confidence = point_cloud.confidence_scores_numpy

        # Log basic point cloud
        self._log_point_cloud(points, colors, confidence)
        
        # Log floor visualization if enabled and floor data is available
        if (self.enable_floor_visualization and 
            floor_normal is not None and floor_offset is not None):
            self._log_floor_visualization(points, colors, floor_normal, floor_offset)
        
        return {}
    
    def _log_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                        confidence: Optional[np.ndarray] = None) -> None:
        """
        Log the basic point cloud to Rerun.
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (optional)
            confidence: N array of confidence scores (optional)
        """
        print("Logging point cloud to Rerun...")
        
        # TODO: Add mode for visualizing confidence scores
        # This could color points based on confidence values
        
        if colors is not None:
            rr.log("world/pointcloud", rr.Points3D(points, colors=colors), static=True)
        else:
            rr.log("world/pointcloud", rr.Points3D(points, colors=[128, 128, 128]), static=True)
        
        print(f"Logged {len(points)} points to Rerun")

    def _log_floor_visualization(self, points: np.ndarray, colors: Optional[np.ndarray],
                                floor_normal: np.ndarray, floor_offset: float) -> None:
        """
        Log floor plane visualization including highlighted floor points and grid.
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (optional)
            floor_normal: Floor plane normal vector
            floor_offset: Floor plane offset
        """
        print("Adding floor visualization...")
        
        if self.highlight_floor_points:
            self._log_highlighted_floor_points(points, colors, floor_normal, floor_offset)
        
        self._log_floor_grid(points, floor_normal, floor_offset)

    def _log_highlighted_floor_points(self, points: np.ndarray, colors: Optional[np.ndarray],
                                     floor_normal: np.ndarray, floor_offset: float) -> None:
        """
        Highlight floor points in the point cloud.
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (optional) 
            floor_normal: Floor plane normal vector
            floor_offset: Floor plane offset
        """
        # Calculate distance of each point to the floor plane
        distances_to_floor = np.abs(np.dot(points, floor_normal) - floor_offset)
        floor_mask = distances_to_floor < self.floor_threshold
        
        # Create modified colors highlighting floor points
        if colors is not None:
            modified_colors = colors.copy()
        else:
            modified_colors = np.full((len(points), 3), [128, 128, 128], dtype=np.uint8)
        
        # Color floor points in bright green
        modified_colors[floor_mask] = [0, 255, 0]  # Bright green for floor
        
        # Log the point cloud with floor highlighting
        rr.log("world/pointcloud_with_floor", rr.Points3D(points, colors=modified_colors), static=True)
        
        # Also log just the floor points separately for better visibility
        floor_points = points[floor_mask]
        if len(floor_points) > 0:
            rr.log("world/floor_points", rr.Points3D(
                floor_points, 
                colors=[0, 255, 0],  # Bright green
                radii=[0.01]
            ), static=True)
            
        print(f"Highlighted {np.sum(floor_mask)} floor points out of {len(points)} total points")

    def _log_floor_grid(self, points: np.ndarray, floor_normal: np.ndarray, floor_offset: float) -> None:
        """
        Log a grid visualization on the floor plane.
        
        Args:
            points: Nx3 array of 3D points (used to determine grid center)
            floor_normal: Floor plane normal vector
            floor_offset: Floor plane offset
        """
        # Calculate floor center based on point cloud center projected to floor
        floor_center = np.mean(points, axis=0)
        floor_center_on_plane = floor_center - np.dot(floor_center - floor_offset * floor_normal, floor_normal) * floor_normal
        
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
            line_start = floor_center_on_plane + (i * self.grid_size/5) * u + (-self.grid_size) * v
            line_end = floor_center_on_plane + (i * self.grid_size/5) * u + (self.grid_size) * v
            grid_lines.append(np.array([line_start, line_end]))
        
        # Create vertical lines  
        for j in range(-5, 6):
            line_start = floor_center_on_plane + (-self.grid_size) * u + (j * self.grid_size/5) * v
            line_end = floor_center_on_plane + (self.grid_size) * u + (j * self.grid_size/5) * v
            grid_lines.append(np.array([line_start, line_end]))
        
        if len(grid_lines) > 0:
            rr.log("world/floor_grid", rr.LineStrips3D(
                strips=grid_lines,
                colors=[255, 255, 0],  # Yellow for floor grid
                radii=[0.002]
            ), static=True)
            
        print(f"Added floor grid visualization with {len(grid_lines)} lines")



