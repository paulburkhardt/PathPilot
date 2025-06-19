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
        n_closest_points: Number of closest points to return (default: 20)
    
    Returns:
        Dictionary containing closest point analysis results for current position
    
    Raises:
        ValueError: If invalid configuration or insufficient data
    """
    
    def __init__(self, 
                 use_view_cone: bool = False, 
                 cone_angle_deg: float = 90.0,
                 max_view_distance: float = 10.0,
                 n_closest_points: int = 20) -> None:
        super().__init__()
        self.use_view_cone = use_view_cone
        self.cone_angle_deg = cone_angle_deg
        self.max_view_distance = max_view_distance
        self.n_closest_points = n_closest_points

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

    def _get_camera_direction(self, camera_pose):
        """
        Extract camera direction vector from pose.
        
        Args:
            camera_pose: Camera pose object (T_WC)
            
        Returns:
            np.ndarray: Unit vector pointing in camera direction (forward direction)
        """
        try:
            # For lietorch.Sim3, get the rotation matrix and extract forward direction
            # Camera typically looks in -Z direction in camera frame
            rotation_matrix = camera_pose.rotation().matrix().cpu().numpy().reshape(3, 3)
            # Forward direction is -Z in camera coordinates, transformed to world coordinates
            camera_forward = rotation_matrix @ np.array([0, 0, -1])
        except AttributeError:
            # Fallback: if it's a matrix, extract rotation part
            if hasattr(camera_pose, 'cpu'):
                pose_matrix = camera_pose.cpu().numpy()
            else:
                pose_matrix = np.array(camera_pose)
            
            if pose_matrix.shape == (4, 4):
                rotation_matrix = pose_matrix[:3, :3]
                camera_forward = rotation_matrix @ np.array([0, 0, -1])
            else:
                raise ValueError(f"Unsupported camera pose format: {type(camera_pose)}")
        
        return camera_forward / np.linalg.norm(camera_forward)

    def _apply_view_cone_filter(self, points_3d, camera_position, camera_direction):
        """
        Apply view cone filtering to points.
        
        Args:
            points_3d: Array of 3D points [N, 3]
            camera_position: Camera position [3]
            camera_direction: Camera direction vector [3]
            
        Returns:
            np.ndarray: Boolean mask indicating which points are in view cone
        """
        # Vector from camera to each point
        points_relative = points_3d - camera_position[np.newaxis, :]
        
        # Distance from camera to each point
        distances = np.linalg.norm(points_relative, axis=1)
        
        # Filter out points beyond max view distance
        distance_mask = distances <= self.max_view_distance
        
        # Normalize relative vectors
        valid_distances = distances > 1e-8  # Avoid division by zero
        normalized_relative = np.zeros_like(points_relative)
        normalized_relative[valid_distances] = points_relative[valid_distances] / distances[valid_distances, np.newaxis]
        
        # Calculate angle between camera direction and vector to each point
        dot_products = np.dot(normalized_relative, camera_direction)
        # Clamp dot products to valid range [-1, 1] to handle numerical errors
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles_rad = np.arccos(dot_products)
        angles_deg = np.degrees(angles_rad)
        
        # Filter points within cone angle
        cone_mask = angles_deg <= self.cone_angle_deg
        
        # Combine distance and cone masks
        view_cone_mask = distance_mask & cone_mask & valid_distances
        
        return view_cone_mask

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
            camera_pose: Current camera pose object (T_WC)
            step_nr: Current step number
            floor_normal: Floor plane normal (if floor distance enabled)
            floor_offset: Floor plane offset (if floor distance enabled)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing closest point analysis results for current position
        """
        
        # Always wait for floor plane to be available
        if floor_normal is None or floor_offset is None:
            # Return empty outputs if floor is not yet available
            outputs = {
                "n_closest_points_3d": np.empty((0, 3),dtype=float),
                "n_closest_points_index": np.empty((0,), dtype=int),
                "n_closest_points_distance_2d": np.empty((0,),dtype=float),
            }
            if self.use_view_cone:
                outputs["view_cone_mask"] = np.array([])
            return outputs
        
        # Get point cloud as numpy array (shape: [N, 3])
        points_3d = point_cloud.as_numpy()  # Shape: [N, 3]
        
        # Check if we have enough points
        if len(points_3d) == 0:
            outputs = {
                "n_closest_points_3d": np.array([]),
                "n_closest_points_index": np.array([]),
                "n_closest_points_distance_2d": np.array([]),
            }
            if self.use_view_cone:
                outputs["view_cone_mask"] = np.array([])
            return outputs
        
        # Extract camera position from T_WC pose
        # T_WC is a lietorch.Sim3 object, we need to get the translation part
        try:
            # Get the camera position in world coordinates
            # For lietorch.Sim3, we can access the translation part
            camera_position = camera_pose.translation().cpu().numpy().reshape(3)
        except AttributeError:
            # Fallback: if it's a matrix, extract translation
            if hasattr(camera_pose, 'cpu'):
                pose_matrix = camera_pose.cpu().numpy()
            else:
                pose_matrix = np.array(camera_pose)
            
            if pose_matrix.shape == (4, 4):
                camera_position = pose_matrix[:3, 3]
            else:
                raise ValueError(f"Unsupported camera pose format: {type(camera_pose)}")
        
        # Apply view cone filtering if enabled
        if self.use_view_cone:
            camera_direction = self._get_camera_direction(camera_pose)
            view_cone_mask = self._apply_view_cone_filter(points_3d, camera_position, camera_direction)
            
            # Filter points to only those in view cone
            if not np.any(view_cone_mask):
                # No points in view cone
                outputs = {
                    "n_closest_points_3d": np.array([]),
                    "n_closest_points_index": np.array([]),
                    "n_closest_points_distance_2d": np.array([]),
                    "view_cone_mask": np.array([])
                }
                return outputs
            
            points_3d_filtered = points_3d[view_cone_mask]
            original_indices = np.where(view_cone_mask)[0]
        else:
            points_3d_filtered = points_3d
            original_indices = np.arange(len(points_3d))
            view_cone_mask = np.ones(len(points_3d), dtype=bool)
        
        # Project all points onto the floor plane
        floor_normal = np.array(floor_normal).reshape(3)
        floor_normal = floor_normal / np.linalg.norm(floor_normal)  # Normalize
        
        # Project points onto floor plane
        # For each point p, projected point = p - (dot(p-floor_point, floor_normal)) * floor_normal
        # where floor_point is any point on the plane: floor_normal * floor_offset
        floor_point = floor_normal * floor_offset
        
        # Project all filtered points
        points_to_floor = points_3d_filtered - floor_point[np.newaxis, :]
        distances_to_plane = np.dot(points_to_floor, floor_normal)
        points_projected = points_3d_filtered - distances_to_plane[:, np.newaxis] * floor_normal[np.newaxis, :]
        
        # Project camera position onto floor plane
        camera_to_floor = camera_position - floor_point
        camera_distance_to_plane = np.dot(camera_to_floor, floor_normal)
        camera_projected = camera_position - camera_distance_to_plane * floor_normal
        
        # Calculate 2D distances from camera to all projected points
        distances_2d = np.linalg.norm(points_projected - camera_projected[np.newaxis, :], axis=1)
        
        # Find the n closest points
        n_points = min(self.n_closest_points, len(points_3d_filtered))
        closest_indices = np.argpartition(distances_2d, n_points)[:n_points]
        
        # Sort the closest indices by distance
        sorted_closest_indices = closest_indices[np.argsort(distances_2d[closest_indices])]
        
        # Get the results (map back to original indices)
        closest_points_3d = points_3d_filtered[sorted_closest_indices]
        closest_distances_2d = distances_2d[sorted_closest_indices]
        closest_original_indices = original_indices[sorted_closest_indices]
        
        outputs = {
            "n_closest_points_3d": closest_points_3d,
            "n_closest_points_index": closest_original_indices,
            "n_closest_points_distance_2d": closest_distances_2d,
        }
        
        # Add view cone mask if enabled
        if self.use_view_cone:
            outputs["view_cone_mask"] = view_cone_mask
        
        return outputs
        