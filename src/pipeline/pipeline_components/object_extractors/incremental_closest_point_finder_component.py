from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from ..abstract_pipeline_component import AbstractPipelineComponent


class IncrementalClosestPointFinderComponent(AbstractPipelineComponent):
    """
    Component for finding closest points to the current camera position in accumulated point cloud.
    Works incrementally and waits for floor detection to be available before calculating floor distances.
    Always filters out points that are on the floor plane.
    
    Args:
        use_view_cone: Enable view cone filtering (default: False)
        cone_angle_deg: Half-angle of view cone in degrees (default: 90.0)
        floor_distance_threshold: Maximum distance from floor plane to consider a point as "on floor" (default: 0.05)
        n_closest_points: Number of closest points to return (default: 20)
    
    Returns:
        Dictionary containing closest point analysis results for current position
    
    Raises:
        ValueError: If invalid configuration or insufficient data
    """
    
    def __init__(self, 
                 use_view_cone: bool = False, 
                 cone_angle_deg: float = 90.0,
                 floor_distance_threshold: float = 0.05,
                 n_closest_points: int = 20) -> None:
        super().__init__()
        self.use_view_cone = use_view_cone
        self.cone_angle_deg = cone_angle_deg
        self.floor_distance_threshold = floor_distance_threshold
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
            "floor_filtered_mask",
        ]

            
        if self.use_view_cone:
            outputs.append("view_cone_mask")
            
        return outputs

    def _get_camera_direction(self, camera_pose):
        """
        Extract camera direction vector from pose using +Z convention.
        
        Args:
            camera_pose: Camera pose object (T_WC)
            
        Returns:
            np.ndarray: Unit vector pointing in camera direction (forward direction)
        """
        try:
            # For lietorch.Sim3, get the rotation matrix and extract forward direction
            rotation_matrix = camera_pose.matrix()[0,0:3,0:3].cpu().numpy().reshape(3, 3)
            
            # Use +Z convention (OpenGL/visualization convention)
            camera_forward = rotation_matrix @ np.array([0, 0, 1])
            
        except AttributeError:
            # Fallback: if it's a matrix, extract rotation part
            if hasattr(camera_pose, 'cpu'):
                pose_matrix = camera_pose.cpu().numpy()
            else:
                pose_matrix = np.array(camera_pose)
            
            if pose_matrix.shape == (4, 4):
                rotation_matrix = pose_matrix[:3, :3]
                # Use +Z convention
                camera_forward = rotation_matrix @ np.array([0, 0, 1])
            else:
                raise ValueError(f"Unsupported camera pose format: {type(camera_pose)}")
        
        return camera_forward / np.linalg.norm(camera_forward)

    def _apply_view_cone_filter(self, points_3d, camera_position, camera_direction, step_nr=None):
        """
        Apply view cone filtering to points with improved stability and debugging.
        
        Args:
            points_3d: Array of 3D points [N, 3]
            camera_position: Camera position [3]
            camera_direction: Camera direction vector [3]
            step_nr: Current step number for debugging (optional)
            
        Returns:
            np.ndarray: Boolean mask indicating which points are in view cone
        """
        if len(points_3d) == 0:
            return np.array([], dtype=bool)
        
        # Vector from camera to each point
        points_relative = points_3d - camera_position[np.newaxis, :]
        
        # Distance from camera to each point
        distances = np.linalg.norm(points_relative, axis=1)
        valid_distances = distances > 1e-6  # More conservative threshold
        
        if not np.any(valid_distances):
            if step_nr is not None:
                print(f"Warning: No valid points with sufficient distance at step {step_nr}")
            return np.zeros(len(points_3d), dtype=bool)
        
        # Normalize relative vectors (only for valid distances)
        normalized_relative = np.zeros_like(points_relative)
        normalized_relative[valid_distances] = (
            points_relative[valid_distances] / distances[valid_distances, np.newaxis]
        )
        
        # Use direct cosine threshold instead of arccos for better stability
        cone_threshold = np.cos(np.radians(self.cone_angle_deg))
        
        # Calculate dot products with camera direction
        dot_products = np.dot(normalized_relative, camera_direction)
        
        # Points within cone: dot_product >= cos(angle)
        cone_mask = dot_products >= cone_threshold
        
        # Combine with valid distances
        view_cone_mask = cone_mask & valid_distances

        
        return view_cone_mask

    def _filter_floor_points(self, points_3d, floor_normal, floor_offset):
        """
        Filter out points that are too close to the floor plane.
        
        Args:
            points_3d: Array of 3D points [N, 3]
            floor_normal: Floor plane normal vector [3]
            floor_offset: Floor plane offset scalar
            
        Returns:
            np.ndarray: Boolean mask indicating which points are NOT on the floor
        """
        floor_normal = np.array(floor_normal).reshape(3)
        floor_normal = floor_normal / np.linalg.norm(floor_normal)  # Normalize
        
        # Calculate distance from each point to the floor plane
        # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
        # Since we have normalized normal vector: distance = |dot(point, normal) - offset|
        distances_to_floor = np.abs(np.dot(points_3d, floor_normal) - floor_offset)
        
        # Keep points that are above the threshold distance from floor
        not_on_floor_mask = distances_to_floor > self.floor_distance_threshold
        
        return not_on_floor_mask

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
                "floor_filtered_mask": np.array([]),
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
                "floor_filtered_mask": np.array([]),
            }
            if self.use_view_cone:
                outputs["view_cone_mask"] = np.array([])
            return outputs
        
        # Extract camera position from T_WC pose
        # T_WC is a lietorch.Sim3 object, we need to get the translation part
        try:
            # Get the camera position in world coordinates
            # For lietorch.Sim3, we can access the translation part
            camera_position = camera_pose.translation()[:,0:3].cpu().numpy().reshape(3)
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
        
        # Always filter out floor points first
        floor_filtered_mask = self._filter_floor_points(points_3d, floor_normal, floor_offset)
        
        if not np.any(floor_filtered_mask):
            # No points above floor
            outputs = {
                "n_closest_points_3d": np.array([]),
                "n_closest_points_index": np.array([]),
                "n_closest_points_distance_2d": np.array([]),
                "floor_filtered_mask": floor_filtered_mask,
            }
            if self.use_view_cone:
                outputs["view_cone_mask"] = np.array([])
            return outputs
        
        points_3d_after_floor = points_3d[floor_filtered_mask]
        indices_after_floor = np.where(floor_filtered_mask)[0]
        
        # Apply view cone filtering if enabled
        if self.use_view_cone:
            camera_direction = self._get_camera_direction(camera_pose)
            view_cone_mask_subset = self._apply_view_cone_filter(points_3d_after_floor, camera_position, camera_direction, step_nr)
            
            # Filter points to only those in view cone
            if not np.any(view_cone_mask_subset):
                # No points in view cone after floor filtering
                # Create full view cone mask for original point cloud
                view_cone_mask_full = np.zeros(len(points_3d), dtype=bool)
                outputs = {
                    "n_closest_points_3d": np.array([]),
                    "n_closest_points_index": np.array([]),
                    "n_closest_points_distance_2d": np.array([]),
                    "floor_filtered_mask": floor_filtered_mask,
                    "view_cone_mask": view_cone_mask_full,
                }
                return outputs
            
            points_3d_filtered = points_3d_after_floor[view_cone_mask_subset]
            original_indices = indices_after_floor[view_cone_mask_subset]
            
            # Create full view cone mask for original point cloud
            view_cone_mask_full = np.zeros(len(points_3d), dtype=bool)
            view_cone_mask_full[indices_after_floor[view_cone_mask_subset]] = True
        else:
            points_3d_filtered = points_3d_after_floor
            original_indices = indices_after_floor
            view_cone_mask_full = np.ones(len(points_3d), dtype=bool)
        
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
            "floor_filtered_mask": floor_filtered_mask,
        }
        
        # Add view cone mask if enabled
        if self.use_view_cone:
            outputs["view_cone_mask"] = view_cone_mask_full
        
        return outputs
        