from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from ..abstract_pipeline_component import AbstractPipelineComponent


class ClosestPointFinderComponent(AbstractPipelineComponent):
    """
    Component for finding closest points in point cloud to camera positions.
    Supports view cone filtering and floor plane distance calculations.
    
    Args:
        use_view_cone: Enable view cone filtering (default: False)
        cone_angle_deg: Half-angle of view cone in degrees (default: 90.0)
        max_view_distance: Maximum distance for view cone filtering (default: 10.0)
        use_floor_distance: Calculate horizontal distances on floor plane (default: False)
    
    Returns:
        Dictionary containing closest point analysis results
    
    Raises:
        ValueError: If invalid configuration or insufficient data
    """
    
    def __init__(self, use_view_cone: bool = False, cone_angle_deg: float = 90.0,
                 max_view_distance: float = 10.0, use_floor_distance: bool = False) -> None:
        super().__init__()
        self.use_view_cone = use_view_cone
        self.cone_angle_deg = cone_angle_deg
        self.max_view_distance = max_view_distance
        self.use_floor_distance = use_floor_distance

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point cloud and camera data."""
        base_inputs = ["point_cloud", "camera_positions"]
        
        if self.use_view_cone:
            base_inputs.append("camera_quaternions")
            
        if self.use_floor_distance:
            base_inputs.extend(["floor_normal", "floor_offset"])
            
        return base_inputs

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs closest point analysis results."""
        outputs = [
            "closest_point_3d", "closest_point_index", "distance_3d", 
            "distances_array"
        ]
        
        if self.use_floor_distance:
            outputs.extend([
                "closest_point_floor", "distance_floor", "projected_point", 
                "floor_distances_array"
            ])
            
        if self.use_view_cone:
            outputs.append("view_cone_mask")
            
        return outputs

    def _run(self, point_cloud, camera_positions: np.ndarray,
             camera_quaternions: Optional[np.ndarray] = None,
             floor_normal: Optional[np.ndarray] = None,
             floor_offset: Optional[float] = None,
             **kwargs: Any) -> Dict[str, Any]:
        """
        Find closest points for each camera position.
        
        Args:
            point_cloud: Point cloud data entity
            camera_positions: Nx3 array of camera positions
            camera_quaternions: Nx4 array of camera quaternions (if view cone enabled)
            floor_normal: Floor plane normal (if floor distance enabled)
            floor_offset: Floor plane offset (if floor distance enabled)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing closest point analysis results
        """
        # Validation
        if self.use_view_cone and camera_quaternions is None:
            raise ValueError("camera_quaternions required when use_view_cone=True")
            
        if self.use_floor_distance and (floor_normal is None or floor_offset is None):
            raise ValueError("floor_normal and floor_offset required when use_floor_distance=True")
        
        # Extract point cloud data
        if hasattr(point_cloud, 'point_cloud_numpy'):
            points = point_cloud.point_cloud_numpy
        else:
            points = point_cloud
            
        print("Building KD-tree for nearest neighbor search...")
        print(f"View cone filtering: {'enabled' if self.use_view_cone else 'disabled'}")
        print(f"Floor distance calculation: {'enabled' if self.use_floor_distance else 'disabled'}")
        
        # Find closest points
        results = self._find_closest_points(
            points, camera_positions, camera_quaternions,
            floor_normal, floor_offset
        )
        
        return results

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        x, y, z, w = q
        
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        
        return R

    def _filter_points_in_view_cone(self, point_cloud: np.ndarray, camera_position: np.ndarray, 
                                   camera_rotation: np.ndarray) -> np.ndarray:
        """Filter point cloud to only include points within the camera's view cone."""
        # Get camera forward direction
        R = self._quaternion_to_rotation_matrix(camera_rotation)
        camera_forward = R[:, 2]  # Z-axis (forward direction)
        
        # Vector from camera to each point
        point_vectors = point_cloud - camera_position
        
        # Distance filter
        distances = np.linalg.norm(point_vectors, axis=1)
        distance_mask = distances <= self.max_view_distance
        
        # Normalize point vectors
        normalized_vectors = point_vectors / (distances[:, np.newaxis] + 1e-8)
        
        # Compute dot product with camera forward direction
        dot_products = np.dot(normalized_vectors, camera_forward)
        
        # Convert cone angle to cosine threshold
        cos_threshold = np.cos(np.radians(self.cone_angle_deg))
        
        # Points are in cone if dot product > cos_threshold
        cone_mask = dot_products > cos_threshold
        
        # Combine distance and cone filters
        return distance_mask & cone_mask

    def _project_points_to_floor(self, points: np.ndarray, floor_normal: np.ndarray, 
                                floor_offset: float) -> np.ndarray:
        """Project 3D points onto the floor plane."""
        # Calculate distance from each point to the floor plane
        distances_to_plane = np.dot(points, floor_normal) - floor_offset
        
        # Project points onto floor plane by moving them along the normal
        projected_points = points - distances_to_plane[:, np.newaxis] * floor_normal
        
        return projected_points

    def _find_closest_points(self, point_cloud: np.ndarray, camera_positions: np.ndarray,
                            camera_quaternions: Optional[np.ndarray] = None,
                            floor_normal: Optional[np.ndarray] = None,
                            floor_offset: Optional[float] = None) -> Dict[str, Any]:
        """
        Find the closest point in the point cloud for each camera position.
        """
        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(point_cloud)
        
        # If floor distance is enabled, also build a 2D tree for floor projections
        floor_tree = None
        if self.use_floor_distance:
            print("Building 2D KD-tree for floor plane distance calculation...")
            projected_cloud = self._project_points_to_floor(point_cloud, floor_normal, floor_offset)
            floor_tree = cKDTree(projected_cloud[:, :2])
        
        print("Finding closest points for each camera pose...")
        
        closest_points = []
        distances = []
        indices = []
        floor_distances = []
        projected_points = []
        view_cone_masks = []
        
        for i, position in enumerate(camera_positions):
            if self.use_view_cone:
                # Filter points within view cone
                quaternion = camera_quaternions[i]
                view_mask = self._filter_points_in_view_cone(
                    point_cloud, position, quaternion
                )
                view_cone_masks.append(view_mask)
                
                if np.any(view_mask):
                    # Search only within visible points
                    visible_points = point_cloud[view_mask]
                    visible_indices = np.where(view_mask)[0]
                    
                    # Build temporary tree for visible points
                    visible_tree = cKDTree(visible_points)
                    dist, local_idx = visible_tree.query(position)
                    
                    # Map back to original indices
                    original_idx = visible_indices[local_idx]
                    closest_point = point_cloud[original_idx]
                else:
                    # No points in view cone, fall back to global closest
                    dist, original_idx = tree.query(position)
                    closest_point = point_cloud[original_idx]
            else:
                # Standard closest point search
                dist, original_idx = tree.query(position)
                closest_point = point_cloud[original_idx]
                view_cone_masks.append(None)
            
            closest_points.append(closest_point)
            distances.append(dist)
            indices.append(original_idx)
            
            # Calculate floor distance if enabled
            if self.use_floor_distance:
                # Project camera position to floor plane
                projected_camera = self._project_points_to_floor(
                    position.reshape(1, -1), floor_normal, floor_offset
                )[0]
                
                if self.use_view_cone and np.any(view_mask):
                    # Build 2D tree for visible points projected to floor
                    visible_points = point_cloud[view_mask]
                    visible_projected = self._project_points_to_floor(visible_points, floor_normal, floor_offset)
                    visible_floor_tree = cKDTree(visible_projected[:, :2])
                    floor_dist, floor_local_idx = visible_floor_tree.query(projected_camera[:2])
                    
                    # Map back to original point
                    floor_projected_point = visible_projected[floor_local_idx]
                else:
                    # Use global floor search
                    floor_dist, floor_idx = floor_tree.query(projected_camera[:2])
                    floor_projected_point = self._project_points_to_floor(
                        point_cloud[floor_idx].reshape(1, -1), floor_normal, floor_offset
                    )[0]
                
                floor_distances.append(floor_dist)
                projected_points.append(floor_projected_point)
        
        closest_points = np.array(closest_points)
        distances = np.array(distances)
        indices = np.array(indices)
        
        print(f"Found closest points. 3D distance range: {distances.min():.3f} to {distances.max():.3f}")
        
        # Prepare results
        results = {
            "closest_point_3d": closest_points[0] if len(closest_points) == 1 else closest_points,
            "closest_point_index": indices[0] if len(indices) == 1 else indices,
            "distance_3d": distances[0] if len(distances) == 1 else distances,
            "distances_array": distances
        }
        
        if self.use_floor_distance:
            floor_distances = np.array(floor_distances)
            projected_points = np.array(projected_points)
            print(f"Floor distance range: {floor_distances.min():.3f} to {floor_distances.max():.3f}")
            
            results.update({
                "closest_point_floor": projected_points[0] if len(projected_points) == 1 else projected_points,
                "distance_floor": floor_distances[0] if len(floor_distances) == 1 else floor_distances,
                "projected_point": projected_points[0] if len(projected_points) == 1 else projected_points,
                "floor_distances_array": floor_distances
            })
        
        if self.use_view_cone:
            results["view_cone_mask"] = view_cone_masks[0] if len(view_cone_masks) == 1 else view_cone_masks
        
        return results 