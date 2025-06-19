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
    
    def __init__(self, use_view_cone: bool = False, cone_angle_deg: float = 90.0,
                 max_view_distance: float = 10.0, use_floor_distance: bool = True,
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
        return ["point_cloud", "camera_pose", "step_nr"]
    
    @property
    def optional_inputs_from_bucket(self) -> List[str]:
        """This component can optionally use floor detection data."""
        if self.use_floor_distance:
            return ["floor_normal", "floor_offset"]
        return []

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs closest point analysis results for current position."""
        outputs = [
            "closest_point_3d", "closest_point_index", "distance_3d", 
            "distances_array"  # For compatibility with visualizers (single-element array)
        ]

            
        if self.use_view_cone:
            outputs.append("view_cone_mask")
            
        return outputs

    def _run(self, point_cloud, camera_pose, step_nr: int,
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
        # Check if floor detection is available
        if self.use_floor_distance:
            if floor_normal is not None and floor_offset is not None:
                if not self.floor_available:
                    print(f"Floor detection now available at step {step_nr}, starting closest point analysis")
                self.floor_available = True
            elif self.wait_for_floor:
                # Wait for floor detection
                print(f"Waiting for floor detection at step {step_nr}")
                return {}
        
        # Extract current camera position from pose
        current_position = self._extract_camera_position(camera_pose)
        if current_position is None:
            print(f"Could not extract camera position at step {step_nr}")
            return {}
        
        # Extract current camera quaternion if needed
        current_quaternion = None
        if self.use_view_cone:
            current_quaternion = self._extract_camera_quaternion(camera_pose)
            if current_quaternion is None:
                print(f"Could not extract camera quaternion at step {step_nr}")
                return {}
        
        # Extract point cloud data
        if hasattr(point_cloud, 'point_cloud_numpy'):
            points = point_cloud.point_cloud_numpy
        else:
            points = point_cloud
            
        if len(points) == 0:
            print(f"No points in point cloud at step {step_nr}")
            return {}
            
        print(f"Finding closest point for camera at step {step_nr} in point cloud with {len(points)} points")
        
        # Find closest point
        results = self._find_closest_point_current(
            points, current_position, current_quaternion,
            floor_normal, floor_offset, step_nr
        )
        
        return results

    def _extract_camera_position(self, camera_pose) -> Optional[np.ndarray]:
        """Extract current camera position from pose object."""
        try:
            if hasattr(camera_pose, 'data'):
                pose_data = camera_pose.data.cpu().numpy().reshape(-1)
                if len(pose_data) >= 7:
                    return pose_data[:3].astype(np.float32)
            elif hasattr(camera_pose, 'matrix'):
                T = camera_pose.matrix().cpu().numpy()
                return T[:3, 3].astype(np.float32)
            elif hasattr(camera_pose, 'translation'):
                return camera_pose.translation.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error extracting camera position: {e}")
        return None

    def _extract_camera_quaternion(self, camera_pose) -> Optional[np.ndarray]:
        """Extract current camera quaternion from pose object."""
        try:
            # First try from the pose object itself
            if hasattr(camera_pose, 'data'):
                pose_data = camera_pose.data.cpu().numpy().reshape(-1)
                if len(pose_data) >= 7:
                    return pose_data[3:7].astype(np.float32)
            elif hasattr(camera_pose, 'matrix'):
                T = camera_pose.matrix().cpu().numpy()
                from scipy.spatial.transform import Rotation
                return Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
            elif hasattr(camera_pose, 'rotation'):
                return camera_pose.rotation.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error extracting camera quaternion: {e}")
        return None

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
        
        
        # Compute dot product with camera forward direction
        dot_products = np.dot(point_vectors, camera_forward)
        
        # Convert cone angle to cosine threshold
        cos_threshold = np.cos(np.radians(self.cone_angle_deg))
        
        # Points are in cone if dot product > cos_threshold
        cone_mask = dot_products > cos_threshold
        
        return  cone_mask

    def _project_points_to_floor(self, points: np.ndarray, floor_normal: np.ndarray, 
                                floor_offset: float) -> np.ndarray:
        """Project 3D points onto the floor plane."""
        # Calculate distance from each point to the floor plane
        distances_to_plane = np.dot(points, floor_normal) - floor_offset
        
        # Project points onto floor plane by moving them along the normal
        projected_points = points - distances_to_plane[:, np.newaxis] * floor_normal
        
        return projected_points

    def _find_closest_point_current(self, point_cloud: np.ndarray, current_position: np.ndarray,
                                   current_quaternion: Optional[np.ndarray] = None,
                                   floor_normal: Optional[np.ndarray] = None,
                                   floor_offset: Optional[float] = None,
                                   step_nr: int = 0) -> Dict[str, Any]:
        """
        Find the closest point in the point cloud for the current camera position.
        """
        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(point_cloud)
        
        # If floor distance is enabled, also build a 2D tree for floor projections
        floor_tree = None
        projected_cloud = None
        if self.use_floor_distance and floor_normal is not None and floor_offset is not None:
            projected_cloud = self._project_points_to_floor(point_cloud, floor_normal, floor_offset)
            floor_tree = cKDTree(projected_cloud[:, :2])
        
        # Filter points by view cone if enabled
        view_mask = None
        search_points = point_cloud
        search_tree = tree
        original_indices = np.arange(len(point_cloud))
        
        if self.use_view_cone and current_quaternion is not None:
            view_mask = self._filter_points_in_view_cone(
                point_cloud, current_position, current_quaternion
            )
            
            if np.any(view_mask):
                # Search only within visible points
                search_points = point_cloud[view_mask]
                original_indices = np.where(view_mask)[0]
                search_tree = cKDTree(search_points)
            else:
                # No points in view cone, fall back to global search
                print(f"Warning: No points in view cone at step {step_nr}, using global search")
        
        # Find closest 3D point
        dist_3d, local_idx = search_tree.query(current_position)
        original_idx = original_indices[local_idx]
        closest_point_3d = point_cloud[original_idx]
        
        print(f"Step {step_nr}: Closest 3D point at distance {dist_3d:.3f}m")
        
        # Prepare results
        results = {
            "closest_point_3d": closest_point_3d,
            "closest_point_index": original_idx,
            "distance_3d": dist_3d,
            "distances_array": np.array([dist_3d])  # Single-element array for compatibility
        }
        
        # Calculate floor distance if enabled
        if self.use_floor_distance and floor_tree is not None and projected_cloud is not None:
            # Project current camera position to floor plane
            projected_camera = self._project_points_to_floor(
                current_position.reshape(1, -1), floor_normal, floor_offset
            )[0]
            
            if self.use_view_cone and view_mask is not None and np.any(view_mask):
                # Build 2D tree for visible points projected to floor
                visible_projected = projected_cloud[view_mask]
                visible_floor_tree = cKDTree(visible_projected[:, :2])
                floor_dist, floor_local_idx = visible_floor_tree.query(projected_camera[:2])
                
                # Map back to original point
                floor_original_idx = original_indices[floor_local_idx]
                floor_projected_point = projected_cloud[floor_original_idx]
            else:
                # Use global floor search
                floor_dist, floor_idx = floor_tree.query(projected_camera[:2])
                floor_projected_point = projected_cloud[floor_idx]
            
            print(f"Step {step_nr}: Closest floor distance {floor_dist:.3f}m")
            
            results.update({
                "closest_point_floor": floor_projected_point,
                "distance_floor": floor_dist,
                "projected_point": floor_projected_point,
                "floor_distances_array": np.array([floor_dist])
            })
        
        if self.use_view_cone:
            results["view_cone_mask"] = view_mask
        
        return results 