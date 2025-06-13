from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from ..abstract_pipeline_component import AbstractPipelineComponent


class FloorDetectionComponent(AbstractPipelineComponent):
    """
    Component for detecting floor planes from point cloud and camera poses
    
    Args:
        sample_ratio: Ratio of points to sample for floor detection (default: 0.1)
        ransac_threshold: Distance threshold for RANSAC inlier detection (default: 0.05)
        min_inliers: Minimum number of inliers required for a valid plane (default: 1000)
        n_movement_poses: Number of first poses to use for movement direction analysis (default: 10)
        floor_threshold: Distance threshold for identifying floor points (default: 0.05)
    
    Returns:
        Dictionary containing floor detection results
    
    Raises:
        ValueError: If insufficient data for floor detection
    """
    
    def __init__(self, sample_ratio: float = 0.1, ransac_threshold: float = 0.05,
                 min_inliers: int = 1000, n_movement_poses: int = 10,
                 floor_threshold: float = 0.05) -> None:
        super().__init__()
        self.sample_ratio = sample_ratio
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.n_movement_poses = n_movement_poses
        self.floor_threshold = floor_threshold

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point cloud and camera trajectory data."""
        return ["point_cloud", "camera_positions", "camera_quaternions"]

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs floor detection results."""
        return ["floor_normal", "floor_offset", "floor_threshold", "floor_points"]

    def _run(self, point_cloud, camera_positions: np.ndarray, 
             camera_quaternions: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """
        Detect the floor plane from point cloud and camera data.
        
        Args:
            point_cloud: Point cloud data entity
            camera_positions: Nx3 array of camera positions
            camera_quaternions: Nx4 array of camera quaternions [x, y, z, w]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing floor detection results
        """
        # Extract point cloud data
        if hasattr(point_cloud, 'point_cloud_numpy'):
            points = point_cloud.point_cloud_numpy
        else:
            points = point_cloud
            
        print("Detecting floor plane using gravity and camera position analysis...")
        
        # Detect floor plane
        floor_normal, floor_offset = self._detect_floor_plane(
            points, camera_positions, camera_quaternions
        )
        
        # Identify floor points
        floor_points = self._identify_floor_points(points, floor_normal, floor_offset)
        
        return {
            "floor_normal": floor_normal,
            "floor_offset": floor_offset,
            "floor_threshold": self.floor_threshold,
            "floor_points": floor_points
        }

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

    def _detect_floor_plane(self, point_cloud: np.ndarray, camera_positions: np.ndarray, 
                           camera_rotations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect the floor plane by finding the largest horizontal plane that supports the camera positions.
        """
        # Step 1: Estimate gravity direction from camera orientations
        n_poses = min(self.n_movement_poses, len(camera_positions))
        if n_poses < 1:
            print("Warning: No camera poses available, using default")
            return np.array([0, 0, 1]), 0.0
        
        # Get gravity direction from camera orientations
        gravity_directions = []
        for i in range(n_poses):
            R = self._quaternion_to_rotation_matrix(camera_rotations[i])
            camera_down = -R[:, 1]  # Negative Y-axis (down direction in camera frame)
            gravity_directions.append(camera_down)
        
        # Average gravity direction
        avg_gravity = np.mean(gravity_directions, axis=0)
        if np.linalg.norm(avg_gravity) > 1e-6:
            avg_gravity = avg_gravity / np.linalg.norm(avg_gravity)
        else:
            print("Warning: Could not determine gravity direction, using default")
            avg_gravity = np.array([0, 0, -1])
        
        # Floor normal should be opposite to gravity
        expected_floor_normal = -avg_gravity
        
        print(f"Estimated gravity direction: {avg_gravity}")
        print(f"Expected floor normal: {expected_floor_normal}")
        
        # Step 2: Find points that could be floor (below camera positions)
        camera_min_height = np.min(camera_positions[:, 2]) if camera_positions.shape[1] > 2 else np.min(camera_positions[:, 1])
        
        # Try different coordinate interpretations (Z-up or Y-up)
        height_coords = [point_cloud[:, 2], point_cloud[:, 1]]
        best_candidate_points = None
        best_height_axis = 2
        
        for height_axis, heights in enumerate([2, 1]):
            below_camera = point_cloud[heights < camera_min_height]
            if len(below_camera) > len(best_candidate_points) if best_candidate_points is not None else True:
                best_candidate_points = below_camera
                best_height_axis = heights
        
        if best_candidate_points is None or len(best_candidate_points) < self.min_inliers:
            print("Warning: Not enough points below camera level, using all points")
            best_candidate_points = point_cloud
        
        print(f"Using {len(best_candidate_points)} candidate points below camera level")
        
        # Step 3: Find the dominant plane among floor candidates using RANSAC
        return self._ransac_floor_detection(best_candidate_points, camera_positions, expected_floor_normal)

    def _ransac_floor_detection(self, sampled_points: np.ndarray, camera_positions: np.ndarray,
                               expected_floor_normal: np.ndarray) -> Tuple[np.ndarray, float]:
        """RANSAC-based floor plane detection."""
        # Sample random points for RANSAC
        n_samples = min(int(len(sampled_points) * self.sample_ratio), 15000)
        if len(sampled_points) < n_samples:
            sample_indices = np.arange(len(sampled_points))
        else:
            sample_indices = np.random.choice(len(sampled_points), n_samples, replace=False)
        
        points_subset = sampled_points[sample_indices]
        
        best_normal = None
        best_offset = None
        best_score = -1
        
        # RANSAC iterations
        max_iterations = 2000
        for iteration in range(max_iterations):
            if len(points_subset) < 3:
                break
                
            # Sample 3 random points to define a plane
            sample_indices = np.random.choice(len(points_subset), 3, replace=False)
            p1, p2, p3 = points_subset[sample_indices]
            
            # Calculate plane normal and offset
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            # Normalize
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            
            # Ensure normal points upward (opposite to gravity)
            if np.dot(normal, expected_floor_normal) < 0:
                normal = -normal
                
            offset = np.dot(normal, p1)
            
            # Evaluate plane quality
            distances = np.abs(np.dot(points_subset, normal) - offset)
            inliers = np.sum(distances < self.ransac_threshold)
            
            # Check how horizontal the plane is
            horizontality = abs(np.dot(normal, expected_floor_normal))
            
            # Check if the plane is below the cameras
            camera_distances_to_plane = np.dot(camera_positions, normal) - offset
            cameras_above_plane = np.sum(camera_distances_to_plane > 0.1)
            camera_support_score = cameras_above_plane / len(camera_positions)
            
            # Combined score
            score = (inliers * horizontality * 10 + 
                    horizontality * 1000 + 
                    camera_support_score * 500)
            
            # Only consider reasonably horizontal planes
            if horizontality > 0.7 and score > best_score and inliers >= self.min_inliers // 4:
                best_score = score
                best_normal = normal
                best_offset = offset
            
            # Early termination for excellent solutions
            if horizontality > 0.95 and camera_support_score > 0.8 and inliers > self.min_inliers:
                print(f"Found excellent horizontal floor plane at iteration {iteration}")
                break
        
        # Fallback if no good plane found
        if best_normal is None:
            print("Warning: Floor detection failed, using gravity-based estimate")
            best_normal = expected_floor_normal
            lowest_camera_pos = camera_positions[np.argmin(camera_positions[:, 2])]
            best_offset = np.dot(best_normal, lowest_camera_pos) - 0.5
        
        # Validation output
        horizontality = abs(np.dot(best_normal, expected_floor_normal))
        camera_distances_to_plane = np.dot(camera_positions, best_normal) - best_offset
        cameras_above = np.sum(camera_distances_to_plane > 0)
        
        print(f"Floor detected:")
        print(f"  - Normal: {best_normal}")
        print(f"  - Offset: {best_offset:.3f}")
        print(f"  - Horizontality: {horizontality:.3f}")
        print(f"  - Cameras above plane: {cameras_above}/{len(camera_positions)}")
        
        return best_normal, best_offset

    def _identify_floor_points(self, points: np.ndarray, floor_normal: np.ndarray, 
                              floor_offset: float) -> np.ndarray:
        """Identify points that belong to the floor plane."""
        distances_to_floor = np.abs(np.dot(points, floor_normal) - floor_offset)
        floor_mask = distances_to_floor < self.floor_threshold
        return points[floor_mask] 