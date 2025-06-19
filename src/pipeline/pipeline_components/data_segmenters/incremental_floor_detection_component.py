from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from ..abstract_pipeline_component import AbstractPipelineComponent


class IncrementalFloorDetectionComponent(AbstractPipelineComponent):
    """
    Component for incrementally detecting floor planes from point cloud and camera poses.
    Waits for a minimum number of frames before starting and can refine the floor plane over time.
    
    Args:
        min_frames: Minimum number of frames before starting floor detection (default: 3)
        sample_ratio: Ratio of points to sample for floor detection (default: 0.1)
        ransac_threshold: Distance threshold for RANSAC inlier detection (default: 0.05)
        min_inliers: Minimum number of inliers required for a valid plane (default: 1000)
        floor_threshold: Distance threshold for identifying floor points (default: 0.05)
        refine_interval: How often to refine the floor plane (every N frames, 0 = no refinement) (default: 0)
        max_refinement_poses: Maximum number of poses to use for refinement (default: 20)
    
    Returns:
        Dictionary containing floor detection results or empty dict if not ready
    
    Raises:
        ValueError: If insufficient data for floor detection
    """
    
    def __init__(self, min_frames: int = 3, sample_ratio: float = 0.1, 
                 ransac_threshold: float = 0.05, min_inliers: int = 1000,
                 floor_threshold: float = 0.05, refine_interval: int = 0,
                 max_refinement_poses: int = 20) -> None:
        super().__init__()
        self.min_frames = min_frames
        self.sample_ratio = sample_ratio
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.floor_threshold = floor_threshold
        self.refine_interval = refine_interval
        self.max_refinement_poses = max_refinement_poses
        
        # State tracking
        self.frame_count = 0
        self.accumulated_poses = []
        self.accumulated_quaternions = []
        self.floor_normal = None
        self.floor_offset = None
        self.floor_detected = False
        self.last_refinement_frame = 0

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point cloud and camera pose data."""
        return ["point_cloud", "camera_pose", "step_nr"]

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs floor detection results when available."""
        return ["floor_normal", "floor_offset", "floor_threshold", "floor_points"]

    def _run(self, point_cloud, camera_pose, step_nr: int, **kwargs: Any) -> Dict[str, Any]:
        """
        Incrementally detect or refine the floor plane.
        
        Args:
            point_cloud: Point cloud data entity
            camera_pose: Current camera pose object
            step_nr: Current step number
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing floor detection results (empty if not ready)
        """
        self.frame_count += 1
        
        # Extract current camera position and quaternion from pose
        current_position = self._extract_camera_position(camera_pose)
        current_quaternion = self._extract_camera_quaternion(camera_pose)
        
        if current_position is None or current_quaternion is None:
            print(f"Warning: Could not extract camera pose at step {step_nr}")
            return {}
        
        # Accumulate camera poses
        self.accumulated_poses.append(current_position.copy())
        self.accumulated_quaternions.append(current_quaternion.copy())
        
        # Keep only the most recent poses for refinement
        if len(self.accumulated_poses) > self.max_refinement_poses:
            self.accumulated_poses = self.accumulated_poses[-self.max_refinement_poses:]
            self.accumulated_quaternions = self.accumulated_quaternions[-self.max_refinement_poses:]
        
        # Extract point cloud data
        if hasattr(point_cloud, 'point_cloud_numpy'):
            points = point_cloud.point_cloud_numpy
        else:
            points = point_cloud
        
        # Check if we should detect or refine the floor
        should_detect = False
        should_refine = False
        
        if not self.floor_detected and self.frame_count >= self.min_frames:
            should_detect = True
            print(f"Starting floor detection at frame {self.frame_count} with {len(self.accumulated_poses)} poses")
        elif (self.floor_detected and self.refine_interval > 0 and 
              (self.frame_count - self.last_refinement_frame) >= self.refine_interval):
            should_refine = True
            print(f"Refining floor detection at frame {self.frame_count}")
            self.last_refinement_frame = self.frame_count
        
        if should_detect or should_refine:
            try:
                # Convert accumulated poses to arrays
                poses_array = np.array(self.accumulated_poses)
                quaternions_array = np.array(self.accumulated_quaternions)
                
                # Detect floor plane
                floor_normal, floor_offset = self._detect_floor_plane(
                    points, poses_array, quaternions_array
                )
                
                if floor_normal is not None and floor_offset is not None:
                    self.floor_normal = floor_normal
                    self.floor_offset = floor_offset
                    self.floor_detected = True
                    
                    # Identify floor points
                    floor_points = self._identify_floor_points(points, floor_normal, floor_offset)
                    
                    action = "detected" if should_detect else "refined"
                    print(f"Floor plane {action} successfully at frame {self.frame_count}")
                    
                    return {
                        "floor_normal": floor_normal,
                        "floor_offset": floor_offset,
                        "floor_threshold": self.floor_threshold,
                        "floor_points": floor_points
                    }
                else:
                    print(f"Floor detection failed at frame {self.frame_count}")
            except Exception as e:
                print(f"Error during floor detection at frame {self.frame_count}: {e}")
        
        # If we have a previously detected floor, return it
        if self.floor_detected:
            floor_points = self._identify_floor_points(points, self.floor_normal, self.floor_offset)
            return {
                "floor_normal": self.floor_normal,
                "floor_offset": self.floor_offset,
                "floor_threshold": self.floor_threshold,
                "floor_points": floor_points
            }
        
        # Not ready yet
        return {}

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

    def _detect_floor_plane(self, point_cloud: np.ndarray, camera_positions: np.ndarray, 
                           camera_rotations: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Detect the floor plane by finding the largest horizontal plane that supports the camera positions.
        """
        n_poses = len(camera_positions)
        if n_poses < 1:
            print("Warning: No camera poses available")
            return None, None
        
        # Step 1: Estimate gravity direction from camera orientations
        gravity_directions = []
        for i in range(min(n_poses, 10)):  # Use up to 10 poses for gravity estimation
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
        
        for height_axis, heights in enumerate([2, 1]):
            below_camera = point_cloud[heights < camera_min_height]
            if len(below_camera) > len(best_candidate_points) if best_candidate_points is not None else True:
                best_candidate_points = below_camera
        
        if best_candidate_points is None or len(best_candidate_points) < self.min_inliers:
            print("Warning: Not enough points below camera level, using all points")
            best_candidate_points = point_cloud
        
        print(f"Using {len(best_candidate_points)} candidate points for floor detection")
        
        # Step 3: Find the dominant plane among floor candidates using RANSAC
        return self._ransac_floor_detection(best_candidate_points, camera_positions, expected_floor_normal)

    def _ransac_floor_detection(self, sampled_points: np.ndarray, camera_positions: np.ndarray,
                               expected_floor_normal: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
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
        max_iterations = 1000  # Reduced for incremental processing
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
            print("Warning: Floor detection failed, no suitable plane found")
            return None, None
        
        # Validation output
        horizontality = abs(np.dot(best_normal, expected_floor_normal))
        camera_distances_to_plane = np.dot(camera_positions, best_normal) - best_offset
        cameras_above = np.sum(camera_distances_to_plane > 0)
        
        print(f"Floor plane quality:")
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