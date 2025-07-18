from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from ..abstract_pipeline_component import AbstractPipelineComponent


class IncrementalClosestPointFinderComponent(AbstractPipelineComponent):
    """
    Component for finding closest points to the current camera position in accumulated point cloud.
    Works incrementally and waits for floor detection to be available before calculating floor distances.
    Always filters out points that are on the floor plane.
    Optionally filters to only consider segmented objects when segmentation data is available.
    
    Args:
        use_view_cone: Enable view cone filtering (default: False)
        cone_angle_deg: Half-angle of view cone in degrees (default: 90.0)
        floor_distance_threshold: Maximum distance from floor plane to consider a point as "on floor" (default: 0.05)
        n_closest_points: Number of closest points to return (default: 20)
        use_segmentation_filter: Enable filtering based on segmentation masks when available (default: True)
    
    Returns:
        Dictionary containing closest point analysis results for current position
    
    Raises:
        ValueError: If invalid configuration or insufficient data
    """
    
    def __init__(self, 
                 use_view_cone: bool = False, 
                 cone_angle_deg: float = 90.0,
                 floor_distance_threshold: float = 0.05,
                 n_closest_points: int = 20,
                 use_segmentation_filter: bool = True) -> None:
        super().__init__()
        self.use_view_cone = use_view_cone
        self.cone_angle_deg = cone_angle_deg
        self.floor_distance_threshold = floor_distance_threshold
        self.n_closest_points = n_closest_points
        self.use_segmentation_filter = use_segmentation_filter
        
        # Persistent mapping from segment ID to class label
        self.segment_id_to_class_mapping = {}

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires point cloud and current camera data."""
        inputs = ["point_cloud", "camera_pose", "step_nr", "floor_normal", "floor_offset"]
        
        # Add segmentation inputs if filtering is enabled
        if self.use_segmentation_filter:
            inputs.extend(["image_segmentation_mask", "segmentation_labels"])
            
        return inputs
    

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
            
        # Add segment IDs if segmentation filtering is enabled
        if self.use_segmentation_filter:
            outputs.append("n_closest_points_segment_ids")
            outputs.append("n_closest_points_class_labels")
            
        return outputs
    
    def _get_camera_position(self, camera_pose):
        """Extract camera position from camera pose."""
        # T_WC is a lietorch.Sim3 object, we need to get the translation part
        try:
            # Get the camera position in world coordinates
            # For lietorch.Sim3, we can access the translation part
            camera_position = camera_pose.translation()[:,0:3].cpu().numpy().reshape(3)
            return camera_position
        except AttributeError:
            # Fallback: if it's a matrix, extract translation
            if hasattr(camera_pose, 'cpu'):
                pose_matrix = camera_pose.cpu().numpy()
            else:
                pose_matrix = np.array(camera_pose)
            
            if pose_matrix.shape == (4, 4):
                return pose_matrix[:3, 3]
            else:
                print(f"Warning: Unsupported camera pose format: {type(camera_pose)}")
                return None

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

    def _filter_segmented_points(self, points_3d, point_cloud, 
                                 image_segmentation_mask=None, segmentation_labels=None):
        """
        Filter points to only include those that belong to segmented objects.
        
        Args:
            points_3d: Array of 3D points [N, 3]
            point_cloud: Point cloud data entity with potential segmentation data
            image_segmentation_mask: Optional image segmentation mask (not used for accumulated point clouds)
            segmentation_labels: Optional mapping from segment IDs to class labels (not used for filtering)
            
        Returns:
            np.ndarray: Boolean mask indicating which points belong to segmented objects
        """
        if len(points_3d) == 0:
            return np.array([], dtype=bool)
        
        # Check if point cloud has segmentation data embedded
        if hasattr(point_cloud, 'segmentation_mask_numpy') and point_cloud.segmentation_mask_numpy is not None:
            point_segmentation_mask = point_cloud.segmentation_mask_numpy.flatten()
            
            # Check if sizes match
            if len(point_segmentation_mask) != len(points_3d):
                # Try to handle size mismatch
                if len(point_segmentation_mask) > len(points_3d):
                    point_segmentation_mask = point_segmentation_mask[:len(points_3d)]
                else:
                    return np.zeros(len(points_3d), dtype=bool)
            
            # Find all object points (segment_id > 0)
            segmented_mask = point_segmentation_mask > 0
            
            # Return mask indicating which points belong to segmented objects
            return segmented_mask
        
        # No point cloud segmentation data available
        return np.zeros(len(points_3d), dtype=bool)

    def _get_closest_points_segmentation_data(self, closest_indices, point_cloud, segmentation_labels=None):
        """
        Get segment IDs and class labels for the closest points using persistent mapping.
        
        Args:
            closest_indices: Indices of the closest points in the original point cloud
            point_cloud: Point cloud data entity with segmentation data
            segmentation_labels: Optional mapping from segment IDs to class labels (updates persistent mapping)
            
        Returns:
            Tuple of (segment_ids, class_labels) lists
        """
        if len(closest_indices) == 0:
            return [], []
        
        # Update persistent mapping with current frame's segmentation_labels
        if segmentation_labels is not None:
            self.segment_id_to_class_mapping.update(segmentation_labels)
        
        # Try to get segmentation mask from point cloud
        if hasattr(point_cloud, 'segmentation_mask_numpy') and point_cloud.segmentation_mask_numpy is not None:
            point_segmentation_mask = point_cloud.segmentation_mask_numpy.flatten()
            
            # Get segment IDs for the closest points (single lookup)
            closest_segment_ids = point_segmentation_mask[closest_indices].tolist()
            
            # Map segment IDs to class labels using persistent mapping
            class_labels = [
                self.segment_id_to_class_mapping.get(seg_id, "unknown") 
                for seg_id in closest_segment_ids
            ]
            
            return closest_segment_ids, class_labels
        
        # No segmentation data available
        empty_segment_ids = [-1] * len(closest_indices)
        empty_class_labels = ["unknown"] * len(closest_indices)
        return empty_segment_ids, empty_class_labels

    def _add_segmentation_outputs(self, outputs, segment_ids=None, class_labels=None):
        """
        Add segmentation outputs to results dictionary if segmentation filtering is enabled.
        
        Args:
            outputs: Dictionary to add segmentation data to
            segment_ids: List of segment IDs (defaults to empty list)
            class_labels: List of class labels (defaults to empty list)
        """
        if self.use_segmentation_filter:
            outputs["n_closest_points_segment_ids"] = segment_ids or []
            outputs["n_closest_points_class_labels"] = class_labels or []

    def _run(self, 
             point_cloud, 
             camera_pose, 
             step_nr: int,
             floor_normal: Optional[np.ndarray] = None,
             floor_offset: Optional[float] = None,
             image_segmentation_mask=None,
             segmentation_labels=None,
             **kwargs: Any) -> Dict[str, Any]:
        """
        Find closest point for the current camera position.
        
        Args:
            point_cloud: Point cloud data entity (accumulated)
            camera_pose: Current camera pose object (T_WC)
            step_nr: Current step number
            floor_normal: Floor plane normal (if floor distance enabled)
            floor_offset: Floor plane offset (if floor distance enabled)
            image_segmentation_mask: Optional image segmentation mask
            segmentation_labels: Optional mapping from segment IDs to class labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing closest point analysis results for current position
        """
        
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
            self._add_segmentation_outputs(outputs)
            return outputs
        
        # Extract camera position from T_WC pose
        camera_position = self._get_camera_position(camera_pose)
        if camera_position is None:
            raise ValueError("Invalid camera pose - cannot extract position")
        
        # Start with all points
        current_points = points_3d.copy()
        current_indices = np.arange(len(points_3d))
        
        # Apply floor filtering if floor data is available
        floor_filtered_mask = np.ones(len(points_3d), dtype=bool)
        if floor_normal is not None and floor_offset is not None:
            floor_filtered_mask = self._filter_floor_points(points_3d, floor_normal, floor_offset)
            
            if not np.any(floor_filtered_mask):
                # No points pass floor filtering
                outputs = {
                    "n_closest_points_3d": np.empty((0, 3), dtype=float),
                    "n_closest_points_index": np.empty((0,), dtype=int),
                    "n_closest_points_distance_2d": np.empty((0,), dtype=float),
                    "floor_filtered_mask": floor_filtered_mask,
                }
                if self.use_view_cone:
                    outputs["view_cone_mask"] = np.array([])
                self._add_segmentation_outputs(outputs)
                return outputs
            
            # Apply floor filtering
            current_points = points_3d[floor_filtered_mask]
            current_indices = np.where(floor_filtered_mask)[0]
        # No floor filtering applied
        
        # Apply segmentation filtering if enabled (only filter if we have actual segmentation data)
        if self.use_segmentation_filter:
            segmentation_filtered_mask = self._filter_segmented_points(
                points_3d, point_cloud, image_segmentation_mask, segmentation_labels
            )
            
            # Combine segmentation filter with floor filter (if available)
            if floor_normal is not None and floor_offset is not None:
                combined_floor_seg_mask = floor_filtered_mask & segmentation_filtered_mask
            else:
                combined_floor_seg_mask = segmentation_filtered_mask
            
            if not np.any(combined_floor_seg_mask):
                # No points pass filtering
                outputs = {
                    "n_closest_points_3d": np.empty((0, 3), dtype=float),
                    "n_closest_points_index": np.empty((0,), dtype=int),
                    "n_closest_points_distance_2d": np.empty((0,), dtype=float),
                    "floor_filtered_mask": floor_filtered_mask,
                }
                if self.use_view_cone:
                    outputs["view_cone_mask"] = np.array([])
                self._add_segmentation_outputs(outputs)
                return outputs
            
            # Update points and indices to reflect both filters
            current_points = points_3d[combined_floor_seg_mask]
            current_indices = np.where(combined_floor_seg_mask)[0]
        else:
            combined_floor_seg_mask = floor_filtered_mask # No segmentation filter, so floor filter is the only one
        
        # Apply view cone filtering if enabled
        if self.use_view_cone:
            camera_direction = self._get_camera_direction(camera_pose)
            view_cone_mask_subset = self._apply_view_cone_filter(current_points, camera_position, camera_direction, step_nr)
            
            # Filter points to only those in view cone
            if not np.any(view_cone_mask_subset):
                # No points in view cone
                outputs = {
                    "n_closest_points_3d": np.empty((0, 3), dtype=float),
                    "n_closest_points_index": np.empty((0,), dtype=int),
                    "n_closest_points_distance_2d": np.empty((0,), dtype=float),
                    "floor_filtered_mask": floor_filtered_mask,
                    "view_cone_mask": np.zeros(len(points_3d), dtype=bool),
                }
                self._add_segmentation_outputs(outputs)
                return outputs
            
            current_points = current_points[view_cone_mask_subset]
            current_indices = current_indices[view_cone_mask_subset]
            
            # Create full view cone mask for original point cloud
            view_cone_mask_full = np.zeros(len(points_3d), dtype=bool)
            view_cone_mask_full[current_indices] = True
        else:
            view_cone_mask_full = np.ones(len(points_3d), dtype=bool)
        
        # Calculate distances based on available data
        if floor_normal is not None and floor_offset is not None:
            # Use floor-projected 2D distances when floor data is available
            # Project all points onto the floor plane
            floor_normal = np.array(floor_normal).reshape(3)
            floor_normal = floor_normal / np.linalg.norm(floor_normal)  # Normalize
            
            # Project points onto floor plane
            # For each point p, projected point = p - (dot(p-floor_point, floor_normal)) * floor_normal
            # where floor_point is any point on the plane: floor_normal * floor_offset
            floor_point = floor_normal * floor_offset
            
            # Project all filtered points
            points_to_floor = current_points - floor_point[np.newaxis, :]
            distances_to_plane = np.dot(points_to_floor, floor_normal)
            points_projected = current_points - distances_to_plane[:, np.newaxis] * floor_normal[np.newaxis, :]
            
            # Project camera position onto floor plane
            camera_to_floor = camera_position - floor_point
            camera_distance_to_plane = np.dot(camera_to_floor, floor_normal)
            camera_projected = camera_position - camera_distance_to_plane * floor_normal
            
            # Calculate 2D distances from camera to all projected points
            distances_2d = np.linalg.norm(points_projected - camera_projected[np.newaxis, :], axis=1)
        else:
            # Use simple 3D euclidean distances when floor data is not available
            distances_2d = np.linalg.norm(current_points - camera_position[np.newaxis, :], axis=1)
        
        # Find the n closest points
        n_points = min(self.n_closest_points, len(current_points))
        if n_points == 0:
            # No points available after filtering
            outputs = {
                "n_closest_points_3d": np.empty((0, 3), dtype=float),
                "n_closest_points_index": np.empty((0,), dtype=int),
                "n_closest_points_distance_2d": np.empty((0,), dtype=float),
                "floor_filtered_mask": floor_filtered_mask,
            }
            if self.use_view_cone:
                outputs["view_cone_mask"] = view_cone_mask_full
            self._add_segmentation_outputs(outputs)
            return outputs
        elif n_points >= len(distances_2d):
            # All points are closest points, no need for argpartition
            closest_indices = np.arange(len(distances_2d))
        else:
            # Use argpartition when we have more points than needed
            closest_indices = np.argpartition(distances_2d, n_points)[:n_points]
        
        # Sort the closest indices by distance
        sorted_closest_indices = closest_indices[np.argsort(distances_2d[closest_indices])]
        
        # Get the results (map back to original indices)
        closest_points_3d = current_points[sorted_closest_indices]
        closest_distances_2d = distances_2d[sorted_closest_indices]
        closest_original_indices = current_indices[sorted_closest_indices]
        
        outputs = {
            "n_closest_points_3d": closest_points_3d,
            "n_closest_points_index": closest_original_indices,
            "n_closest_points_distance_2d": closest_distances_2d,
            "floor_filtered_mask": floor_filtered_mask,
        }
        
        # Add view cone mask if enabled
        if self.use_view_cone:
            outputs["view_cone_mask"] = view_cone_mask_full
        
        # Add segmentation data if enabled
        if self.use_segmentation_filter:
            segment_ids, class_labels = self._get_closest_points_segmentation_data(
                closest_original_indices, point_cloud, segmentation_labels
            )
            outputs["n_closest_points_segment_ids"] = segment_ids
            outputs["n_closest_points_class_labels"] = class_labels
        
        return outputs
        