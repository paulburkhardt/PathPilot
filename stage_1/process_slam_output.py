#!/usr/bin/env python3
"""
PathPilot: Mast3r-Slam Output Processing Pipeline
Processes Mast3r-Slam outputs to create a Rerun visualization showing:
- 3D point cloud reconstruction
- Camera trajectory through the scene  
- Closest point to camera at each pose
- Distance measurements over time
"""

import argparse
import numpy as np
import pathlib
from typing import Tuple, Optional
import rerun as rr
from plyfile import PlyData
from scipy.spatial import cKDTree
import uuid


def load_point_cloud(ply_path: pathlib.Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load point cloud from PLY file.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (if available)
    """
    print(f"Loading point cloud from: {ply_path}")
    
    ply_data = PlyData.read(str(ply_path))
    vertex_element = ply_data['vertex']
    
    # Access the actual data array
    vertices = vertex_element.data
    
    # Extract 3D coordinates
    points = np.column_stack([
        vertices['x'].astype(np.float32),
        vertices['y'].astype(np.float32), 
        vertices['z'].astype(np.float32)
    ])
    
    # Extract colors if available
    colors = None
    try:
        # Check if color properties exist in the dtype
        if hasattr(vertices.dtype, 'names') and vertices.dtype.names is not None:
            if 'red' in vertices.dtype.names:
                colors = np.column_stack([
                    vertices['red'].astype(np.uint8),
                    vertices['green'].astype(np.uint8),
                    vertices['blue'].astype(np.uint8)
                ])
    except Exception as e:
        print(f"Warning: Could not extract colors from PLY file: {e}")
        colors = None
    
    print(f"Loaded {len(points)} points with {'colors' if colors is not None else 'no colors'}")
    return points, colors


def load_camera_trajectory(txt_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load camera trajectory from TXT file.
    
    Args:
        txt_path: Path to the trajectory TXT file
        
    Returns:
        timestamps: Array of timestamps
        positions: Nx3 array of camera positions
        quaternions: Nx4 array of camera orientations (x,y,z,w)
    """
    print(f"Loading camera trajectory from: {txt_path}")
    
    # Read the file line by line to handle formatting issues
    valid_lines = []
    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Split the line into values
            values = line.split()
            
            # Only keep lines with exactly 8 values
            if len(values) == 8:
                try:
                    # Try to convert to float to validate
                    float_values = [float(v) for v in values]
                    valid_lines.append(float_values)
                except ValueError:
                    print(f"Warning: Skipping line {line_num} - invalid float values")
            else:
                print(f"Warning: Skipping line {line_num} - has {len(values)} columns instead of 8")
    
    if not valid_lines:
        raise ValueError("No valid trajectory data found in file")
    
    # Convert to numpy array
    data = np.array(valid_lines, dtype=np.float64)
    
    timestamps = data[:, 0]
    positions = data[:, 1:4].astype(np.float32)
    quaternions = data[:, 4:8].astype(np.float32)
    
    print(f"Loaded trajectory with {len(positions)} poses (skipped {line_num - len(positions)} invalid lines)")
    print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} seconds")
    print(f"Position range: [{positions.min(axis=0)}, {positions.max(axis=0)}]")
    
    return timestamps, positions, quaternions


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion as [x, y, z, w]
        
    Returns:
        3x3 rotation matrix
    """
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


def detect_floor_plane(point_cloud: np.ndarray, camera_positions: np.ndarray, 
                      camera_rotations: np.ndarray, sample_ratio: float = 0.1,
                      ransac_threshold: float = 0.05, min_inliers: int = 1000,
                      n_movement_poses: int = 10) -> Tuple[np.ndarray, float]:
    """
    Detect the floor plane by finding the largest horizontal plane that supports the camera positions.
    
    Args:
        point_cloud: Nx3 array of 3D points
        camera_positions: Nx3 array of camera positions (use first few for movement direction)
        camera_rotations: Nx4 array of camera rotation quaternions [x, y, z, w]
        sample_ratio: Ratio of points to sample for floor detection
        ransac_threshold: Distance threshold for RANSAC inlier detection
        min_inliers: Minimum number of inliers required for a valid plane
        n_movement_poses: Number of first poses to use for movement direction analysis
        
    Returns:
        floor_normal: Normal vector of the floor plane
        floor_offset: Plane offset (distance from origin)
    """
    print("Detecting floor plane using gravity and camera position analysis...")
    
    # Step 1: Estimate gravity direction from camera orientations
    n_poses = min(n_movement_poses, len(camera_positions))
    if n_poses < 1:
        print("Warning: No camera poses available, using default")
        return np.array([0, 0, 1]), 0.0
    
    # Get gravity direction from camera orientations (average of camera "down" vectors)
    gravity_directions = []
    for i in range(n_poses):
        R = quaternion_to_rotation_matrix(camera_rotations[i])
        camera_down = -R[:, 1]  # Negative Y-axis (down direction in camera frame)
        gravity_directions.append(camera_down)
    
    # Average gravity direction (should point down towards floor)
    avg_gravity = np.mean(gravity_directions, axis=0)
    if np.linalg.norm(avg_gravity) > 1e-6:
        avg_gravity = avg_gravity / np.linalg.norm(avg_gravity)
    else:
        print("Warning: Could not determine gravity direction, using default")
        avg_gravity = np.array([0, 0, -1])  # Default: negative Z
    
    # Floor normal should be opposite to gravity
    expected_floor_normal = -avg_gravity
    
    print(f"Estimated gravity direction: {avg_gravity}")
    print(f"Expected floor normal: {expected_floor_normal}")
    
    # Step 2: Find points that could be floor (below camera positions)
    camera_min_height = np.min(camera_positions[:, 2]) if camera_positions.shape[1] > 2 else np.min(camera_positions[:, 1])
    
    # Try different coordinate interpretations (Z-up or Y-up)
    height_coords = [point_cloud[:, 2], point_cloud[:, 1]]  # Try Z then Y as height
    best_candidate_points = None
    best_height_axis = 2  # Default to Z
    
    for height_axis, heights in enumerate([2, 1]):  # Z-axis first, then Y-axis
        # Points below camera level
        below_camera = point_cloud[heights < camera_min_height]
        if len(below_camera) > len(best_candidate_points) if best_candidate_points is not None else True:
            best_candidate_points = below_camera
            best_height_axis = heights
    
    if best_candidate_points is None or len(best_candidate_points) < min_inliers:
        print("Warning: Not enough points below camera level, using all points")
        best_candidate_points = point_cloud
    
    print(f"Using {len(best_candidate_points)} candidate points below camera level")
    
    # Step 3: Find the dominant plane among floor candidates
    # Sample random points for RANSAC
    n_samples = min(int(len(best_candidate_points) * sample_ratio), 15000)
    if len(best_candidate_points) < n_samples:
        sampled_points = best_candidate_points
    else:
        sampled_indices = np.random.choice(len(best_candidate_points), n_samples, replace=False)
        sampled_points = best_candidate_points[sampled_indices]
    
    best_normal = None
    best_offset = None
    best_inlier_count = 0
    best_score = -1
    
    # RANSAC iterations
    max_iterations = 2000
    for iteration in range(max_iterations):
        # Sample 3 random points to define a plane
        if len(sampled_points) < 3:
            break
            
        sample_indices = np.random.choice(len(sampled_points), 3, replace=False)
        p1, p2, p3 = sampled_points[sample_indices]
        
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
        
        # Step 4: Evaluate plane quality
        # Count inliers in the sampled points
        distances = np.abs(np.dot(sampled_points, normal) - offset)
        inliers = np.sum(distances < ransac_threshold)
        
        # Check how horizontal the plane is (should be perpendicular to gravity)
        horizontality = abs(np.dot(normal, expected_floor_normal))  # Should be close to 1
        
        # Check if the plane is below the cameras (floor should support cameras)
        camera_distances_to_plane = np.dot(camera_positions, normal) - offset
        cameras_above_plane = np.sum(camera_distances_to_plane > 0.1)  # Cameras should be above floor
        camera_support_score = cameras_above_plane / len(camera_positions)
        
        # Prefer planes that are:
        # 1. Horizontal (high horizontality score)
        # 2. Have many inliers
        # 3. Support the cameras (cameras are above the plane)
        score = (inliers * horizontality * 10 + 
                 horizontality * 1000 + 
                 camera_support_score * 500)
        
        # Only consider planes that are reasonably horizontal
        if horizontality > 0.7 and score > best_score and inliers >= min_inliers // 4:
            best_score = score
            best_inlier_count = inliers
            best_normal = normal
            best_offset = offset
        
        # Early termination if we find a very good horizontal solution
        if horizontality > 0.95 and camera_support_score > 0.8 and inliers > min_inliers:
            print(f"Found excellent horizontal floor plane at iteration {iteration}")
            break
    
    if best_normal is None:
        print("Warning: Floor detection failed, using gravity-based estimate")
        # Fallback: create a horizontal plane at the lowest camera position
        best_normal = expected_floor_normal
        lowest_camera_pos = camera_positions[np.argmin(camera_positions[:, best_height_axis])]
        best_offset = np.dot(best_normal, lowest_camera_pos) - 0.5  # 0.5m below lowest camera
        best_inlier_count = 0
    
    # Final validation
    horizontality = abs(np.dot(best_normal, expected_floor_normal))
    camera_distances_to_plane = np.dot(camera_positions, best_normal) - best_offset
    cameras_above = np.sum(camera_distances_to_plane > 0)
    
    print(f"Floor detected:")
    print(f"  - Normal: {best_normal}")
    print(f"  - Offset: {best_offset:.3f}")
    print(f"  - Inliers: {best_inlier_count}")
    print(f"  - Horizontality: {horizontality:.3f} (should be close to 1.0)")
    print(f"  - Cameras above plane: {cameras_above}/{len(camera_positions)}")
    
    return best_normal, best_offset


def transform_to_floor_coordinates(points: np.ndarray, positions: np.ndarray, quaternions: np.ndarray,
                                 floor_normal: np.ndarray, floor_offset: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform coordinate system so that the floor becomes the XY plane.
    
    Args:
        points: Point cloud
        positions: Camera positions
        quaternions: Camera quaternions
        floor_normal: Floor plane normal vector
        floor_offset: Floor plane offset
        
    Returns:
        Transformed points, positions, and quaternions
    """
    print("Transforming to floor-aligned coordinate system...")
    
    # Create rotation matrix to align floor normal with Z-axis
    z_axis = np.array([0, 0, 1])
    
    # If floor normal is already aligned with Z, no rotation needed
    if np.abs(np.dot(floor_normal, z_axis) - 1.0) < 1e-6:
        translation = np.array([0, 0, -floor_offset])
        return points + translation, positions + translation, quaternions
    
    # Calculate rotation axis and angle
    rotation_axis = np.cross(floor_normal, z_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm < 1e-6:
        # Vectors are parallel but opposite
        rotation_axis = np.array([1, 0, 0])  # Arbitrary axis
        rotation_angle = np.pi
    else:
        rotation_axis = rotation_axis / rotation_axis_norm
        rotation_angle = np.arccos(np.clip(np.dot(floor_normal, z_axis), -1.0, 1.0))
    
    # Create rotation matrix using Rodrigues' formula
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    
    R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
    
    # Transform points and positions
    transformed_points = np.dot(points, R.T)
    transformed_positions = np.dot(positions, R.T)
    
    # Translate so floor is at Z=0
    floor_z = floor_offset / np.linalg.norm(floor_normal)
    transformed_points[:, 2] -= floor_z
    transformed_positions[:, 2] -= floor_z
    
    # Transform quaternions (this is more complex, for now keep original)
    # In practice, we might want to update the quaternions as well
    transformed_quaternions = quaternions.copy()  # Simplified for now
    
    print(f"Coordinate system transformed. Floor is now at Z=0")
    return transformed_points, transformed_positions, transformed_quaternions


def filter_points_in_view_cone(point_cloud: np.ndarray, camera_position: np.ndarray, 
                               camera_rotation: np.ndarray, cone_angle_deg: float = 90.0,
                               max_distance: float = 10.0) -> np.ndarray:
    """
    Filter point cloud to only include points within the camera's view cone.
    
    Args:
        point_cloud: Nx3 array of 3D points
        camera_position: 3D camera position
        camera_rotation: Camera rotation quaternion [x, y, z, w]
        cone_angle_deg: Half-angle of the view cone in degrees (default: 90°)
        max_distance: Maximum distance to consider (default: 10m)
        
    Returns:
        Boolean mask indicating which points are in the view cone
    """
    # Get camera forward direction (typically -Z in camera coordinates)
    R = quaternion_to_rotation_matrix(camera_rotation)
    camera_forward = R[:, 2]  # Z-axis (forward direction)
    
    # Vector from camera to each point
    point_vectors = point_cloud - camera_position
    
    # Distance filter
    distances = np.linalg.norm(point_vectors, axis=1)
    distance_mask = distances <= max_distance
    
    # Normalize point vectors
    normalized_vectors = point_vectors / (distances[:, np.newaxis] + 1e-8)
    
    # Compute dot product with camera forward direction
    dot_products = np.dot(normalized_vectors, camera_forward)
    
    # Convert cone angle to cosine threshold
    cos_threshold = np.cos(np.radians(cone_angle_deg))
    
    # Points are in cone if dot product > cos_threshold
    cone_mask = dot_products > cos_threshold
    
    # Combine distance and cone filters
    return distance_mask & cone_mask


def find_closest_points(point_cloud: np.ndarray, camera_positions: np.ndarray, 
                       camera_quaternions: Optional[np.ndarray] = None,
                       use_view_cone: bool = False, cone_angle_deg: float = 90.0,
                       max_view_distance: float = 10.0,
                       floor_normal: Optional[np.ndarray] = None,
                       floor_offset: Optional[float] = None,
                       use_floor_distance: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the closest point in the point cloud for each camera position.
    
    Args:
        point_cloud: Nx3 array of 3D points
        camera_positions: Mx3 array of camera positions
        camera_quaternions: Mx4 array of camera orientations (optional, required if use_view_cone=True)
        use_view_cone: If True, only consider points within camera's view cone
        cone_angle_deg: Half-angle of the view cone in degrees
        max_view_distance: Maximum distance to consider for view cone
        floor_normal: Floor plane normal vector (optional, for floor distance calculation)
        floor_offset: Floor plane offset (optional, for floor distance calculation)
        use_floor_distance: If True, calculate horizontal distance on floor plane
        
    Returns:
        closest_points: Mx3 array of closest points (3D)
        distances: M array of 3D distances to closest points
        indices: M array of indices of closest points in point_cloud
        floor_distances: M array of horizontal distances on floor plane (if use_floor_distance)
        projected_points: Mx3 array of floor-projected closest points (if use_floor_distance)
    """
    if use_view_cone and camera_quaternions is None:
        raise ValueError("camera_quaternions required when use_view_cone=True")
    
    if use_floor_distance and (floor_normal is None or floor_offset is None):
        raise ValueError("floor_normal and floor_offset required when use_floor_distance=True")
    
    print(f"Building KD-tree for nearest neighbor search...")
    print(f"View cone filtering: {'enabled' if use_view_cone else 'disabled'}")
    print(f"Floor distance calculation: {'enabled' if use_floor_distance else 'disabled'}")
    if use_view_cone:
        print(f"  - Cone half-angle: {cone_angle_deg}°")
        print(f"  - Max view distance: {max_view_distance}m")
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(point_cloud)
    
    # If floor distance is enabled, also build a 2D tree for floor projections
    floor_tree = None
    if use_floor_distance:
        print("Building 2D KD-tree for floor plane distance calculation...")
        # Project all points onto the floor plane
        projected_cloud = project_points_to_floor(point_cloud, floor_normal, floor_offset)
        # Use only X,Y coordinates for 2D distance calculation
        floor_tree = cKDTree(projected_cloud[:, :2])
    
    print("Finding closest points for each camera pose...")
    
    closest_points = []
    distances = []
    indices = []
    floor_distances = []
    projected_points = []
    
    for i, position in enumerate(camera_positions):
        if use_view_cone:
            # Filter points within view cone
            quaternion = camera_quaternions[i]
            view_mask = filter_points_in_view_cone(
                point_cloud, position, quaternion, cone_angle_deg, max_view_distance
            )
            
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
        
        closest_points.append(closest_point)
        distances.append(dist)
        indices.append(original_idx)
        
        # Calculate floor distance if enabled
        if use_floor_distance:
            # Project camera position to floor plane
            projected_camera = project_points_to_floor(position.reshape(1, -1), floor_normal, floor_offset)[0]
            
            if use_view_cone and np.any(view_mask):
                # Build 2D tree for visible points projected to floor
                visible_projected = project_points_to_floor(visible_points, floor_normal, floor_offset)
                visible_floor_tree = cKDTree(visible_projected[:, :2])
                floor_dist, floor_local_idx = visible_floor_tree.query(projected_camera[:2])
                
                # Map back to original point
                floor_closest_point = visible_points[floor_local_idx]
                floor_projected_point = visible_projected[floor_local_idx]
            else:
                # Use global floor search
                floor_dist, floor_idx = floor_tree.query(projected_camera[:2])
                floor_closest_point = point_cloud[floor_idx]
                floor_projected_point = project_points_to_floor(floor_closest_point.reshape(1, -1), floor_normal, floor_offset)[0]
            
            floor_distances.append(floor_dist)
            projected_points.append(floor_projected_point)
    
    closest_points = np.array(closest_points)
    distances = np.array(distances)
    indices = np.array(indices)
    
    print(f"Found closest points. 3D distance range: {distances.min():.3f} to {distances.max():.3f}")
    
    if use_floor_distance:
        floor_distances = np.array(floor_distances)
        projected_points = np.array(projected_points)
        print(f"Floor distance range: {floor_distances.min():.3f} to {floor_distances.max():.3f}")
        return closest_points, distances, indices, floor_distances, projected_points
    else:
        return closest_points, distances, indices, np.array([]), np.array([])


def project_points_to_floor(points: np.ndarray, floor_normal: np.ndarray, floor_offset: float) -> np.ndarray:
    """
    Project 3D points onto the floor plane.
    
    Args:
        points: Nx3 array of 3D points
        floor_normal: Floor plane normal vector
        floor_offset: Floor plane offset
        
    Returns:
        Nx3 array of projected points on the floor plane
    """
    # Calculate distance from each point to the floor plane
    distances_to_plane = np.dot(points, floor_normal) - floor_offset
    
    # Project points onto floor plane by moving them along the normal
    projected_points = points - distances_to_plane[:, np.newaxis] * floor_normal
    
    return projected_points


def setup_rerun(recording_name: str = "PathPilot") -> None:
    """Initialize Rerun with proper coordinate system."""
    rr.init(recording_name, recording_id=uuid.uuid4(), spawn=True)
    
    # Set up right-handed coordinate system (Y up, Z forward)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    print("Initialized Rerun viewer")


def log_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None, 
                    floor_normal: Optional[np.ndarray] = None, floor_offset: Optional[float] = None,
                    floor_threshold: float = 0.05) -> None:
    """Log the 3D point cloud to Rerun."""
    print("Logging point cloud to Rerun...")
    
    # If floor detection was performed, highlight floor points
    if floor_normal is not None and floor_offset is not None:
        print("Highlighting detected floor points...")
        
        # Calculate distance of each point to the floor plane
        distances_to_floor = np.abs(np.dot(points, floor_normal) - floor_offset)
        floor_mask = distances_to_floor < floor_threshold
        
        # Create modified colors
        if colors is not None:
            modified_colors = colors.copy()
        else:
            modified_colors = np.full((len(points), 3), [128, 128, 128], dtype=np.uint8)
        
        # Color floor points in bright green
        modified_colors[floor_mask] = [0, 255, 0]  # Bright green for floor
        
        # Log the point cloud with floor highlighting
        rr.log("world/pointcloud", rr.Points3D(points, colors=modified_colors), static=True)
        
        # Also log just the floor points separately for better visibility
        floor_points = points[floor_mask]
        if len(floor_points) > 0:
            rr.log("world/floor_points", rr.Points3D(
                floor_points, 
                colors=[0, 255, 0],  # Bright green
                radii=[0.01]
            ), static=True)
            
        print(f"Highlighted {np.sum(floor_mask)} floor points out of {len(points)} total points")
        
        # Log floor plane visualization
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
    else:
        # Standard point cloud logging
        if colors is not None:
            rr.log("world/pointcloud", rr.Points3D(points, colors=colors), static=True)
        else:
            rr.log("world/pointcloud", rr.Points3D(points, colors=[128, 128, 128]), static=True)  # Gray default


def log_camera_trajectory(timestamps: np.ndarray, positions: np.ndarray, quaternions: np.ndarray,
                         closest_points: np.ndarray, distances: np.ndarray, 
                         use_view_cone: bool = False, cone_angle_deg: float = 90.0,
                         floor_distances: Optional[np.ndarray] = None,
                         projected_points: Optional[np.ndarray] = None) -> None:
    """Log camera trajectory and closest point analysis to Rerun."""
    print("Logging camera trajectory and analysis...")
    
    # Log the complete camera trajectory as a static path
    rr.log("world/camera_trajectory", rr.LineStrips3D(
        strips=[positions],
        colors=[255, 100, 100],  # Light red for trajectory
        radii=[0.01]
    ), static=True)
    
    # Log each camera pose with timeline
    for i, (timestamp, position, quaternion, closest_point, distance) in enumerate(
        zip(timestamps, positions, quaternions, closest_points, distances)
    ):
        # Set timeline using updated API format
        rr.set_time_seconds("timestamp", timestamp)
        rr.set_time_sequence("frame", i)
        
        # Log camera pose using Transform3D with proper quaternion format
        # Create a proper rotation quaternion - Rerun expects [x,y,z,w] format
        qx, qy, qz, qw = quaternion
        rr.log("world/camera", rr.Transform3D(
            translation=position,
            rotation=rr.datatypes.Quaternion(xyzw=[qx, qy, qz, qw])
        ))
        
        # Log camera position as a point
        rr.log("world/camera_path", rr.Points3D(
            positions=position.reshape(1, 3),
            colors=[255, 0, 0],  # Red for camera
            radii=[0.02]
        ))
        
        # Log closest point (3D)
        rr.log("world/closest_point_3d", rr.Points3D(
            positions=closest_point.reshape(1, 3),
            colors=[0, 255, 0],  # Green for closest point
            radii=[0.03]
        ))
        
        # Log line connecting camera to closest point (3D)
        rr.log("world/distance_line_3d", rr.LineStrips3D(
            strips=[np.array([position, closest_point])],
            colors=[255, 255, 0],  # Yellow line
            radii=[0.005]
        ))
        
        # Log 3D distance as scalar plot
        rr.log("plots/distance_3d", rr.Scalars(scalars=[distance]))
        
        # Log floor distance information if available
        if floor_distances is not None and projected_points is not None:
            floor_distance = floor_distances[i]
            projected_point = projected_points[i]
            
            # Log projected point on floor plane
            rr.log("world/closest_point_floor", rr.Points3D(
                positions=projected_point.reshape(1, 3),
                colors=[255, 150, 0],  # Orange for floor-projected point
                radii=[0.025]
            ))
            
            # Log line from camera to floor-projected closest point
            rr.log("world/distance_line_floor", rr.LineStrips3D(
                strips=[np.array([position, projected_point])],
                colors=[255, 150, 0],  # Orange line
                radii=[0.007]
            ))
            
            # Log floor distance as scalar plot
            rr.log("plots/distance_floor", rr.Scalars(scalars=[floor_distance]))
            
            # Log combined distance text
            rr.log("world/distance_text", rr.TextDocument(
                f"3D: {distance:.3f}m | Floor: {floor_distance:.3f}m"
            ))
        else:
            # Log only 3D distance text
            rr.log("world/distance_text", rr.TextDocument(
                f"Distance: {distance:.3f}m"
            ))
        
        # Log camera viewing cone if enabled
        if use_view_cone:
            # Create a simple cone visualization
            R = quaternion_to_rotation_matrix(quaternion)
            forward = R[:, 2]  # Camera forward direction
            
            # Create cone apex and base points
            cone_length = min(distance * 1.5, 2.0)  # Adaptive cone length
            apex = position
            
            # Create cone base circle
            angle_rad = np.radians(cone_angle_deg)
            base_center = position + forward * cone_length
            
            # Simple cone representation with lines
            up = R[:, 1]
            right = R[:, 0]
            base_radius = cone_length * np.tan(angle_rad)
            
            # 8 points around the cone base
            cone_points = []
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                point = base_center + base_radius * (np.cos(theta) * right + np.sin(theta) * up)
                cone_points.append(point)
                # Line from apex to base edge
                rr.log("world/view_cone", rr.LineStrips3D(
                    strips=[np.array([apex, point])],
                    colors=[255, 255, 255, 100],  # Semi-transparent white
                    radii=[0.002]
                ))
            
            # Draw base circle
            cone_points = np.array(cone_points)
            for i in range(len(cone_points)):
                next_i = (i + 1) % len(cone_points)
                rr.log("world/view_cone", rr.LineStrips3D(
                    strips=[np.array([cone_points[i], cone_points[next_i]])],
                    colors=[255, 255, 255, 100],  # Semi-transparent white
                    radii=[0.002]
                ))


def save_rerun_recording(output_path: pathlib.Path) -> None:
    """Save the current Rerun recording to an .rrd file."""
    print(f"Saving Rerun recording to: {output_path}")
    rr.save(str(output_path))


def create_summary_plot(timestamps: np.ndarray, distances: np.ndarray, 
                       floor_distances: Optional[np.ndarray] = None) -> None:
    """Create summary statistics and plots."""
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total trajectory duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
    print(f"Number of poses: {len(timestamps)}")
    print(f"Average 3D distance to closest point: {distances.mean():.3f}m")
    print(f"Minimum 3D distance: {distances.min():.3f}m")
    print(f"Maximum 3D distance: {distances.max():.3f}m")
    print(f"Standard deviation 3D: {distances.std():.3f}m")
    
    if floor_distances is not None and len(floor_distances) > 0:
        print(f"\n=== FLOOR DISTANCE STATISTICS ===")
        print(f"Average floor distance: {floor_distances.mean():.3f}m")
        print(f"Minimum floor distance: {floor_distances.min():.3f}m")
        print(f"Maximum floor distance: {floor_distances.max():.3f}m")
        print(f"Standard deviation floor: {floor_distances.std():.3f}m")
        print(f"Floor distance is useful for collision detection as it represents")
        print(f"horizontal clearance regardless of object height.")
    
    # Log summary statistics to Rerun using updated API
    rr.log("stats/trajectory_duration", rr.Scalars(scalars=[timestamps[-1] - timestamps[0]]))
    rr.log("stats/average_distance_3d", rr.Scalars(scalars=[distances.mean()]))
    rr.log("stats/min_distance_3d", rr.Scalars(scalars=[distances.min()]))
    rr.log("stats/max_distance_3d", rr.Scalars(scalars=[distances.max()]))
    
    if floor_distances is not None and len(floor_distances) > 0:
        rr.log("stats/average_distance_floor", rr.Scalars(scalars=[floor_distances.mean()]))
        rr.log("stats/min_distance_floor", rr.Scalars(scalars=[floor_distances.min()]))
        rr.log("stats/max_distance_floor", rr.Scalars(scalars=[floor_distances.max()]))


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Process Mast3r-Slam outputs for PathPilot visualization")
    parser.add_argument("--ply", "-p", type=str, required=True,
                        help="Path to PLY point cloud file")
    parser.add_argument("--trajectory", "-t", type=str, required=True,
                        help="Path to trajectory TXT file")
    parser.add_argument("--output", "-o", type=str, default="pathpilot_output.rrd",
                        help="Output RRD file path")
    parser.add_argument("--name", "-n", type=str, default="PathPilot",
                        help="Recording name for Rerun")
    parser.add_argument("--use-view-cone", action="store_true",
                        help="Only consider points within camera's view cone")
    parser.add_argument("--cone-angle", type=float, default=90.0,
                        help="Half-angle of view cone in degrees (default: 90°)")
    parser.add_argument("--max-view-distance", type=float, default=10.0,
                        help="Maximum distance for view cone filtering (default: 10m)")
    parser.add_argument("--align-floor", action="store_true",
                        help="Detect and align coordinate system with floor plane")
    parser.add_argument("--floor-threshold", type=float, default=0.05,
                        help="Distance threshold for floor plane detection (default: 0.05m)")
    parser.add_argument("--use-floor-distance", action="store_true",
                        help="Calculate horizontal distances on floor plane (useful for collision detection)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    ply_path = pathlib.Path(args.ply)
    traj_path = pathlib.Path(args.trajectory)
    output_path = pathlib.Path(args.output)
    
    # Validate input files
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    print("=== PathPilot: Mast3r-Slam Output Processing Pipeline ===\n")
    
    # Step 1: Load data
    print("Step 1: Loading input data...")
    points, colors = load_point_cloud(ply_path)
    timestamps, positions, quaternions = load_camera_trajectory(traj_path)
    
    # Step 2: Floor detection and coordinate transformation
    floor_normal = None
    floor_offset = None
    
    if args.align_floor:
        print("\nStep 2: Detecting and aligning with floor plane...")
        # Use first few camera poses for floor detection
        floor_normal, floor_offset = detect_floor_plane(
            points, positions, quaternions, 
            ransac_threshold=args.floor_threshold
        )
        
        # Store original floor info for visualization
        original_floor_normal = floor_normal.copy()
        original_floor_offset = floor_offset
        
        # Transform coordinate system
        points, positions, quaternions = transform_to_floor_coordinates(
            points, positions, quaternions, floor_normal, floor_offset
        )
        
        # After transformation, floor becomes XY plane at Z=0
        floor_normal = np.array([0, 0, 1])  # Z-axis after transformation
        floor_offset = 0.0
        
        print("Floor-aligned coordinate system:")
        print(f"  - X, Y: Horizontal plane (floor)")
        print(f"  - Z: Vertical axis (height above floor)")
    else:
        print("\nStep 2: Using original coordinate system...")
    
    # Step 3: Find closest points
    print(f"\nStep {3 if args.align_floor else 2}: Computing closest points...")
    
    # Determine if we should use floor distance calculation
    use_floor_dist = args.use_floor_distance and args.align_floor
    if args.use_floor_distance and not args.align_floor:
        print("Warning: --use-floor-distance requires --align-floor. Floor distance calculation disabled.")
        use_floor_dist = False
    
    closest_points, distances, indices, floor_distances, projected_points = find_closest_points(
        points, positions, quaternions if args.use_view_cone else None,
        use_view_cone=args.use_view_cone, 
        cone_angle_deg=args.cone_angle,
        max_view_distance=args.max_view_distance,
        floor_normal=floor_normal if use_floor_dist else None,
        floor_offset=floor_offset if use_floor_dist else None,
        use_floor_distance=use_floor_dist
    )
    
    # Initialize Rerun
    step_num = 4 if args.align_floor else 3
    print(f"\nStep {step_num}: Setting up Rerun visualization...")
    setup_rerun(args.name)
    
    # Log data to Rerun
    step_num += 1
    print(f"\nStep {step_num}: Creating visualization...")
    log_point_cloud(points, colors, floor_normal, floor_offset, args.floor_threshold)
    log_camera_trajectory(timestamps, positions, quaternions, closest_points, distances,
                         use_view_cone=args.use_view_cone, cone_angle_deg=args.cone_angle,
                         floor_distances=floor_distances if use_floor_dist else None,
                         projected_points=projected_points if use_floor_dist else None)
    
    # Create summary
    step_num += 1
    print(f"\nStep {step_num}: Generating summary...")
    create_summary_plot(timestamps, distances, floor_distances if use_floor_dist else None)
    
    # Save recording
    step_num += 1
    print(f"\nStep {step_num}: Saving recording...")
    save_rerun_recording(output_path)
    
    print(f"\n=== Pipeline Complete! ===")
    print(f"Rerun recording saved to: {output_path}")
    print(f"Open with: rerun {output_path}")


if __name__ == "__main__":
    main() 