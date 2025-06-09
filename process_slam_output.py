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


# Removed quaternion_to_rotation_matrix function as we use quaternions directly with Rerun


def find_closest_points(point_cloud: np.ndarray, camera_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the closest point in the point cloud for each camera position.
    
    Args:
        point_cloud: Nx3 array of 3D points
        camera_positions: Mx3 array of camera positions
        
    Returns:
        closest_points: Mx3 array of closest points
        distances: M array of distances to closest points
        indices: M array of indices of closest points in point_cloud
    """
    print("Building KD-tree for nearest neighbor search...")
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(point_cloud)
    
    print("Finding closest points for each camera pose...")
    
    # Find nearest neighbors
    distances, indices = tree.query(camera_positions)
    closest_points = point_cloud[indices]
    
    print(f"Found closest points. Distance range: {distances.min():.3f} to {distances.max():.3f}")
    
    return closest_points, distances, indices


def setup_rerun(recording_name: str = "PathPilot") -> None:
    """Initialize Rerun with proper coordinate system."""
    rr.init(recording_name, recording_id=uuid.uuid4(), spawn=True)
    
    # Set up right-handed coordinate system (Y up, Z forward)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    print("Initialized Rerun viewer")


def log_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """Log the 3D point cloud to Rerun."""
    print("Logging point cloud to Rerun...")
    
    if colors is not None:
        rr.log("world/pointcloud", rr.Points3D(points, colors=colors), static=True)
    else:
        rr.log("world/pointcloud", rr.Points3D(points, colors=[128, 128, 128]), static=True)  # Gray default


def log_camera_trajectory(timestamps: np.ndarray, positions: np.ndarray, quaternions: np.ndarray,
                         closest_points: np.ndarray, distances: np.ndarray) -> None:
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
        
        # Log closest point
        rr.log("world/closest_point", rr.Points3D(
            positions=closest_point.reshape(1, 3),
            colors=[0, 255, 0],  # Green for closest point
            radii=[0.03]
        ))
        
        # Log line connecting camera to closest point
        rr.log("world/distance_line", rr.LineStrips3D(
            strips=[np.array([position, closest_point])],
            colors=[255, 255, 0],  # Yellow line
            radii=[0.005]
        ))
        
        # Log distance as scalar plot using updated API
        rr.log("plots/distance_to_closest_point", rr.Scalars(scalars=[distance]))
        
        # Log distance as text annotation
        midpoint = (position + closest_point) / 2
        rr.log("world/distance_text", rr.TextDocument(
            f"Distance: {distance:.3f}m"
        ))


def save_rerun_recording(output_path: pathlib.Path) -> None:
    """Save the current Rerun recording to an .rrd file."""
    print(f"Saving Rerun recording to: {output_path}")
    rr.save(str(output_path))


def create_summary_plot(timestamps: np.ndarray, distances: np.ndarray) -> None:
    """Create summary statistics and plots."""
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total trajectory duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
    print(f"Number of poses: {len(timestamps)}")
    print(f"Average distance to closest point: {distances.mean():.3f}m")
    print(f"Minimum distance: {distances.min():.3f}m")
    print(f"Maximum distance: {distances.max():.3f}m")
    print(f"Standard deviation: {distances.std():.3f}m")
    
    # Log summary statistics to Rerun using updated API
    rr.log("stats/trajectory_duration", rr.Scalars(scalars=[timestamps[-1] - timestamps[0]]))
    rr.log("stats/average_distance", rr.Scalars(scalars=[distances.mean()]))
    rr.log("stats/min_distance", rr.Scalars(scalars=[distances.min()]))
    rr.log("stats/max_distance", rr.Scalars(scalars=[distances.max()]))


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
    
    # Step 2: Find closest points
    print("\nStep 2: Computing closest points...")
    closest_points, distances, indices = find_closest_points(points, positions)
    
    # Step 3: Initialize Rerun
    print("\nStep 3: Setting up Rerun visualization...")
    setup_rerun(args.name)
    
    # Step 4: Log data to Rerun
    print("\nStep 4: Creating visualization...")
    log_point_cloud(points, colors)
    log_camera_trajectory(timestamps, positions, quaternions, closest_points, distances)
    
    # Step 5: Create summary
    print("\nStep 5: Generating summary...")
    create_summary_plot(timestamps, distances)
    
    # Step 6: Save recording
    print("\nStep 6: Saving recording...")
    save_rerun_recording(output_path)
    
    print(f"\n=== Pipeline Complete! ===")
    print(f"Rerun recording saved to: {output_path}")
    print(f"Open with: rerun {output_path}")


if __name__ == "__main__":
    main() 