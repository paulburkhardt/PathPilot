#!/usr/bin/env python3
"""
SLAM Pipeline Output Visualizer

A standalone script to visualize SLAM pipeline outputs in rerun-sdk.
Loads and displays point clouds, camera trajectories, floor detection,
and closest point analysis with configurable visualization options.

Usage:
    python visualize_slam_output.py /path/to/slam_analysis_output_dir [options]
"""

import argparse
import json
import sys
import numpy as np
import rerun as rr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


class SLAMOutputVisualizer:
    """
    Visualizer for SLAM pipeline output data using rerun-sdk.
    """
    
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Initialize the visualizer with output directory and configuration.
        
        Args:
            output_dir: Path to SLAM analysis output directory
            config: Configuration dictionary with visualization options
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.data = {}
        
        # Validate output directory
        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
    
    def load_data(self) -> None:
        """Load all available SLAM output data files."""
        print("Loading SLAM output data...")
        
        # Load metadata first to get file paths
        self._load_metadata()
        
        # Load each data type if enabled and available
        if self.config['show_point_cloud']:
            self._load_point_cloud()
        
        if self.config['show_trajectory']:
            self._load_trajectory()
        
        if self.config['show_floor']:
            self._load_floor_data()
        
        if self.config['show_closest_points']:
            self._load_closest_points()
    
    def _load_metadata(self) -> None:
        """Load metadata.json if available."""
        metadata_path = self.output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.data['metadata'] = json.load(f)
            print(f"Loaded metadata from {metadata_path}")
            
            # Update visualization config based on pipeline configuration
            self._update_config_from_metadata()
        else:
            print("No metadata.json found, will search for standard file names")
            self.data['metadata'] = {}
    
    def _update_config_from_metadata(self) -> None:
        """Update visualization configuration based on pipeline metadata."""
        if 'pipeline_configuration' not in self.data['metadata']:
            return
        
        pipeline_config = self.data['metadata']['pipeline_configuration']
        
        # Look for IncrementalClosestPointFinderComponent configuration
        if 'pipeline' in pipeline_config and 'components' in pipeline_config['pipeline']:
            for component in pipeline_config['pipeline']['components']:
                if component.get('type') == 'IncrementalClosestPointFinderComponent':
                    component_config = component.get('config', {})
                    
                    # Override view cone settings from pipeline config
                    use_view_cone = component_config.get('use_view_cone', False)
                    cone_angle = component_config.get('cone_angle_deg', 90.0)
                    
                    # Update visualization config
                    # If view cones were enabled in pipeline but user didn't explicitly disable them
                    if use_view_cone and not hasattr(self, '_user_disabled_view_cones'):
                        self.config['show_view_cones'] = True
                    elif not use_view_cone:
                        self.config['show_view_cones'] = False
                    
                    self.config['view_cone_angle'] = cone_angle
                    
                    print(f"Updated view cone settings from pipeline config: enabled={self.config['show_view_cones']}, angle={cone_angle}°")
                    break
    
    def _load_point_cloud(self) -> None:
        """Load point cloud from PLY file."""
        # Try to get path from metadata, otherwise use standard name
        if 'file_paths' in self.data['metadata']:
            pc_file = Path(self.data['metadata']['file_paths'].get('point_cloud_path', ''))
            if not pc_file.is_absolute():
                pc_file = self.output_dir / pc_file.name
        else:
            # Try multiple possible filenames
            possible_names = [
                "slam_analysis_pointcloud.ply",
                "incremental_analysis_detailed_pointcloud.ply"
            ]
            pc_file = None
            for name in possible_names:
                candidate = self.output_dir / name
                if candidate.exists():
                    pc_file = candidate
                    break
            if pc_file is None:
                pc_file = self.output_dir / possible_names[0]  # Default fallback
        
        if pc_file.exists():
            try:
                # Try to use open3d if available, otherwise basic PLY parsing
                try:
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(str(pc_file))
                    self.data['point_cloud'] = {
                        'points': np.asarray(pcd.points),
                        'colors': np.asarray(pcd.colors) if pcd.has_colors() else None
                    }
                except ImportError:
                    print("Open3D not available, using basic PLY parsing")
                    self.data['point_cloud'] = self._parse_ply_file(pc_file)
                
                print(f"Loaded point cloud with {len(self.data['point_cloud']['points'])} points from {pc_file}")
            except Exception as e:
                print(f"Failed to load point cloud from {pc_file}: {e}")
        else:
            print(f"Point cloud file not found: {pc_file}")
    
    def _parse_ply_file(self, ply_file: Path) -> Dict[str, np.ndarray]:
        """Basic PLY file parser for when Open3D is not available."""
        points = []
        colors = []
        has_colors = False
        
        with open(ply_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        vertex_count = 0
        data_start = 0
        properties = []
        
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('property'):
                properties.append(line.strip())
                if 'red' in line or 'green' in line or 'blue' in line:
                    has_colors = True
            elif line.startswith('end_header'):
                data_start = i + 1
                break
        
        # Parse vertex data
        for line in lines[data_start:data_start + vertex_count]:
            values = line.strip().split()
            if len(values) >= 3:
                # Extract x, y, z
                points.append([float(values[0]), float(values[1]), float(values[2])])
                
                # Extract colors if available (assuming RGB are after xyz)
                if has_colors and len(values) >= 6:
                    colors.append([int(values[3]), int(values[4]), int(values[5])])
        
        return {
            'points': np.array(points),
            'colors': np.array(colors) if colors else None
        }
    
    def _load_trajectory(self) -> None:
        """Load camera trajectory from text file."""
        # Try to get path from metadata, otherwise use standard name
        if 'file_paths' in self.data['metadata']:
            traj_file = Path(self.data['metadata']['file_paths'].get('trajectory_path', ''))
            if not traj_file.is_absolute():
                traj_file = self.output_dir / traj_file.name
        else:
            # Try multiple possible filenames
            possible_names = [
                "slam_analysis_trajectory.txt",
                "incremental_analysis_detailed_trajectory.txt"
            ]
            traj_file = None
            for name in possible_names:
                candidate = self.output_dir / name
                if candidate.exists():
                    traj_file = candidate
                    break
            if traj_file is None:
                traj_file = self.output_dir / possible_names[0]  # Default fallback
        
        if traj_file.exists():
            try:
                trajectory_data = np.loadtxt(traj_file)
                self.data['trajectory'] = {
                    'timestamps': trajectory_data[:, 0],
                    'positions': trajectory_data[:, 1:4],
                    'quaternions': trajectory_data[:, 4:8]  # [qx, qy, qz, qw]
                }
                print(f"Loaded trajectory with {len(self.data['trajectory']['positions'])} poses from {traj_file}")
            except Exception as e:
                print(f"Failed to load trajectory from {traj_file}: {e}")
        else:
            print(f"Trajectory file not found: {traj_file}")
    
    def _load_floor_data(self) -> None:
        """Load floor detection data."""
        # Try JSON first, then CSV with multiple possible filenames
        floor_files = [
            self.output_dir / "slam_analysis_floor_data.json",
            self.output_dir / "slam_analysis_floor_data.csv",
            self.output_dir / "incremental_analysis_detailed_floor_data.json",
            self.output_dir / "incremental_analysis_detailed_floor_data.csv"
        ]
        
        for floor_file in floor_files:
            if floor_file.exists():
                try:
                    if floor_file.suffix == '.json':
                        with open(floor_file, 'r') as f:
                            floor_data = json.load(f)
                        self.data['floor'] = {
                            'normal': np.array(floor_data['floor_normal']),
                            'offset': floor_data['floor_offset'],
                            'detection_step': floor_data.get('detection_step'),
                            'timestamp': floor_data.get('timestamp')
                        }
                    else:  # CSV
                        # Basic CSV parsing
                        with open(floor_file, 'r') as f:
                            lines = f.readlines()
                        if len(lines) > 1:  # Header + data
                            data_line = lines[1].strip().split(',')
                            self.data['floor'] = {
                                'normal': np.array([float(data_line[0]), float(data_line[1]), float(data_line[2])]),
                                'offset': float(data_line[3]),
                                'detection_step': int(data_line[4]) if len(data_line) > 4 else None,
                                'timestamp': float(data_line[5]) if len(data_line) > 5 else None
                            }
                    print(f"Loaded floor data from {floor_file}")
                    break
                except Exception as e:
                    print(f"Failed to load floor data from {floor_file}: {e}")
        else:
            print("No floor data file found")
    
    def _load_closest_points(self) -> None:
        """Load closest points analysis data."""
        # Try JSON first, then CSV with multiple possible filenames
        closest_files = [
            self.output_dir / "slam_analysis_closest_points.json",
            self.output_dir / "slam_analysis_closest_points.csv",
            self.output_dir / "incremental_analysis_detailed_closest_points.json",
            self.output_dir / "incremental_analysis_detailed_closest_points.csv"
        ]
        
        for closest_file in closest_files:
            if closest_file.exists():
                try:
                    if closest_file.suffix == '.json':
                        with open(closest_file, 'r') as f:
                            closest_data = json.load(f)
                        self.data['closest_points'] = {
                            'points_3d': np.array(closest_data['n_closest_points_3d']),
                            'indices': np.array(closest_data['n_closest_points_indices']),
                            'distances': np.array(closest_data['n_closest_points_distances']),
                            'summary': closest_data.get('analysis_summary', {})
                        }
                    else:  # CSV
                        # Basic CSV parsing for closest points
                        with open(closest_file, 'r') as f:
                            lines = f.readlines()
                        
                        if len(lines) > 1:
                            # Group by step and filter for point_idx = 0 (first closest point)
                            step_data = {}
                            for line in lines[1:]:  # Skip header
                                parts = line.strip().split(',')
                                if len(parts) >= 7:
                                    step = int(parts[0])
                                    point_idx = int(parts[1])
                                    
                                    # Only take the first closest point (point_idx = 0)
                                    if point_idx == 0:
                                        step_data[step] = {
                                            'point': [float(parts[2]), float(parts[3]), float(parts[4])],
                                            'distance': float(parts[6])
                                        }
                            
                            # Convert to arrays - each step has one closest point
                            points_3d = []
                            distances = []
                            steps = []
                            for step in sorted(step_data.keys()):
                                points_3d.append([step_data[step]['point']])  # Wrap in list for consistency
                                distances.append([step_data[step]['distance']])  # Wrap in list for consistency
                                steps.append(step)  # Store the step index
                            
                            self.data['closest_points'] = {
                                'points_3d': np.array(points_3d, dtype=object),
                                'distances': np.array(distances, dtype=object),
                                'steps': np.array(steps),  # Store step indices for mapping
                                'summary': {}
                            }
                    print(f"Loaded closest points data from {closest_file}")
                    break
                except Exception as e:
                    print(f"Failed to load closest points data from {closest_file}: {e}")
        else:
            print("No closest points data file found")
    
    def visualize(self) -> None:
        """Visualize all loaded data in rerun."""
        print("Starting visualization in rerun...")
        
        # Initialize rerun with app ID
        rr.init("slam_output_viewer", spawn=False)
        
        # Connect to existing rerun web session
        try:
            # Try to connect to a running rerun viewer
            rr.connect("127.0.0.1:9876")  # Default rerun port
            print("Connected to rerun web session")
        except Exception as e:
            # If connection fails, try to spawn a new viewer
            try:
                rr.spawn()
                print("Spawned new rerun viewer")
            except Exception as e2:
                print(f"Failed to connect to rerun: {e}")
                print(f"Failed to spawn rerun viewer: {e2}")
                print("Please start rerun viewer manually with: rerun")
                return
        
        # Visualize static data first
        if self.config['show_point_cloud'] and 'point_cloud' in self.data:
            self._visualize_point_cloud()
        
        if self.config['show_floor'] and 'floor' in self.data:
            self._visualize_floor_plane()
        
        # Visualize temporal data
        if self.config['show_trajectory'] and 'trajectory' in self.data:
            self._visualize_trajectory()
        
        print("Visualization complete!")
    
    def _visualize_point_cloud(self) -> None:
        """Visualize the point cloud."""
        points = self.data['point_cloud']['points']
        colors = self.data['point_cloud']['colors']
        
        print(f"Visualizing point cloud with {len(points)} points...")
        
        # Log basic point cloud
        if colors is not None and len(colors) > 0:
            # Convert colors from [0,1] to [0,255] if needed
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            rr.log("world/pointcloud", rr.Points3D(points, colors=colors), static=True)
        else:
            rr.log("world/pointcloud", rr.Points3D(points, colors=[128, 128, 128]), static=True)
        
        # Highlight floor points if floor data is available
        if self.config['highlight_floor_points'] and 'floor' in self.data:
            self._highlight_floor_points(points, colors)
    
    def _highlight_floor_points(self, points: np.ndarray, colors: Optional[np.ndarray]) -> None:
        """Highlight floor points in the point cloud."""
        floor_normal = self.data['floor']['normal']
        floor_offset = self.data['floor']['offset']
        floor_threshold = self.config['floor_threshold']
        
        # Calculate distance of each point to the floor plane
        distances_to_floor = np.abs(np.dot(points, floor_normal) - floor_offset)
        floor_mask = distances_to_floor < floor_threshold
        
        # Create modified colors highlighting floor points
        if colors is not None:
            modified_colors = colors.copy()
        else:
            modified_colors = np.full((len(points), 3), [128, 128, 128], dtype=np.uint8)
        
        # Color floor points in bright green
        modified_colors[floor_mask] = [0, 255, 0]  # Bright green for floor
        
        # Log the point cloud with floor highlighting
        rr.log("world/pointcloud_with_floor", rr.Points3D(points, colors=modified_colors), static=True)
        
        # Also log just the floor points separately
        floor_points = points[floor_mask]
        if len(floor_points) > 0:
            rr.log("world/floor_points", rr.Points3D(
                floor_points, 
                colors=[0, 255, 0],  # Bright green
                radii=[0.01]
            ), static=True)
            
        print(f"Highlighted {np.sum(floor_mask)} floor points out of {len(points)} total points")
    
    def _visualize_floor_plane(self) -> None:
        """Visualize the floor plane as a grid."""
        floor_normal = self.data['floor']['normal']
        floor_offset = self.data['floor']['offset']
        grid_size = self.config['grid_size']
        
        print("Visualizing floor plane...")
        
        # Calculate floor center (use point cloud center if available)
        if 'point_cloud' in self.data:
            floor_center = np.mean(self.data['point_cloud']['points'], axis=0)
        else:
            floor_center = np.array([0.0, 0.0, 0.0])
        
        # Project center onto floor plane
        floor_center_on_plane = floor_center - np.dot(floor_center - floor_offset * floor_normal, floor_normal) * floor_normal
        
        # Find two orthogonal vectors in the floor plane
        if abs(floor_normal[0]) < 0.9:
            u = np.cross(floor_normal, [1, 0, 0])
        else:
            u = np.cross(floor_normal, [0, 1, 0])
        
        u = u / np.linalg.norm(u)
        v = np.cross(floor_normal, u)
        v = v / np.linalg.norm(v)
        
        # Create a grid to visualize the floor plane
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
            
        print(f"Added floor grid visualization with {len(grid_lines)} lines")
    
    def _visualize_trajectory(self) -> None:
        """Visualize camera trajectory with temporal data."""
        positions = self.data['trajectory']['positions']
        quaternions = self.data['trajectory']['quaternions']
        timestamps = self.data['trajectory']['timestamps']
        
        print(f"Visualizing trajectory with {len(positions)} poses...")
        
        # Log static trajectory path
        if self.config['show_trajectory_path']:
            rr.log("world/camera_trajectory", rr.LineStrips3D(
                strips=[positions],
                colors=[255, 100, 100],  # Light red for trajectory
                radii=[0.01]
            ), static=True)
        
        # Prepare closest points data if available
        closest_points_data = None
        if 'closest_points' in self.data and self.config['show_closest_points']:
            closest_points_data = self._prepare_closest_points_data(len(positions))
        
        # Log temporal poses
        self._log_temporal_poses(positions, quaternions, timestamps, closest_points_data)
        
        # Log summary statistics
        self._log_trajectory_statistics(positions, timestamps, closest_points_data)
    
    def _prepare_closest_points_data(self, num_poses: int) -> Dict[str, np.ndarray]:
        """Prepare closest points data for visualization."""
        closest_data = self.data['closest_points']
        
        # Handle different data formats
        if 'points_3d' in closest_data:
            points_3d = closest_data['points_3d']
            distances = closest_data['distances']
            steps = closest_data.get('steps', None)
            
            # If we have step information, map data to trajectory poses
            if steps is not None:
                print(f"Mapping {len(steps)} closest point entries to {num_poses} trajectory poses using step indices")
                
                # Create arrays for all poses, filled with None initially
                mapped_points = [None] * num_poses
                mapped_distances = [None] * num_poses
                
                # Map data based on step indices
                for i, step in enumerate(steps):
                    if step < num_poses and len(points_3d[i]) > 0:
                        mapped_points[step] = points_3d[i][0]  # First closest point
                        mapped_distances[step] = distances[i][0]  # First distance
                
                # Fill missing data with None or default values
                final_points = []
                final_distances = []
                
                for i in range(num_poses):
                    if mapped_points[i] is not None:
                        final_points.append(mapped_points[i])
                        final_distances.append(mapped_distances[i])
                    else:
                        # Use trajectory position as fallback for missing data
                        final_points.append(self.data['trajectory']['positions'][i])
                        final_distances.append(0.0)
                
                return {
                    'points': np.array(final_points),
                    'distances': np.array(final_distances),
                    'valid_mask': np.array([mapped_points[i] is not None for i in range(num_poses)])
                }
            
            else:
                # Fallback to old behavior if no step information
                if len(points_3d) != num_poses:
                    print(f"Warning: Closest points data length ({len(points_3d)}) doesn't match trajectory length ({num_poses})")
                    # Take the minimum length
                    min_len = min(len(points_3d), num_poses)
                    points_3d = points_3d[:min_len]
                    distances = distances[:min_len]
                
                # Extract first closest point and distance for each pose
                first_closest_points = []
                first_distances = []
                
                for i in range(len(points_3d)):
                    if len(points_3d[i]) > 0:
                        first_closest_points.append(points_3d[i][0])
                        first_distances.append(distances[i][0])
                    else:
                        # Use trajectory position as fallback
                        first_closest_points.append(self.data['trajectory']['positions'][i])
                        first_distances.append(0.0)
                
                return {
                    'points': np.array(first_closest_points),
                    'distances': np.array(first_distances)
                }
        
        return None
    
    def _log_temporal_poses(self, positions: np.ndarray, quaternions: np.ndarray, 
                           timestamps: np.ndarray, closest_points_data: Optional[Dict]) -> None:
        """Log temporal camera poses and analysis data."""
        
        for i, (timestamp, position, quaternion) in enumerate(zip(timestamps, positions, quaternions)):
            # Set timeline
            rr.set_time("timestamp", timestamp=timestamp)
            rr.set_time("frame", sequence=i)
            
            # Log camera pose
            qx, qy, qz, qw = quaternion
            rr.log("world/camera", rr.Transform3D(
                translation=position,
                rotation=rr.datatypes.Quaternion(xyzw=[qx, qy, qz, qw])
            ))
            
            # Log camera position as a point
            rr.log("world/camera_path", rr.Points3D(
                positions=position.reshape(1, 3),
                colors=[255, 0, 0],  # Red for camera
                radii=[0.03]
            ))
            
            # Log closest point analysis if available
            if closest_points_data is not None and i < len(closest_points_data['points']):
                closest_point = closest_points_data['points'][i]
                distance = closest_points_data['distances'][i]
                
                # Check if this pose has valid closest point data
                has_valid_data = True
                if 'valid_mask' in closest_points_data:
                    has_valid_data = closest_points_data['valid_mask'][i]
                
                if has_valid_data and distance > 0:  # Only show if we have real data
                    # Log closest point
                    rr.log("world/closest_point", rr.Points3D(
                        positions=closest_point.reshape(1, 3),
                        colors=[0, 255, 0],  # Green for closest point
                        radii=[0.025]
                    ))
                    
                    # Log distance line
                    if self.config['show_distance_lines']:
                        rr.log("world/distance_line", rr.LineStrips3D(
                            strips=[np.array([position, closest_point])],
                            colors=[255, 255, 0],  # Yellow line
                            radii=[0.005]
                        ))
                    
                    # Log distance plot
                    rr.log("plots/distance_to_closest", rr.Scalars(scalars=[distance]))
                    
                    # Log distance text
                    rr.log("world/distance_text", rr.TextDocument(
                        f"Distance to closest: {distance:.3f}m"
                    ))
                else:
                    # Clear visualizations for poses without valid data
                    rr.log("world/closest_point", rr.Clear(recursive=False))
                    rr.log("world/distance_line", rr.Clear(recursive=False))
                    rr.log("world/distance_text", rr.TextDocument("No closest point data"))
            
            # Log view cone if enabled
            if self.config['show_view_cones']:
                distance_for_cone = closest_points_data['distances'][i] if closest_points_data and i < len(closest_points_data['distances']) else 1.0
                self._log_view_cone(position, quaternion, distance_for_cone)
    
    def _log_view_cone(self, camera_position: np.ndarray, camera_quaternion: np.ndarray,
                      distance: float) -> None:
        """Log camera view cone visualization."""
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = camera_quaternion
        
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm > 0:
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        forward = R[:, 2]  # Camera forward direction
        
        # Create cone parameters
        cone_length = min(distance * self.config['cone_length_factor'], self.config['max_cone_length'])
        apex = camera_position
        
        # Create cone base circle
        angle_rad = np.radians(self.config['view_cone_angle'])
        base_center = camera_position + forward * cone_length
        
        # Cone base vectors
        up = R[:, 1]
        right = R[:, 0]
        base_radius = cone_length * np.tan(angle_rad)
        
        # 8 points around the cone base
        cone_lines = []
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            point = base_center + base_radius * (np.cos(theta) * right + np.sin(theta) * up)
            # Line from apex to base edge
            cone_lines.append(np.array([apex, point]))
        
        # Draw base circle
        base_points = []
        for theta in np.linspace(0, 2*np.pi, 16, endpoint=False):
            point = base_center + base_radius * (np.cos(theta) * right + np.sin(theta) * up)
            base_points.append(point)
        
        base_points = np.array(base_points)
        for i in range(len(base_points)):
            next_i = (i + 1) % len(base_points)
            cone_lines.append(np.array([base_points[i], base_points[next_i]]))
        
        if cone_lines:
            rr.log("world/view_cone", rr.LineStrips3D(
                strips=cone_lines,
                colors=[255, 255, 255, 100],  # Semi-transparent white
                radii=[0.002]
            ))
    
    def _log_trajectory_statistics(self, positions: np.ndarray, timestamps: np.ndarray,
                                  closest_points_data: Optional[Dict]) -> None:
        """Log summary statistics."""
        print("\n=== TRAJECTORY SUMMARY STATISTICS ===")
        print(f"Total trajectory duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Number of poses: {len(timestamps)}")
        
        # Log basic statistics
        rr.log("stats/trajectory_duration", rr.Scalars(scalars=[timestamps[-1] - timestamps[0]]))
        rr.log("stats/num_poses", rr.Scalars(scalars=[len(positions)]))
        
        if closest_points_data is not None:
            distances = closest_points_data['distances']
            print(f"Average distance to closest point: {distances.mean():.3f}m")
            print(f"Minimum distance: {distances.min():.3f}m")
            print(f"Maximum distance: {distances.max():.3f}m")
            print(f"Standard deviation: {distances.std():.3f}m")
            
            # Log distance statistics
            rr.log("stats/average_distance", rr.Scalars(scalars=[distances.mean()]))
            rr.log("stats/min_distance", rr.Scalars(scalars=[distances.min()]))
            rr.log("stats/max_distance", rr.Scalars(scalars=[distances.max()]))


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize SLAM pipeline output data in rerun-sdk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_slam_output.py ./enhanced_slam_outputs/slam_analysis_20240115_143045
    python visualize_slam_output.py ./outputs --no-view-cones --no-floor
    python visualize_slam_output.py ./outputs --cone-angle 45 --grid-size 5.0
    
Note: View cone settings will be automatically configured from pipeline metadata if available.
        """
    )
    
    # Required argument
    parser.add_argument(
        'output_dir',
        help='Path to SLAM analysis output directory'
    )
    
    # Visualization toggles
    parser.add_argument('--no-point-cloud', action='store_true',
                       help='Disable point cloud visualization')
    parser.add_argument('--no-trajectory', action='store_true',
                       help='Disable trajectory visualization')
    parser.add_argument('--no-floor', action='store_true',
                       help='Disable floor plane visualization')
    parser.add_argument('--no-closest-points', action='store_true',
                       help='Disable closest points analysis visualization')
    parser.add_argument('--no-view-cones', action='store_true',
                       help='Disable view cone visualization')
    parser.add_argument('--no-trajectory-path', action='store_true',
                       help='Disable static trajectory path')
    parser.add_argument('--no-distance-lines', action='store_true',
                       help='Disable distance lines to closest points')
    parser.add_argument('--no-highlight-floor', action='store_true',
                       help='Disable floor point highlighting')
    
    # Visualization parameters
    parser.add_argument('--floor-threshold', type=float, default=0.05,
                       help='Distance threshold for floor point identification (meters)')
    parser.add_argument('--grid-size', type=float, default=2.0,
                       help='Size of floor grid visualization (meters)')
    parser.add_argument('--cone-angle', type=float, default=45.0,
                       help='View cone angle in degrees')
    parser.add_argument('--cone-length-factor', type=float, default=1.5,
                       help='Factor for view cone length relative to distance')
    parser.add_argument('--max-cone-length', type=float, default=2.0,
                       help='Maximum length for view cones (meters)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Build configuration from arguments
    config = {
        'show_point_cloud': not args.no_point_cloud,
        'show_trajectory': not args.no_trajectory,
        'show_floor': not args.no_floor,
        'show_closest_points': not args.no_closest_points,
        'show_view_cones': not args.no_view_cones,
        'show_trajectory_path': not args.no_trajectory_path,
        'show_distance_lines': not args.no_distance_lines,
        'highlight_floor_points': not args.no_highlight_floor,
        'floor_threshold': args.floor_threshold,
        'grid_size': args.grid_size,
        'view_cone_angle': args.cone_angle,
        'cone_length_factor': args.cone_length_factor,
        'max_cone_length': args.max_cone_length
    }
    
    try:
        # Create and run visualizer
        visualizer = SLAMOutputVisualizer(args.output_dir, config)
        
        # Track if user explicitly disabled view cones
        if args.no_view_cones:
            visualizer._user_disabled_view_cones = True
        
        visualizer.load_data()
        visualizer.visualize()
        
        print("\nVisualization complete! Check the rerun web viewer.")
        print("Press Ctrl+C to exit when done viewing.")
        
        # Keep the script running to maintain the connection
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 