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
import pandas as pd
import rerun as rr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import re
from plyfile import PlyData, PlyElement


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
        
        # Warning system state tracking
        self.warning_state = {
            'previous_distance': None,
            'previous_warning_level': 'normal',
            'frame_count': 0
        }
        
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
    
        if self.config['show_video']:
            self._load_video_data()
            
        if self.config['show_yolo_detections']:
            self._load_yolo_detections()
    
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
        """Load point cloud(s) from PLY file(s), including intermediate step point clouds."""
        # First, try to load intermediate point clouds
        intermediate_point_clouds = self._load_intermediate_point_clouds()
        
        if intermediate_point_clouds:
            # Use intermediate point clouds for temporal visualization
            self.data['point_cloud'] = intermediate_point_clouds
            print(f"Loaded {len(intermediate_point_clouds['steps'])} intermediate point clouds")
            return
        
        # Fallback to loading single final point cloud
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
                    #import open3d as o3d
                    #pcd = o3d.io.read_point_cloud(str(pc_file))
                    #self.data['point_cloud'] = {
                    #    'points': np.asarray(pcd.points),
                    #    'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
                    #    'is_temporal': False  # Mark as static point cloud
                    #}
                    plydata = PlyData.read(str(pc_file))
                
                    assert len(plydata.elements) == 1, "Expected exactly 1 element in the plydata. Don't know whats wrong."

                    point_cloud_data = {
                        "points": np.stack([
                            np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])]
                            ,1),
                        "colors": np.stack([
                            np.asarray(plydata.elements[0]["red"]),
                            np.asarray(plydata.elements[0]["green"]),
                            np.asarray(plydata.elements[0]["blue"])]
                            ,1),
                        "classIds": np.asarray(plydata.elements[0]["segmentation"]),
                        "is_temporal": False
                    }
                    self.data['point_cloud'] = point_cloud_data

                except ImportError:
                    print("Open3D not available, using basic PLY parsing")
                    point_cloud_data = self._parse_ply_file(pc_file)
                    point_cloud_data['is_temporal'] = False
                    self.data['point_cloud'] = point_cloud_data
                
                print(f"Loaded point cloud with {len(self.data['point_cloud']['points'])} points from {pc_file}")
            except Exception as e:
                print(f"Failed to load point cloud from {pc_file}: {e}")
        else:
            print(f"Point cloud file not found: {pc_file}")
    
    def _load_intermediate_point_clouds(self) -> Optional[Dict[str, Any]]:
        """Load intermediate point clouds from step directories."""
        intermediate_dir = self.output_dir / "intermediate"
        if not intermediate_dir.exists():
            return None
        
        print("Searching for intermediate point clouds...")
        
        # Find all step directories
        step_dirs = []
        for item in intermediate_dir.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                try:
                    step_num = int(item.name.replace("step_", ""))
                    pointcloud_file = item / "pointcloud.ply"
                    if pointcloud_file.exists():
                        step_dirs.append((step_num, pointcloud_file))
                except ValueError:
                    continue
        
        if not step_dirs:
            print("No intermediate point clouds found")
            return None
        
        # Sort by step number
        step_dirs.sort(key=lambda x: x[0])
        
        print(f"Found {len(step_dirs)} intermediate point clouds")
        
        # Load all point clouds
        point_clouds = {}
        steps = []
        
        for step_num, pointcloud_file in step_dirs:
            try:
                # Read the PLY file and extract points, colors, confidence, and segmentation if present
                plydata = PlyData.read(str(pointcloud_file))
                
                assert len(plydata.elements) == 1, "Expected exactly 1 element in the plydata. Don't know whats wrong."

                point_cloud_data = {
                    "points": np.stack([
                        np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])]
                        ,1),
                    "colors": np.stack([
                        np.asarray(plydata.elements[0]["red"]),
                        np.asarray(plydata.elements[0]["green"]),
                        np.asarray(plydata.elements[0]["blue"])]
                        ,1),
                    "classIds": np.asarray(plydata.elements[0]["segmentation"])
                }

                
                #try:
                #    import open3d as o3d
                #    pcd = o3d.io.read_point_cloud(str(pointcloud_file))
                #    point_cloud_data = {
                #        'points': np.asarray(pcd.points),
                #        'colors': np.asarray(pcd.colors) if pcd.has_colors() else None
#
                #    }
                #except ImportError:
                #    point_cloud_data = self._parse_ply_file(pointcloud_file)
                
                point_clouds[step_num] = point_cloud_data
                steps.append(step_num)
                
                print(f"  Step {step_num:06d}: {len(point_cloud_data['points'])} points")
                
            except Exception as e:
                print(f"  Failed to load step {step_num:06d}: {e}")
                continue
        
        if not point_clouds:
            return None
        
        return {
            'point_clouds': point_clouds,
            'steps': sorted(steps),
            'is_temporal': True
        }
    
    def _parse_ply_file(self, ply_file: Path) -> Dict[str, np.ndarray]:
        """Basic PLY file parser for when Open3D is not available."""
        points = []
        colors = []
        classIds = []
        has_classIds = False
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
                if "segmentation" in line:
                    has_classIds= True
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
        
                # Extract class Ids if available
                if has_classIds and len(values)>=7:
                    classIds.append(int(values[6]))

        return {
            'points': np.array(points),
            'colors': np.array(colors) if colors else None,
            "classIds": np.array(classIds) if classIds else None,
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
        """Load closest points or closest objects analysis data."""
        # Check user preference
        prefer_points = self.config.get('prefer_closest_points', False)
        
        if prefer_points:
            print("User preference: closest points over closest objects")
            self._load_traditional_closest_points()
            return
        
        # First try to load closest objects (with segment IDs) - this is the preferred format
        closest_objects_files = [
            self.output_dir / "slam_analysis_closest_objects.csv",
            self.output_dir / "incremental_analysis_detailed_closest_objects.csv"
        ]
        
        for closest_objects_file in closest_objects_files:
            if closest_objects_file.exists():
                try:
                    print(f"Loading closest objects data from {closest_objects_file}")
                    df = pd.read_csv(closest_objects_file)
                    
                    # Validate required columns for closest objects
                    required_columns = ['step', 'closest_3d_x', 'closest_3d_y', 'closest_3d_z', 'closest_2d_distance', 'segment_id']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Warning: Missing columns in closest objects CSV: {missing_columns}")
                        continue
                    
                    # Group by step and aggregate closest objects data
                    objects_by_step = {}
                    for _, row in df.iterrows():
                        step = int(row['step'])
                        if step not in objects_by_step:
                            objects_by_step[step] = {
                                'points': [],
                                'distances': [],
                                'segment_ids': []
                            }
                        
                        point = [row['closest_3d_x'], row['closest_3d_y'], row['closest_3d_z']]
                        distance = row['closest_2d_distance']
                        segment_id = int(row['segment_id'])
                        
                        objects_by_step[step]['points'].append(point)
                        objects_by_step[step]['distances'].append(distance)
                        objects_by_step[step]['segment_ids'].append(segment_id)
                    
                    # Convert to arrays for visualization compatibility
                    points_3d = []
                    distances = []
                    segment_ids = []
                    steps = []
                    
                    for step in sorted(objects_by_step.keys()):
                        step_data = objects_by_step[step]
                        points_3d.append(step_data['points'])
                        distances.append(step_data['distances'])
                        segment_ids.append(step_data['segment_ids'])
                        steps.append(step)
                    
                    self.data['closest_points'] = {
                        'points_3d': np.array(points_3d, dtype=object),
                        'distances': np.array(distances, dtype=object),
                        'segment_ids': np.array(segment_ids, dtype=object),  # New: segment IDs
                        'steps': np.array(steps),
                        'data_type': 'objects',  # New: mark as objects data
                        'summary': {
                            'total_steps': len(steps),
                            'unique_segments': len(set([sid for step_sids in segment_ids for sid in step_sids])),
                            'segments_per_step': {step: len(set(step_data['segment_ids'])) for step, step_data in objects_by_step.items()}
                        }
                    }
                    
                    print(f"Loaded closest objects data: {len(df)} entries across {len(steps)} steps")
                    print(f"Unique segment IDs: {sorted(set([sid for step_sids in segment_ids for sid in step_sids]))}")
                    return
                    
                except Exception as e:
                    print(f"Failed to load closest objects data from {closest_objects_file}: {e}")
                    continue
        
        # Fallback: Try to load traditional closest points data
        print("No closest objects data found, trying closest points data...")
        self._load_traditional_closest_points()
    
    def _load_traditional_closest_points(self) -> None:
        """Load traditional closest points data."""
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
                            'data_type': 'points',  # Mark as traditional points data
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
                                'data_type': 'points',  # Mark as traditional points data
                                'summary': {}
                            }
                    print(f"Loaded closest points data from {closest_file}")
                    break
                except Exception as e:
                    print(f"Failed to load closest points data from {closest_file}: {e}")
        else:
            print("No closest points data file found")
    
    def _load_video_data(self) -> None:
        """Load video data from pipeline configuration."""
        # Extract video path from pipeline configuration
        video_path = self._get_video_path_from_config()
        
        if not video_path:
            print("No video path found in pipeline configuration")
            return
        
        # Make path relative to the script's location if it's not absolute
        if not Path(video_path).is_absolute():
            # Try relative to current working directory first
            video_file = Path(video_path)
            if not video_file.exists():
                # Try relative to the output directory
                video_file = self.output_dir.parent / video_path
                if not video_file.exists():
                    # Try relative to the script's directory
                    script_dir = Path(__file__).parent
                    video_file = script_dir / video_path
        else:
            video_file = Path(video_path)
        
        # Check for rerun-compatible version first (converted by separate script)
        compatible_video_file = video_file.parent / f"{video_file.stem}_rerun_compatible.mp4"
        
        if compatible_video_file.exists():
            print(f"Found rerun-compatible video: {compatible_video_file}")
            video_file = compatible_video_file
        elif video_file.exists():
            print(f"Using original video: {video_file}")
            print("  Note: If video doesn't play, run: python convert_videos_for_rerun.py Data/Videos/")
        else:
            print(f"Warning: Video file not found: {video_file}")
            if not compatible_video_file.exists():
                print(f"  Also checked for: {compatible_video_file}")
            return
        
        try:
            # Store video information for later use in visualization
            self.data['video'] = {
                'path': str(video_file),
                'video_path_config': video_path  # Store original config path for reference
            }
            print(f"Video ready for visualization: {video_file}")
        except Exception as e:
            print(f"Failed to load video data: {e}")
            # Remove video from data so we don't try to use it later
            if 'video' in self.data:
                del self.data['video']

    def _get_video_path_from_config(self) -> Optional[str]:
        """Extract video path from pipeline configuration."""
        if 'pipeline_configuration' not in self.data['metadata']:
            return None
        
        pipeline_config = self.data['metadata']['pipeline_configuration']
        
        # Look for MAST3RSLAMVideoDataset component configuration
        if 'pipeline' in pipeline_config and 'components' in pipeline_config['pipeline']:
            for component in pipeline_config['pipeline']['components']:
                if component.get('type') == 'MAST3RSLAMVideoDataset':
                    component_config = component.get('config', {})
                    return component_config.get('video_path')
        
        return None
    
    def _load_yolo_detections(self) -> None:
        """Load YOLO detection data from CSV file."""
        # Look for YOLO detections CSV file
        yolo_files = [
            self.output_dir / "incremental_analysis_detailed_yolo_detections.csv",
            self.output_dir / "slam_analysis_yolo_detections.csv",
            self.output_dir / "yolo_detections.csv"
        ]
        
        for yolo_file in yolo_files:
            if yolo_file.exists():
                try:
                    print(f"Loading YOLO detections from {yolo_file}")
                    df = pd.read_csv(yolo_file)
                    
                    # Validate required columns
                    required_columns = ['step_nr', 'timestamp', 'class_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence', 'class_id']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Warning: Missing columns in YOLO CSV: {missing_columns}")
                        continue
                    
                    # Filter out rows with missing detection data (empty bounding boxes)
                    df = df.dropna(subset=['class_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
                    
                    # Group detections by step number for easier lookup
                    detections_by_step = {}
                    for _, row in df.iterrows():
                        step_nr = int(row['step_nr'])
                        if step_nr not in detections_by_step:
                            detections_by_step[step_nr] = []
                        
                        # Convert bounding box from [x1, y1, x2, y2] to [x, y, width, height] for Rerun
                        x1, y1, x2, y2 = row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
                        width = x2 - x1
                        height = y2 - y1
                        
                        detection = {
                            'class_name': row['class_name'],
                            'bbox': [x1, y1, width, height],  # Rerun format: [x, y, width, height]
                            'confidence': row['confidence'],
                            'class_id': int(row['class_id']),
                            'detection_id': row.get('detection_id', 0)
                        }
                        detections_by_step[step_nr].append(detection)
                    
                    self.data['yolo_detections'] = {
                        'detections_by_step': detections_by_step,
                        'total_detections': len(df)
                    }
                    
                    print(f"Loaded {len(df)} YOLO detections across {len(detections_by_step)} steps")
                    return
                    
                except Exception as e:
                    print(f"Failed to load YOLO detections from {yolo_file}: {e}")
                    continue
        
        print("No YOLO detections file found")
    
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
        
        # Visualize video first (static)
        if self.config['show_video'] and 'video' in self.data:
            self._visualize_video()
        
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
        """Visualize the point cloud(s)."""
        point_cloud_data = self.data['point_cloud']
        
        # Check if we have temporal point clouds
        if point_cloud_data.get('is_temporal', False):
            print("Temporal point clouds will be visualized during trajectory playback")
            # Temporal point clouds will be handled in _log_temporal_poses
            return
        
        # Handle static point cloud
        points = point_cloud_data['points']
        colors = point_cloud_data['colors']
        classIds = point_cloud_data["classIds"]
        
        print(f"Visualizing static point cloud with {len(points)} points...")
        
        # Log basic point cloud
        # Optionally color point cloud by classId
        color_by_class = self.config.get("color_pointcloud_by_classIds", False)
        has_colors = colors is not None and len(colors) > 0
        has_classIds = classIds is not None and len(classIds) > 0

        classIds = classIds.astype(int) if classIds is not None else None

        if color_by_class and has_classIds:
            rr.log(
            "world/pointcloud",
            rr.Points3D(points, class_ids=classIds),
            static=True
            )
        elif has_colors:
            # Convert colors from [0,1] to [0,255] if needed
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            if has_classIds:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=colors, class_ids=classIds),
                    static=True
                )
            else:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=colors),
                    static=True
                )
        else:
            default_color = [128, 128, 128]
            if has_classIds:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=default_color, class_ids=classIds),
                    static=True
                )
            else:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=default_color),
                    static=True
                )

        # Highlight floor points if floor data is available
        if self.config['highlight_floor_points'] and 'floor' in self.data:
            self._highlight_floor_points(points, colors)
            
        # Show segmentation masks separately if requested
        if self.config.get('show_segmentation_masks_separately', False) and has_classIds:
            self._log_segmentation_masks_separately_static(points, colors, classIds)
    
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
    
    def _log_segmentation_masks_separately_static(self, points: np.ndarray, colors: Optional[np.ndarray], classIds: np.ndarray) -> None:
        """Log segmentation masks as separate static entities for each class."""
        unique_class_ids = np.unique(classIds)
        
        # Define a color map for different classes
        class_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green  
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [255, 192, 203], # Pink
            [128, 128, 128], # Gray
            [139, 69, 19],   # Brown
            [0, 128, 0],     # Dark Green
        ]
        
        for i, class_id in enumerate(unique_class_ids):
            if class_id == 0:  # Skip background class
                continue
                
            # Get points belonging to this class
            class_mask = classIds == class_id
            class_points = points[class_mask]
            
            if len(class_points) == 0:
                continue
                
            # Use original colors if available, otherwise use class-specific color
            if colors is not None:
                class_point_colors = colors[class_mask]
            else:
                class_color = class_colors[i % len(class_colors)]
                class_point_colors = [class_color] * len(class_points)
            
            # Log this class as a separate static entity
            rr.log(f"world/segmentation/class_{class_id:03d}", rr.Points3D(
                class_points,
                colors=class_point_colors,
                radii=[0.008]  # Slightly smaller radius for individual classes
            ), static=True)
            
        print(f"Logged {len(unique_class_ids)} segmentation classes separately")
    
    def _visualize_floor_plane(self) -> None:
        """Visualize the floor plane as a grid."""
        floor_normal = self.data['floor']['normal']
        floor_offset = self.data['floor']['offset']
        grid_size = self.config['grid_size']
        
        print("Visualizing floor plane...")
        
        # Calculate floor center (use point cloud center if available)
        if 'point_cloud' in self.data:
            point_cloud_data = self.data['point_cloud']
            if point_cloud_data.get('is_temporal', False):
                # For temporal point clouds, use the last step for floor center calculation
                if point_cloud_data['steps']:
                    last_step = max(point_cloud_data['steps'])
                    if last_step in point_cloud_data['point_clouds']:
                        floor_center = np.mean(point_cloud_data['point_clouds'][last_step]['points'], axis=0)
                    else:
                        floor_center = np.array([0.0, 0.0, 0.0])
                else:
                    floor_center = np.array([0.0, 0.0, 0.0])
            else:
                # Static point cloud
                floor_center = np.mean(point_cloud_data['points'], axis=0)
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
        
    
    def _prepare_closest_points_data(self, num_poses: int) -> Dict[str, np.ndarray]:
        """Prepare closest points/objects data for visualization."""
        closest_data = self.data['closest_points']
        data_type = closest_data.get('data_type', 'points')
        
        print(f"Preparing visualization data for {data_type} (type: {data_type})")
        
        # Handle different data formats
        if 'points_3d' in closest_data:
            points_3d = closest_data['points_3d']
            distances = closest_data['distances']
            steps = closest_data.get('steps', None)
            segment_ids = closest_data.get('segment_ids', None)  # New: segment IDs
            
            # If we have step information, map data to trajectory poses
            if steps is not None:
                print(f"Mapping {len(steps)} {data_type} entries to {num_poses} trajectory poses using step indices")
                
                # Create arrays for all poses, filled with None initially
                mapped_points = [None] * num_poses
                mapped_distances = [None] * num_poses
                mapped_segment_ids = [None] * num_poses  # New: segment IDs mapping
                
                # Map data based on step indices
                for i, step in enumerate(steps):
                    if step < num_poses and len(points_3d[i]) > 0:
                        mapped_points[step] = points_3d[i][0]  # First closest point/object
                        mapped_distances[step] = distances[i][0]  # First distance
                        
                        # Map segment IDs if available
                        if segment_ids is not None and len(segment_ids[i]) > 0:
                            mapped_segment_ids[step] = segment_ids[i][0]  # First segment ID
                
                # Fill missing data with None or default values
                final_points = []
                final_distances = []
                final_segment_ids = []
                valid_mask = []
                
                for i in range(num_poses):
                    if mapped_points[i] is not None:
                        final_points.append(mapped_points[i])
                        final_distances.append(mapped_distances[i])
                        final_segment_ids.append(mapped_segment_ids[i])
                        valid_mask.append(True)
                    else:
                        # Use trajectory position as fallback for missing data
                        final_points.append(self.data['trajectory']['positions'][i])
                        final_distances.append(0.0)
                        final_segment_ids.append(-1)  # -1 indicates no segment data
                        valid_mask.append(False)
                
                result = {
                    'points': np.array(final_points),
                    'distances': np.array(final_distances),
                    'valid_mask': np.array(valid_mask),
                    'data_type': data_type
                }
                
                # Add segment IDs if available
                if data_type == 'objects':
                    result['segment_ids'] = np.array(final_segment_ids)
                    print(f"Mapped segment IDs available for visualization")
                
                return result
            
            else:
                # Fallback to old behavior if no step information
                if len(points_3d) != num_poses:
                    print(f"Warning: {data_type} data length ({len(points_3d)}) doesn't match trajectory length ({num_poses})")
                    # Take the minimum length
                    min_len = min(len(points_3d), num_poses)
                    points_3d = points_3d[:min_len]
                    distances = distances[:min_len]
                    if segment_ids is not None:
                        segment_ids = segment_ids[:min_len]
                
                # Extract first closest point/object and distance for each pose
                first_closest_points = []
                first_distances = []
                first_segment_ids = []
                
                for i in range(len(points_3d)):
                    if len(points_3d[i]) > 0:
                        first_closest_points.append(points_3d[i][0])
                        first_distances.append(distances[i][0])
                        if segment_ids is not None and len(segment_ids[i]) > 0:
                            first_segment_ids.append(segment_ids[i][0])
                        else:
                            first_segment_ids.append(-1)
                    else:
                        # Use trajectory position as fallback
                        first_closest_points.append(self.data['trajectory']['positions'][i])
                        first_distances.append(0.0)
                        first_segment_ids.append(-1)
                
                result = {
                    'points': np.array(first_closest_points),
                    'distances': np.array(first_distances),
                    'data_type': data_type
                }
                
                # Add segment IDs if this is objects data
                if data_type == 'objects':
                    result['segment_ids'] = np.array(first_segment_ids)
                
                return result
        
        return None
    
    def _log_temporal_poses(self, positions: np.ndarray, quaternions: np.ndarray, 
                           timestamps: np.ndarray, closest_points_data: Optional[Dict]) -> None:
        """Log temporal camera poses and analysis data."""
        
        # Prepare temporal point cloud data if available
        temporal_point_clouds = None
        step_to_trajectory_mapping = None
        if 'point_cloud' in self.data and self.data['point_cloud'].get('is_temporal', False):
            temporal_point_clouds = self.data['point_cloud']
            step_to_trajectory_mapping = self._map_steps_to_trajectory(
                temporal_point_clouds['steps'], len(positions)
            )
        
        # Prepare video frame data if available
        video_frame_mapping = None
        if 'video' in self.data:
            video_frame_mapping = self._map_trajectory_to_video_frames(len(positions))
        
        for i, (timestamp, position, quaternion) in enumerate(zip(timestamps, positions, quaternions)):
            # Set timeline
            rr.set_time("timestamp", timestamp=timestamp)
            rr.set_time("frame", sequence=i)
            
            # Log video frame if available
            if video_frame_mapping and i < len(video_frame_mapping):
                video_timestamp_ns = video_frame_mapping[i]
                if video_timestamp_ns is not None:
                    rr.log("video/frame", rr.VideoFrameReference(
                        timestamp=rr.datatypes.VideoTimestamp(timestamp_ns=video_timestamp_ns),
                        video_reference="video/asset"
                    ))
            
            # Log YOLO bounding boxes if available for this step
            if self.config['show_yolo_detections'] and 'yolo_detections' in self.data:
                self._log_yolo_bounding_boxes(i)
            
            # Log temporal point cloud if available for this step
            if temporal_point_clouds and step_to_trajectory_mapping:
                self._log_temporal_point_cloud(i, temporal_point_clouds, step_to_trajectory_mapping)
            
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
                data_type = closest_points_data.get('data_type', 'points')
                
                # Get segment ID if available
                segment_id = None
                if 'segment_ids' in closest_points_data and i < len(closest_points_data['segment_ids']):
                    segment_id = closest_points_data['segment_ids'][i]
                
                # Check if this pose has valid closest point data
                has_valid_data = True
                if 'valid_mask' in closest_points_data:
                    has_valid_data = closest_points_data['valid_mask'][i]
                
                if has_valid_data and distance > 0:  # Only show if we have real data
                    # Update warning system state
                    self.warning_state['frame_count'] += 1
                    
                    # Calculate warning level
                    warning_level = self._calculate_warning_level(distance)
                    
                    # Calculate direction to closest point
                    direction = ""
                    if self.config['enable_directional_warnings']:
                        direction = self._calculate_direction_to_point(position, quaternion, closest_point)
                    
                    # Generate warning message (now includes segment info)
                    warning_message = self._generate_warning_message(
                        distance, direction, warning_level, data_type, segment_id
                    )
                    
                    # Get warning-appropriate colors and styling
                    warning_colors = self._get_warning_colors(warning_level)
                    warning_radius = self._get_warning_radius(warning_level)
                    
                    # Log closest point with warning-appropriate styling
                    entity_name = "world/closest_object" if data_type == 'objects' else "world/closest_point"
                    rr.log(entity_name, rr.Points3D(
                        positions=closest_point.reshape(1, 3),
                        colors=warning_colors,
                        radii=[warning_radius]
                    ))
                    
                    # Log distance line with warning colors
                    if self.config['show_distance_lines']:
                        line_colors = warning_colors if warning_level != 'normal' else [255, 255, 0]
                        line_width = 0.008 if warning_level in ['strong', 'critical'] else 0.005
                        line_entity = "world/distance_line_object" if data_type == 'objects' else "world/distance_line"
                        rr.log(line_entity, rr.LineStrips3D(
                            strips=[np.array([position, closest_point])],
                            colors=line_colors,
                            radii=[line_width]
                        ))
                    
                    # Log distance plot
                    plot_name = "plots/distance_to_closest_object" if data_type == 'objects' else "plots/distance_to_closest"
                    rr.log(plot_name, rr.Scalars(scalars=[distance]))
                    
                    # Log warning level plot
                    warning_level_numeric = {'normal': 0, 'caution': 1, 'strong': 2, 'critical': 3}
                    rr.log("plots/warning_level", rr.Scalars(scalars=[warning_level_numeric[warning_level]]))
                    
                    # Log segment ID plot if available
                    if segment_id is not None and segment_id >= 0:
                        rr.log("plots/closest_object_segment_id", rr.Scalars(scalars=[segment_id]))
                    
                    # Log enhanced warning message
                    rr.log("world/distance_text", rr.TextDocument(warning_message))
                    
                    # Update warning state for next frame
                    self.warning_state['previous_distance'] = distance
                    self.warning_state['previous_warning_level'] = warning_level
                else:
                    # Clear visualizations for poses without valid data
                    rr.log("world/closest_point", rr.Clear(recursive=False))
                    rr.log("world/closest_object", rr.Clear(recursive=False))
                    rr.log("world/distance_line", rr.Clear(recursive=False))
                    rr.log("world/distance_line_object", rr.Clear(recursive=False))
                    text_message = f"No closest {data_type} data"
                    rr.log("world/distance_text", rr.TextDocument(text_message))
            
            # Log view cone if enabled
            if self.config['show_view_cones']:
                distance_for_cone = closest_points_data['distances'][i] if closest_points_data and i < len(closest_points_data['distances']) else 1.0
                self._log_view_cone(position, quaternion, distance_for_cone)
    
    def _map_steps_to_trajectory(self, point_cloud_steps: List[int], num_trajectory_poses: int) -> Dict[int, int]:
        """Map point cloud steps to trajectory pose indices."""
        if not point_cloud_steps:
            return {}
        
        # Create mapping from trajectory index to point cloud step
        # Assume linear mapping between trajectory poses and point cloud steps
        mapping = {}
        
        if len(point_cloud_steps) == num_trajectory_poses:
            # Perfect 1:1 mapping
            for i, step in enumerate(point_cloud_steps):
                mapping[i] = step
        else:
            # Map trajectory poses to the closest available point cloud step
            max_step = max(point_cloud_steps)
            min_step = min(point_cloud_steps)
            
            for i in range(num_trajectory_poses):
                # Linear interpolation to find the corresponding step
                if num_trajectory_poses > 1:
                    progress = i / (num_trajectory_poses - 1)
                else:
                    progress = 0.0
                
                target_step = int(min_step + progress * (max_step - min_step))
                
                # Find the closest available step
                closest_step = min(point_cloud_steps, key=lambda x: abs(x - target_step))
                mapping[i] = closest_step
        
        print(f"Mapped {num_trajectory_poses} trajectory poses to {len(set(mapping.values()))} unique point cloud steps")
        return mapping
    
    def _log_temporal_point_cloud(self, trajectory_index: int, temporal_point_clouds: Dict[str, Any], 
                                 step_mapping: Dict[int, int]) -> None:
        """Log the point cloud for the current trajectory step."""
        if trajectory_index not in step_mapping:
            return
        
        step_num = step_mapping[trajectory_index]
        if step_num not in temporal_point_clouds['point_clouds']:
            return
        
        point_cloud_data = temporal_point_clouds['point_clouds'][step_num]
        points = point_cloud_data['points']
        colors = point_cloud_data.get('colors')
        classIds = point_cloud_data.get('classIds')
        
        # Apply spatial-aware subsampling to manage memory while preserving structure
        # Check if this is the final step (highest step number gets 100% of points)
        max_step = max(temporal_point_clouds['steps'])
        is_final_step = (step_num == max_step)
        points, colors, classIds = self._spatially_subsample_points(points, colors,classIds, is_final_step)
        
        # Log point cloud for this step
        color_by_class = self.config.get("color_pointcloud_by_classIds", False)
        has_colors = colors is not None and len(colors) > 0
        has_classIds = classIds is not None and len(classIds) > 0

        classIds = classIds.astype(int) if classIds is not None else None

        if color_by_class and has_classIds:
            rr.log(
            "world/pointcloud_colored_by_class",
            rr.Points3D(points, class_ids=classIds)
            )
        elif has_colors:
            # Convert colors from [0,1] to [0,255] if needed
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            if has_classIds:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=colors, class_ids=classIds)
                )
            else:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=colors)
                )
        else:
            default_color = [128, 128, 128]
            if has_classIds:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=default_color, class_ids=classIds)
                )
            else:
                rr.log(
                    "world/pointcloud",
                    rr.Points3D(points, colors=default_color)
                )

        # Highlight floor points if floor data is available
        if self.config['highlight_floor_points'] and 'floor' in self.data:
            self._highlight_floor_points_temporal(points, colors)
            
        # Show segmentation masks separately if requested
        if self.config.get('show_segmentation_masks_separately', False) and has_classIds:
            self._log_segmentation_masks_separately(points, colors, classIds)
    
    def _spatially_subsample_points(self, points: np.ndarray, colors: Optional[np.ndarray],classIds:Optional[np.ndarray], is_final_step: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Subsample point cloud while preserving spatial consistency and structure.
        Uses percentage-based subsampling - final step shows 100% of points.
        """
        # Check if subsampling is disabled
        if not self.config.get('enable_subsampling', True):
            return points, colors, classIds
        
        # Final step shows all points (100%)
        if is_final_step:
            print(f"Final step: showing all {len(points)} points (100%)")
            return points, colors, classIds
        
        # Get percentage to keep
        subsample_percentage = self.config.get('subsample_percentage', 0.6)
        target_points = int(len(points) * subsample_percentage)
                
        # If we already have fewer points than target, keep all
        if len(points) <= target_points:
            return points, colors, classIds
        
        print(f"Subsampling {len(points)} points to {target_points} ({subsample_percentage*100:.1f}%) while preserving spatial structure...")
        
        # Use direct percentage-based sampling for precise control
        return self._percentage_based_subsampling(points, colors,classIds, subsample_percentage)
    
    def _percentage_based_subsampling(self, points: np.ndarray, colors: Optional[np.ndarray],classIds:Optional[np.ndarray], percentage: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Direct percentage-based subsampling using spatial grid for uniform distribution.
        Guarantees exact percentage while maintaining spatial consistency.
        """
        target_points = int(len(points) * percentage)
        
        # If requesting more points than we have, return all
        if target_points >= len(points):
            return points, colors, classIds
        
        # Method 1: Try voxel-based with fallback to random if not precise enough
        try:
            import open3d as o3d
            
            # Try voxel-based first to see if we can get close
            voxel_points, voxel_colors,voxel_classIds = self._voxel_based_subsampling(points, colors,classIds, target_points)
            
            # Check if voxel method got us close enough (within 5%)
            actual_percentage = len(voxel_points) / len(points)
            target_percentage = percentage
            
            if abs(actual_percentage - target_percentage) <= 0.05:  # Within 5%
                print(f"  Voxel method achieved {actual_percentage*100:.1f}% (target: {target_percentage*100:.1f}%)")
                return voxel_points, voxel_colors, voxel_classIds
            else:
                print(f"  Voxel method achieved {actual_percentage*100:.1f}% (target: {target_percentage*100:.1f}%), using random spatial sampling")
                
        except ImportError:
            print("  Open3D not available, using random spatial sampling")
        
        # Method 2: Random spatial sampling for exact percentage
        return self._random_spatial_subsampling(points, colors,classIds, target_points)
    
    def _random_spatial_subsampling(self, points: np.ndarray, colors: Optional[np.ndarray],classIds: Optional[np.ndarray], target_points: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Random spatial subsampling that maintains spatial distribution while achieving exact count.
        """
        # Use spatial stratification to maintain distribution
        # Divide space into grid cells and sample proportionally from each
        
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Create a reasonable grid (aim for ~10-20 points per cell on average)
        avg_points_per_cell = 15
        total_cells = len(points) // avg_points_per_cell
        cells_per_dim = max(2, int(total_cells ** (1/3)))
        
        # Calculate grid dimensions
        grid_size = (max_coords - min_coords) / cells_per_dim
        
        # Assign points to grid cells
        grid_indices = np.floor((points - min_coords) / grid_size).astype(np.int32)
        grid_indices = np.clip(grid_indices, 0, cells_per_dim - 1)  # Ensure within bounds
        
        # Group points by grid cell
        cell_points = {}
        for i, cell_idx in enumerate(grid_indices):
            key = tuple(cell_idx)
            if key not in cell_points:
                cell_points[key] = []
            cell_points[key].append(i)
        
        # Calculate how many points to sample from each cell
        sampling_ratio = target_points / len(points)
        selected_indices = []
        
        for cell_indices in cell_points.values():
            cell_target = max(1, int(len(cell_indices) * sampling_ratio))
            if len(cell_indices) <= cell_target:
                selected_indices.extend(cell_indices)
            else:
                # Randomly sample from this cell
                sampled = np.random.choice(cell_indices, cell_target, replace=False)
                selected_indices.extend(sampled)
        
        # If we have too many points, randomly remove some
        if len(selected_indices) > target_points:
            selected_indices = np.random.choice(selected_indices, target_points, replace=False)
        
        # If we have too few points, randomly add some more
        elif len(selected_indices) < target_points:
            remaining_indices = list(set(range(len(points))) - set(selected_indices))
            if remaining_indices:
                additional_needed = target_points - len(selected_indices)
                additional_needed = min(additional_needed, len(remaining_indices))
                additional = np.random.choice(remaining_indices, additional_needed, replace=False)
                selected_indices.extend(additional)
        
        # Convert to numpy array and extract points
        selected_indices = np.array(selected_indices)
        filtered_points = points[selected_indices]
        filtered_colors = colors[selected_indices] if colors is not None else None
        filtered_classIds = classIds[selected_indices] if classIds is not None else None
        
        actual_percentage = len(filtered_points) / len(points)
        print(f"  Random spatial sampling: {len(points)} -> {len(filtered_points)} points ({actual_percentage*100:.1f}%)")
        
        return filtered_points, filtered_colors, filtered_classIds
    
    def _voxel_based_subsampling(self, points: np.ndarray, colors: Optional[np.ndarray],classIds: Optional[np.ndarray], max_points: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Use Open3D's voxel downsampling for optimal spatial distribution."""
        import open3d as o3d
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            # Normalize colors to [0,1] range if needed
            if colors.max() > 1.0:
                colors_normalized = colors / 255.0
            else:
                colors_normalized = colors
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        if classIds is not None:
            pcd.normals = o3d.utility.Vector3dVector(
                np.tile(classIds.reshape(-1, 1), (1, 3))
            )
        
        # Calculate an intelligent initial voxel size based on point cloud characteristics
        # Get bounding box to understand the scale
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        bbox_diagonal = np.linalg.norm(max_coords - min_coords)
        
        # Estimate voxel size based on desired reduction ratio
        # For N points wanting to keep M points, rough voxel count = M
        # So voxel_size ≈ bbox_diagonal / (M^(1/3))
        target_ratio = max_points / len(points)
        estimated_voxels_per_dim = int((max_points) ** (1/3))
        initial_voxel_size = bbox_diagonal / max(10, estimated_voxels_per_dim)
        
        # Ensure reasonable bounds (between 0.01m and 1.0m)
        voxel_size = max(0.01, min(1.0, initial_voxel_size))
        
        print(f"  Initial voxel calculation: bbox={bbox_diagonal:.2f}m, target_ratio={target_ratio:.2f}, initial_voxel={voxel_size:.3f}")
        
        # Iteratively adjust voxel size to get close to target point count
        for attempt in range(8):  # Increased attempts for better convergence
            pcd_downsampled = pcd.voxel_down_sample(voxel_size)
            downsampled_count = len(pcd_downsampled.points)
            
            if downsampled_count == 0:
                # Voxel size too large, make it much smaller
                voxel_size *= 0.1
                continue
                
            # Check if we're close enough (within 20% of target)
            if abs(downsampled_count - max_points) <= max_points * 0.2:
                break
            elif downsampled_count > max_points:
                # Too many points, increase voxel size
                adjustment = min(2.0, (downsampled_count / max_points) ** 0.5)
                voxel_size *= adjustment
            else:
                # Too few points, decrease voxel size
                adjustment = max(0.5, (downsampled_count / max_points) ** 0.5)
                voxel_size *= adjustment
                
            # Prevent infinite loops with extreme values
            voxel_size = max(0.001, min(10.0, voxel_size))
        
        # Extract results
        filtered_points = np.asarray(pcd_downsampled.points)
        filtered_colors = None
        filtered_classIds = None
        if colors is not None and pcd_downsampled.has_colors():
            filtered_colors = np.asarray(pcd_downsampled.colors)
            # Convert back to [0,255] if original was in that range
            if colors.max() > 1.0:
                filtered_colors = (filtered_colors * 255).astype(np.uint8)

        if classIds is not None and pcd_downsampled.has_normals():
            filtered_classIds = np.asarray(pcd_downsampled.normals)
            filtered_classIds = filtered_classIds[:,0]
        
        print(f"  Voxel filtering: {len(points)} -> {len(filtered_points)} points (voxel_size: {voxel_size:.3f})")
        return filtered_points, filtered_colors, filtered_classIds
    
    def _log_segmentation_masks_separately(self, points: np.ndarray, colors: Optional[np.ndarray], classIds: np.ndarray) -> None:
        """Log segmentation masks as separate entities for each class."""
        unique_class_ids = np.unique(classIds)
        
        # Define a color map for different classes
        class_colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green  
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [255, 192, 203], # Pink
            [128, 128, 128], # Gray
            [139, 69, 19],   # Brown
            [0, 128, 0],     # Dark Green
        ]
        
        for i, class_id in enumerate(unique_class_ids):
            if class_id == 0:  # Skip background class
                continue
                
            # Get points belonging to this class
            class_mask = classIds == class_id
            class_points = points[class_mask]
            
            if len(class_points) == 0:
                continue
                
            # Use original colors if available, otherwise use class-specific color
            if colors is not None:
                class_point_colors = colors[class_mask]
            else:
                class_color = class_colors[i % len(class_colors)]
                class_point_colors = [class_color] * len(class_points)
            
            # Log this class as a separate entity
            rr.log(f"world/segmentation/class_{class_id:03d}", rr.Points3D(
                class_points,
                colors=class_point_colors,
                radii=[0.008]  # Slightly smaller radius for individual classes
            ))

    def _highlight_floor_points_temporal(self, points: np.ndarray, colors: Optional[np.ndarray]) -> None:
        """Highlight floor points in the temporal point cloud."""
        if 'floor' not in self.data:
            return
            
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
        rr.log("world/pointcloud_with_floor", rr.Points3D(points, colors=modified_colors))
        
        # Also log just the floor points separately
        floor_points = points[floor_mask]
        if len(floor_points) > 0:
            rr.log("world/floor_points", rr.Points3D(
                floor_points, 
                colors=[0, 255, 0],  # Bright green
                radii=[0.01]
            ))
    
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
    
    def _log_yolo_bounding_boxes(self, trajectory_index: int) -> None:
        """Log YOLO bounding boxes for the current trajectory step."""
        yolo_data = self.data['yolo_detections']
        detections_by_step = yolo_data['detections_by_step']
        
        # Map trajectory index to step number (assuming 1:1 mapping for now)
        # In the future, this could use a more sophisticated mapping like the point cloud mapping
        step_nr = trajectory_index
        
        if step_nr in detections_by_step:
            detections = detections_by_step[step_nr]
            
            if len(detections) > 0:
                # Prepare data for Rerun Boxes2D
                boxes = []
                labels = []
                colors = []
                class_ids = []
                
                # Color mapping for different classes
                class_color_map = {
                    'Chair': [255, 165, 0],      # Orange
                    'Window': [0, 191, 255],     # Deep sky blue
                    'Person': [255, 20, 147],    # Deep pink
                    'Car': [255, 0, 0],          # Red
                    'Table': [50, 205, 50],      # Lime green
                    'Door': [138, 43, 226],      # Blue violet
                }
                
                for detection in detections:
                    bbox = detection['bbox']  # [x, y, width, height]
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    class_id = detection['class_id']
                    
                    # Only show detections above confidence threshold
                    if confidence >= self.config.get('yolo_confidence_threshold', 0.3):
                        boxes.append(bbox)
                        
                        # Create label with class name and confidence
                        label = f"{class_name} ({confidence:.2f})"
                        labels.append(label)
                        
                        # Get color for this class
                        color = class_color_map.get(class_name, [128, 128, 128])  # Default gray
                        colors.append(color)
                        
                        class_ids.append(class_id)
                
                # Log bounding boxes if we have any valid detections
                if len(boxes) > 0:
                    boxes_array = np.array(boxes)
                    
                    # Convert to half-sizes format that Rerun expects for Boxes2D
                    # Rerun Boxes2D expects: centers and half_sizes
                    centers = []
                    half_sizes = []
                    
                    for box in boxes:
                        x, y, width, height = box
                        center_x = x + width / 2
                        center_y = y + height / 2
                        half_width = width / 2
                        half_height = height / 2
                        
                        centers.append([center_x, center_y])
                        half_sizes.append([half_width, half_height])
                    
                    centers_array = np.array(centers)
                    half_sizes_array = np.array(half_sizes)
                    
                    # Log the bounding boxes
                    rr.log("video/yolo_detections", rr.Boxes2D(
                        centers=centers_array,
                        half_sizes=half_sizes_array,
                        colors=colors,
                        labels=labels,
                        class_ids=class_ids
                    ))
                else:
                    # Clear bounding boxes if no valid detections
                    rr.log("video/yolo_detections", rr.Clear(recursive=False))
            else:
                # Clear bounding boxes if no detections for this step
                rr.log("video/yolo_detections", rr.Clear(recursive=False))
        else:
            # Clear bounding boxes if no data for this step
            rr.log("video/yolo_detections", rr.Clear(recursive=False))
    
    def _calculate_warning_level(self, distance: float) -> str:
        """Calculate warning level based on distance thresholds."""
        if distance <= self.config['warning_distance_critical']:
            return 'critical'
        elif distance <= self.config['warning_distance_strong']:
            return 'strong'
        elif distance <= self.config['warning_distance_caution']:
            return 'caution'
        else:
            return 'normal'
    
    def _calculate_direction_to_point(self, camera_position: np.ndarray, camera_quaternion: np.ndarray, 
                                    closest_point: np.ndarray) -> str:
        """Calculate direction from camera to closest point in human-readable terms."""
        # Convert quaternion to rotation matrix (same as in _log_view_cone)
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
        
        # Calculate vector from camera to closest point
        to_point = closest_point - camera_position
        
        # Transform to camera coordinate system
        # Camera coordinates: X=right, Y=up, Z=forward
        camera_coords = R.T @ to_point
        
        # Determine primary direction components
        x, y, z = camera_coords
        
        # Create direction description
        direction_parts = []
        
        # Front/Back (Z-axis)
        if abs(z) > 0.1:  # Threshold to avoid noise
            if z > 0:
                direction_parts.append("ahead")
            else:
                direction_parts.append("behind")
        
        # Left/Right (X-axis)
        if abs(x) > 0.1:
            if x > 0:
                direction_parts.append("right")
            else:
                direction_parts.append("left")
        
        # Up/Down (Y-axis)
        if abs(y) > 0.1:
            if y > 0:
                direction_parts.append("below")
            else:
                direction_parts.append("above")
        
        # Combine directions intelligently
        if len(direction_parts) == 0:
            return "directly in front"
        elif len(direction_parts) == 1:
            return f"to your {direction_parts[0]}"
        elif len(direction_parts) == 2:
            # For two directions, combine them naturally
            if "ahead" in direction_parts or "behind" in direction_parts:
                front_back = next((d for d in direction_parts if d in ["ahead", "behind"]), "")
                other = next((d for d in direction_parts if d not in ["ahead", "behind"]), "")
                return f"{front_back}-{other}"
            else:
                return f"{direction_parts[0]}-{direction_parts[1]}"
        else:
            # Three directions - combine all
            return f"{direction_parts[0]}-{direction_parts[1]}-{direction_parts[2]}"
    

    
    def _generate_warning_message(self, distance: float, direction: str, 
                                warning_level: str, data_type: str = 'points', 
                                segment_id: Optional[int] = None) -> str:
        """Generate appropriate warning message based on context."""
        if not self.config['enable_warnings']:
            if segment_id is not None and segment_id >= 0:
                return f"Distance: {distance:.3f}m (Segment {segment_id})"
            else:
                return f"Distance: {distance:.3f}m"
        
        # Format distance
        distance_str = f"{distance:.2f}m"
        
        # Create object/point descriptor
        if data_type == 'objects' and segment_id is not None and segment_id >= 0:
            object_descriptor = f"Object #{segment_id}"
        elif data_type == 'objects':
            object_descriptor = "Object"
        else:
            object_descriptor = "Point"
        
        # Base message components
        warning_prefixes = {
            'normal': "",
            'caution': "⚠️ CAUTION: ",
            'strong': "⚠️ WARNING: ",
            'critical': "🚨 CRITICAL: "
        }
        
        # Build message
        prefix = warning_prefixes[warning_level]
        
        if warning_level == 'normal':
            if self.config['enable_directional_warnings'] and direction:
                if segment_id is not None and segment_id >= 0:
                    return f"Distance: {distance_str} ({direction}) - {object_descriptor}"
                else:
                    return f"Distance: {distance_str} ({direction})"
            else:
                if segment_id is not None and segment_id >= 0:
                    return f"Distance: {distance_str} - {object_descriptor}"
                else:
                    return f"Distance: {distance_str}"
        
        # Warning messages
        if self.config['enable_directional_warnings'] and direction:
            if warning_level == 'critical':
                return f"{prefix}{object_descriptor} {distance_str} {direction}!"
            else:
                return f"{prefix}{object_descriptor} {distance_str} {direction}"
        else:
            if warning_level == 'critical':
                return f"{prefix}{object_descriptor} at {distance_str}!"
            else:
                return f"{prefix}{object_descriptor} at {distance_str}"
    
    def _get_warning_colors(self, warning_level: str) -> Tuple[int, int, int]:
        """Get color scheme for warning level."""
        warning_colors = {
            'normal': [0, 255, 0],      # Green
            'caution': [255, 255, 0],   # Yellow  
            'strong': [255, 165, 0],    # Orange
            'critical': [255, 0, 0]     # Red
        }
        return warning_colors.get(warning_level, [0, 255, 0])
    
    def _get_warning_radius(self, warning_level: str) -> float:
        """Get point radius based on warning level."""
        warning_radii = {
            'normal': 0.025,
            'caution': 0.035,
            'strong': 0.045,
            'critical': 0.055
        }
        return warning_radii.get(warning_level, 0.025)
    

    def _map_trajectory_to_video_frames(self, num_poses: int) -> Optional[List[Optional[int]]]:
        """Map trajectory steps to video frames using step-based division."""
        if 'video' not in self.data or 'frame_timestamps' not in self.data['video']:
            return None
        
        video_timestamps_ns = self.data['video']['frame_timestamps']
        num_video_frames = len(video_timestamps_ns)
        
        if num_video_frames == 0:
            return None
        
        print(f"Step-based mapping: {num_poses} trajectory poses to {num_video_frames} video frames")
        
        # Calculate the step size for video frames
        # We want to map the entire trajectory to the entire video
        if num_poses <= 1:
            # Special case: only one pose, use first frame
            return [video_timestamps_ns[0]]
        
        # Map trajectory steps evenly across video frames
        frame_mapping = []
        
        for i in range(num_poses):
            # Calculate progress through trajectory (0.0 to 1.0)
            progress = i / (num_poses - 1)
            
            # Map to video frame index
            video_frame_idx = int(progress * (num_video_frames - 1))
            video_frame_idx = min(video_frame_idx, num_video_frames - 1)  # Clamp to valid range
            
            # Get the timestamp for this frame
            frame_timestamp_ns = video_timestamps_ns[video_frame_idx]
            frame_mapping.append(frame_timestamp_ns)
        
        # Log mapping statistics
        unique_frames = len(set(frame_mapping))
        print(f"Mapped trajectory steps to {unique_frames} unique video frames")
        print(f"Video frame step size: {num_video_frames / num_poses:.2f} frames per trajectory step")
        
        return frame_mapping

    def _visualize_video(self) -> None:
        """Visualize the video as a static asset and prepare frame references."""
        video_path = self.data['video']['path']
        
        print(f"Loading video asset: {video_path}")
        
        try:
            # Log video asset (static)
            video_asset = rr.AssetVideo(path=video_path)
            rr.log("video/asset", video_asset, static=True)
            
            # Store video asset for frame reference generation
            self.data['video']['asset'] = video_asset
            self.data['video']['frame_timestamps'] = video_asset.read_frame_timestamps_nanos()
            
            print(f"Loaded video with {len(self.data['video']['frame_timestamps'])} frames")
            
        except Exception as e:
            print(f"Failed to load video asset: {e}")
            # Remove video from data so we don't try to use it later
            if 'video' in self.data:
                del self.data['video']






def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize SLAM pipeline output data in rerun-sdk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_slam_output.py ./enhanced_slam_outputs/slam_analysis_20240115_143045
    python visualize_slam_output.py ./outputs --no-view-cones --no-floor --no-video
    python visualize_slam_output.py ./outputs --cone-angle 45 --grid-size 5.0
    python visualize_slam_output.py ./outputs --subsample-percentage 0.8 --voxel-size 0.15
    python visualize_slam_output.py ./outputs --color_pointcloud_by_classIds --show_segmentation_masks_separately
    python visualize_slam_output.py ./outputs --subsample-percentage 0.8
    python visualize_slam_output.py ./outputs --warning-distance-critical 0.15 --warning-distance-caution 0.8
    python visualize_slam_output.py ./outputs --no-warnings
    python visualize_slam_output.py ./outputs --no-directional-warnings
    python visualize_slam_output.py ./outputs --show-yolo-detections --yolo-confidence-threshold 0.5
    python visualize_slam_output.py ./outputs --show-yolo-detections --no-video
    python visualize_slam_output.py ./outputs --prefer-closest-points
    
Note: 
- **Closest Objects vs Points**: The script automatically detects and prefers closest objects data (with segment IDs) 
  over traditional closest points when available. Objects data provides richer information about which specific 
  segmented object is closest. Use --prefer-closest-points to force using traditional closest points data.
- **Segment ID Visualization**: When closest objects data is available, segment IDs are displayed in warning 
  messages and logged as a separate plot (plots/closest_object_segment_id).
- View cone settings will be automatically configured from pipeline metadata if available.
- Point cloud subsampling uses percentage-based reduction to maintain consistent density across steps.
- The final temporal step always shows 100% of points, earlier steps show the specified percentage.
- Use --no-subsampling to disable subsampling (may cause memory issues with large temporal datasets).
- Segmentation masks are now temporal and only show for the current frame when playing back the trajectory.
- Use --show_segmentation_masks_separately to view each segmentation class as a separate entity in the scene tree.
- Video synchronization uses step-based mapping: trajectory steps are evenly distributed across video frames.
- Video will be automatically loaded from pipeline configuration if available.
- Warning system provides proximity alerts with color-coded visualization and directional guidance.
- Warning levels: Normal (>0.6m, green), Caution (0.4-0.6m, yellow), Strong (0.2-0.4m, orange), Critical (<0.2m, red).
- YOLO detection visualization: Use --show-yolo-detections to overlay bounding boxes on video frames.
- YOLO data is loaded from incremental_analysis_detailed_yolo_detections.csv in the output directory.
- Bounding boxes are color-coded by class and include confidence scores in labels.
- Use --yolo-confidence-threshold to filter detections below a certain confidence level.

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
    parser.add_argument('--prefer-closest-points', action='store_true',
                       help='Prefer closest points over closest objects when both are available')
    parser.add_argument('--no-view-cones', action='store_true',
                       help='Disable view cone visualization')
    parser.add_argument('--no-trajectory-path', action='store_true',
                       help='Disable static trajectory path')
    parser.add_argument('--no-distance-lines', action='store_true',
                       help='Disable distance lines to closest points')
    parser.add_argument('--no-highlight-floor', action='store_true',
                       help='Disable floor point highlighting')
    parser.add_argument("--color_pointcloud_by_classIds", action="store_true",
                        help ="Enables colouring by classIds")
    parser.add_argument("--show_segmentation_masks_separately", action="store_true",
                        help="Show segmentation masks as separate entities per class")

    parser.add_argument('--no-video', action='store_true',
                       help='Disable video visualization')
    parser.add_argument('--show-yolo-detections', action='store_true',
                       help='Enable YOLO bounding box visualization on video frames')
    
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
    
    # Point cloud subsampling parameters
    parser.add_argument('--subsample-percentage', type=float, default=0.5,
                       help='Percentage of points to keep for temporal steps (0.0-1.0), final step shows 100%%')
    parser.add_argument('--voxel-size', type=float, default=0.1,
                       help='Initial voxel size for spatial subsampling (meters)')
    parser.add_argument('--no-subsampling', action='store_true',
                       help='Disable point cloud subsampling (may cause memory issues with large datasets)')
    
    # Video synchronization parameters  
    parser.add_argument('--video-sync-tolerance', type=float, default=1.0,
                       help='(Deprecated - now using step-based sync) Maximum time difference (seconds) to consider trajectory and video frames synchronized')
    
    # YOLO detection parameters
    parser.add_argument('--yolo-confidence-threshold', type=float, default=0.3,
                       help='Minimum confidence threshold for displaying YOLO detections (0.0-1.0)')
    
    # Warning system parameters
    parser.add_argument('--warning-distance-critical', type=float, default=0.2,
                       help='Distance threshold for critical warnings (meters)')
    parser.add_argument('--warning-distance-strong', type=float, default=0.4,
                       help='Distance threshold for strong warnings (meters)')
    parser.add_argument('--warning-distance-caution', type=float, default=0.6,
                       help='Distance threshold for caution warnings (meters)')
    parser.add_argument('--no-warnings', action='store_true',
                       help='Disable proximity warning system')
    parser.add_argument('--no-directional-warnings', action='store_true',
                       help='Disable directional warnings (show distance only)')
    
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
        'show_video': not args.no_video,
        'show_yolo_detections': args.show_yolo_detections,
        'yolo_confidence_threshold': args.yolo_confidence_threshold,
        'floor_threshold': args.floor_threshold,
        'grid_size': args.grid_size,
        'view_cone_angle': args.cone_angle,
        'cone_length_factor': args.cone_length_factor,
        'max_cone_length': args.max_cone_length,
        'subsample_percentage': args.subsample_percentage,
        'voxel_size': args.voxel_size,
        'enable_subsampling': not args.no_subsampling,
        "color_pointcloud_by_classIds": args.color_pointcloud_by_classIds,
        "show_segmentation_masks_separately": args.show_segmentation_masks_separately,
        'video_sync_tolerance': args.video_sync_tolerance,
        # Warning system configuration
        'enable_warnings': not args.no_warnings,
        'enable_directional_warnings': not args.no_directional_warnings,
        'warning_distance_critical': args.warning_distance_critical,
        'warning_distance_strong': args.warning_distance_strong,
        'warning_distance_caution': args.warning_distance_caution,
        'prefer_closest_points': args.prefer_closest_points
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