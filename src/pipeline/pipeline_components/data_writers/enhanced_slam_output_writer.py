from typing import List, Dict, Any, Optional
import os
import json
import csv
import numpy as np
from datetime import datetime
import pathlib
from plyfile import PlyElement, PlyData
from .abstract_data_writer import AbstractDataWriter


class EnhancedSLAMOutputWriter(AbstractDataWriter):
    """
    Enhanced writer component for saving comprehensive SLAM analysis outputs.
    Handles point clouds, trajectories, floor detection, and closest point analysis.
    
    Args:
        output_dir: Directory where outputs will be written (default: "enhanced_slam_outputs")
        output_name: Base name for output files (default: "slam_analysis")
        save_point_cloud: Whether to save point cloud as PLY file (default: True)
        save_trajectory: Whether to save camera trajectory as TXT file (default: True)
        save_floor_data: Whether to save floor detection results (default: True)
        save_closest_points: Whether to save closest point analysis (default: True)
        save_intermediate: Whether to save intermediate results every N frames (default: False)
        intermediate_interval: Save interval for intermediate results (default: 10)
        create_timestamped_dir: Whether to create timestamped subdirectory (default: True)
        analysis_format: Format for analysis data files - 'json' or 'csv' (default: 'json')
        
    Returns:
        Dictionary containing paths to saved files
        
    Raises:
        ValueError: If required data is missing
    """
    
    def __init__(self, 
                 output_dir: str = "enhanced_slam_outputs", 
                 output_name: str = "slam_analysis",
                 save_point_cloud: bool = True,
                 save_trajectory: bool = True,
                 save_floor_data: bool = True,
                 save_closest_points: bool = True,
                 save_intermediate: bool = False,
                 intermediate_interval: int = 10,
                 create_timestamped_dir: bool = True,
                 analysis_format: str = 'json') -> None:
        super().__init__(output_dir)
        self.output_name = output_name
        self.save_point_cloud = save_point_cloud
        self.save_trajectory = save_trajectory
        self.save_floor_data = save_floor_data
        self.save_closest_points = save_closest_points
        self.save_intermediate = save_intermediate
        self.intermediate_interval = intermediate_interval
        self.create_timestamped_dir = create_timestamped_dir
        self.analysis_format = analysis_format.lower()
        
        if self.analysis_format not in ['json', 'csv']:
            raise ValueError("analysis_format must be 'json' or 'csv'")
        
        # Initialize data accumulation
        self.accumulated_timestamps = []
        self.accumulated_positions = []
        self.accumulated_quaternions = []
        self.accumulated_closest_points_3d = []
        self.accumulated_distances_3d = []
        self.accumulated_closest_points_floor = []
        self.accumulated_distances_floor = []
        self.floor_data = None
        self.final_point_cloud = None
        
        # Create output directory structure
        self._setup_output_directory()
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires comprehensive SLAM analysis data."""
        inputs = ["step_nr"]  # Always need step number
        
        if self.save_point_cloud:
            inputs.append("point_cloud")
        if self.save_trajectory:
            inputs.extend(["camera_pose", "timestamp"])
        if self.save_floor_data:
            inputs.extend(["floor_normal", "floor_offset"])
        if self.save_closest_points:
            inputs.extend([
                "n_closest_points_3d", 
                "n_closest_points_index", 
                "n_closest_points_distance_2d", 
        ])
        # Floor data and closest points are optional and will be handled via optional_inputs
            
        return inputs
    
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs file paths for the next stage."""
        return []
    
    def _setup_output_directory(self) -> None:
        """Set up the output directory structure."""
        base_dir = pathlib.Path(self._output_dir if self._output_dir else "enhanced_slam_outputs")
        
        if self.create_timestamped_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.final_output_dir = base_dir / f"{self.output_name}_{timestamp}"
        else:
            self.final_output_dir = base_dir / self.output_name
            
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        if self.save_intermediate:
            (self.final_output_dir / "intermediate").mkdir(exist_ok=True)
            
        print(f"Enhanced SLAM outputs will be saved to: {self.final_output_dir}")

    def _run(self, step_nr: int, 
             point_cloud=None, camera_pose=None, timestamp=None,
             floor_normal=None, floor_offset=None,
             n_closest_points_3d=None, n_closest_points_distance_2d=None,
             **kwargs: Any) -> Dict[str, Any]:
        """
        Accumulate SLAM analysis data and save according to configuration.
        
        Args:
            step_nr: Step number within the pipeline
            point_cloud: Point cloud data entity
            camera_pose: Camera pose object
            timestamp: Frame timestamp
            floor_normal: Floor plane normal vector
            floor_offset: Floor plane offset
            n_closest_points_3d: Current closest 3D point
            n_closest_points_distance_2d: Current 3D distance to closest point
            closest_point_floor: Current closest floor-projected point
            distance_floor: Current floor distance
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing paths to saved files
        """
        results = {"output_directory": str(self.final_output_dir)}
        
        # Accumulate trajectory data
        if self.save_trajectory and camera_pose is not None and timestamp is not None:
            self._accumulate_trajectory_data(camera_pose, timestamp, step_nr)
        
        # Store floor data when available (only store the latest/best)
        if self.save_floor_data and floor_normal is not None and floor_offset is not None:
            self.floor_data = {
                "floor_normal": floor_normal.tolist() if isinstance(floor_normal, np.ndarray) else floor_normal,
                "floor_offset": float(floor_offset),
                "detection_step": step_nr,
                "timestamp": float(timestamp) if timestamp is not None else None
            }
        
        # Accumulate closest point data
        if self.save_closest_points:
            if n_closest_points_3d is not None and n_closest_points_distance_2d is not None:
                self.accumulated_closest_points_3d.append(
                    n_closest_points_3d.tolist() if isinstance(n_closest_points_3d, np.ndarray) else n_closest_points_3d
                )
                self.accumulated_distances_3d.append(float(n_closest_points_distance_2d))
            
            if closest_point_floor is not None and distance_floor is not None:
                self.accumulated_closest_points_floor.append(
                    closest_point_floor.tolist() if isinstance(closest_point_floor, np.ndarray) else closest_point_floor
                )
                self.accumulated_distances_floor.append(float(distance_floor))
        
        # Store the latest point cloud
        if self.save_point_cloud and point_cloud is not None:
            self.final_point_cloud = point_cloud
        
        # Save intermediate results if configured
        if self.save_intermediate and (step_nr + 1) % self.intermediate_interval == 0:
            self._save_intermediate_results(step_nr)
        
        # Always save final results (overwrite previous)
        if self.save_point_cloud and self.final_point_cloud is not None:
            ply_path = self._save_point_cloud()
            results["point_cloud_path"] = str(ply_path)
        
        if self.save_trajectory and len(self.accumulated_timestamps) > 0:
            txt_path = self._save_trajectory()
            results["trajectory_path"] = str(txt_path)
        
        if self.save_floor_data and self.floor_data is not None:
            floor_path = self._save_floor_data()
            results["floor_data_path"] = str(floor_path)
        
        if self.save_closest_points and len(self.accumulated_distances_3d) > 0:
            closest_path = self._save_closest_points_data()
            results["closest_points_path"] = str(closest_path)
        
        # Save metadata
        self._save_metadata(results, step_nr)
        
        return {}
    
    def _accumulate_trajectory_data(self, camera_pose, timestamp, step_nr: int) -> None:
        """Accumulate trajectory data from individual frames."""
        try:
            # Extract position and quaternion from the pose object
            if hasattr(camera_pose, 'data'):
                pose_data = camera_pose.data.cpu().numpy().reshape(-1)
                if len(pose_data) >= 7:
                    position = pose_data[:3].astype(np.float32)
                    quaternion = pose_data[3:7].astype(np.float32)
                else:
                    print(f"Warning: Unexpected pose data format at step {step_nr}")
                    return
            elif hasattr(camera_pose, 'matrix'):
                T = camera_pose.matrix().cpu().numpy()
                position = T[:3, 3].astype(np.float32)
                from scipy.spatial.transform import Rotation
                quaternion = Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
            else:
                print(f"Warning: Unknown camera pose format at step {step_nr}")
                return
            
            self.accumulated_timestamps.append(float(timestamp))
            self.accumulated_positions.append(position)
            self.accumulated_quaternions.append(quaternion)
            
        except Exception as e:
            print(f"Error accumulating trajectory data at step {step_nr}: {e}")
    
    def _save_point_cloud(self) -> pathlib.Path:
        """Save point cloud as PLY file."""
        # Extract point cloud data
        if hasattr(self.final_point_cloud, 'point_cloud_numpy'):
            points = self.final_point_cloud.point_cloud_numpy
        else:
            points = self.final_point_cloud
            
        colors = None
        if hasattr(self.final_point_cloud, 'rgb_numpy'):
            colors = self.final_point_cloud.rgb_numpy
            
        confidence = None
        if hasattr(self.final_point_cloud, 'confidence_scores_numpy'):
            confidence = self.final_point_cloud.confidence_scores_numpy
        
        ply_path = self.final_output_dir / f"{self.output_name}_pointcloud.ply"
        
        print(f"Saving point cloud with {len(points)} points to: {ply_path}")
        
        # Prepare vertex data
        if colors is not None and confidence is not None:
            vertex_data = [
                (points[i, 0], points[i, 1], points[i, 2],
                 colors[i, 0], colors[i, 1], colors[i, 2], confidence[i])
                for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('confidence', 'f4')]
        elif colors is not None:
            vertex_data = [
                (points[i, 0], points[i, 1], points[i, 2],
                 colors[i, 0], colors[i, 1], colors[i, 2])
                for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        else:
            vertex_data = [
                (points[i, 0], points[i, 1], points[i, 2]) for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        
        vertex_array = np.array(vertex_data, dtype=vertex_dtype)
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        PlyData([vertex_element]).write(str(ply_path))
        
        return ply_path
    
    def _save_trajectory(self) -> pathlib.Path:
        """Save accumulated camera trajectory as TXT file."""
        txt_path = self.final_output_dir / f"{self.output_name}_trajectory.txt"
        
        print(f"Saving trajectory with {len(self.accumulated_timestamps)} poses to: {txt_path}")
        
        timestamps = np.array(self.accumulated_timestamps)
        positions = np.array(self.accumulated_positions)
        quaternions = np.array(self.accumulated_quaternions)
        
        with open(txt_path, 'w') as f:
            for i in range(len(timestamps)):
                timestamp = timestamps[i]
                pos = positions[i]
                quat = quaternions[i]
                
                f.write(f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
        
        return txt_path
    
    def _save_floor_data(self) -> pathlib.Path:
        """Save floor detection data."""
        if self.analysis_format == 'json':
            floor_path = self.final_output_dir / f"{self.output_name}_floor_data.json"
            with open(floor_path, 'w') as f:
                json.dump(self.floor_data, f, indent=2)
        else:  # csv
            floor_path = self.final_output_dir / f"{self.output_name}_floor_data.csv"
            with open(floor_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['normal_x', 'normal_y', 'normal_z', 'offset', 'detection_step', 'timestamp'])
                normal = self.floor_data['floor_normal']
                writer.writerow([normal[0], normal[1], normal[2], 
                               self.floor_data['floor_offset'],
                               self.floor_data['detection_step'],
                               self.floor_data['timestamp']])
        
        return floor_path
    
    def _save_closest_points_data(self) -> pathlib.Path:
        """Save closest point analysis data."""
        if self.analysis_format == 'json':
            data = {
                "closest_points_3d": self.accumulated_closest_points_3d,
                "distances_3d": self.accumulated_distances_3d,
                "closest_points_floor": self.accumulated_closest_points_floor,
                "distances_floor": self.accumulated_distances_floor,
                "num_poses": len(self.accumulated_timestamps),
                "analysis_summary": {
                    "avg_n_closest_points_distance_2d": float(np.mean(self.accumulated_distances_3d)) if self.accumulated_distances_3d else None,
                    "min_n_closest_points_distance_2d": float(np.min(self.accumulated_distances_3d)) if self.accumulated_distances_3d else None,
                    "max_n_closest_points_distance_2d": float(np.max(self.accumulated_distances_3d)) if self.accumulated_distances_3d else None,
                    "avg_distance_floor": float(np.mean(self.accumulated_distances_floor)) if self.accumulated_distances_floor else None,
                    "min_distance_floor": float(np.min(self.accumulated_distances_floor)) if self.accumulated_distances_floor else None,
                    "max_distance_floor": float(np.max(self.accumulated_distances_floor)) if self.accumulated_distances_floor else None,
                }
            }
            
            closest_path = self.final_output_dir / f"{self.output_name}_closest_points.json"
            with open(closest_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:  # csv
            closest_path = self.final_output_dir / f"{self.output_name}_closest_points.csv"
            with open(closest_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'closest_3d_x', 'closest_3d_y', 'closest_3d_z', 'n_closest_points_distance_2d',
                               'closest_floor_x', 'closest_floor_y', 'closest_floor_z', 'distance_floor'])
                
                max_len = max(len(self.accumulated_closest_points_3d), len(self.accumulated_closest_points_floor))
                for i in range(max_len):
                    row = [i]
                    
                    if i < len(self.accumulated_closest_points_3d):
                        pt3d = self.accumulated_closest_points_3d[i]
                        row.extend([pt3d[0], pt3d[1], pt3d[2], self.accumulated_distances_3d[i]])
                    else:
                        row.extend([None, None, None, None])
                    
                    if i < len(self.accumulated_closest_points_floor):
                        ptfloor = self.accumulated_closest_points_floor[i]
                        row.extend([ptfloor[0], ptfloor[1], ptfloor[2], self.accumulated_distances_floor[i]])
                    else:
                        row.extend([None, None, None, None])
                    
                    writer.writerow(row)
        
        return closest_path
    
    def _save_intermediate_results(self, step_nr: int) -> None:
        """Save intermediate results for this step."""
        intermediate_dir = self.final_output_dir / "intermediate" / f"step_{step_nr:06d}"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current point cloud if available
        if self.save_point_cloud and self.final_point_cloud is not None:
            self._save_intermediate_point_cloud(intermediate_dir)
        
        # Save current trajectory
        if self.save_trajectory and len(self.accumulated_timestamps) > 0:
            self._save_intermediate_trajectory(intermediate_dir)
            
        print(f"Saved intermediate results for step {step_nr}")
    
    def _save_intermediate_point_cloud(self, output_dir: pathlib.Path) -> None:
        """Save intermediate point cloud."""
        if hasattr(self.final_point_cloud, 'point_cloud_numpy'):
            points = self.final_point_cloud.point_cloud_numpy
        else:
            points = self.final_point_cloud
        
        ply_path = output_dir / "pointcloud.ply"
        
        vertex_data = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(len(points))]
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        
        vertex_array = np.array(vertex_data, dtype=vertex_dtype)
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        PlyData([vertex_element]).write(str(ply_path))
    
    def _save_intermediate_trajectory(self, output_dir: pathlib.Path) -> None:
        """Save intermediate trajectory."""
        txt_path = output_dir / "trajectory.txt"
        
        timestamps = np.array(self.accumulated_timestamps)
        positions = np.array(self.accumulated_positions)
        quaternions = np.array(self.accumulated_quaternions)
        
        with open(txt_path, 'w') as f:
            for i in range(len(timestamps)):
                timestamp = timestamps[i]
                pos = positions[i]
                quat = quaternions[i]
                
                f.write(f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
    
    def _save_metadata(self, results: Dict[str, Any], step_nr: int) -> None:
        """Save comprehensive metadata about the analysis."""
        metadata_path = self.final_output_dir / "metadata.json"
        
        metadata = {
            "generation_info": {
                "generated": datetime.now().isoformat(),
                "final_step_number": step_nr,
                "output_directory": str(self.final_output_dir)
            },
            "configuration": {
                "save_point_cloud": self.save_point_cloud,
                "save_trajectory": self.save_trajectory,
                "save_floor_data": self.save_floor_data,
                "save_closest_points": self.save_closest_points,
                "save_intermediate": self.save_intermediate,
                "intermediate_interval": self.intermediate_interval,
                "analysis_format": self.analysis_format,
                "output_name": self.output_name
            },
            "data_summary": {
                "total_poses": len(self.accumulated_timestamps),
                "trajectory_duration": (max(self.accumulated_timestamps) - min(self.accumulated_timestamps)) if len(self.accumulated_timestamps) > 1 else 0,
                "has_floor_data": self.floor_data is not None,
                "closest_points_3d_count": len(self.accumulated_distances_3d),
                "closest_points_floor_count": len(self.accumulated_distances_floor)
            },
            "file_paths": results
        }
        
        if self.floor_data:
            metadata["floor_detection"] = self.floor_data
        
        if len(self.accumulated_distances_3d) > 0:
            metadata["distance_analysis"] = {
                "avg_n_closest_points_distance_2d": float(np.mean(self.accumulated_distances_3d)),
                "min_n_closest_points_distance_2d": float(np.min(self.accumulated_distances_3d)),
                "max_n_closest_points_distance_2d": float(np.max(self.accumulated_distances_3d)),
                "std_n_closest_points_distance_2d": float(np.std(self.accumulated_distances_3d))
            }
            
        if len(self.accumulated_distances_floor) > 0:
            metadata["distance_analysis"]["avg_distance_floor"] = float(np.mean(self.accumulated_distances_floor))
            metadata["distance_analysis"]["min_distance_floor"] = float(np.min(self.accumulated_distances_floor))
            metadata["distance_analysis"]["max_distance_floor"] = float(np.max(self.accumulated_distances_floor))
            metadata["distance_analysis"]["std_distance_floor"] = float(np.std(self.accumulated_distances_floor))
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2) 