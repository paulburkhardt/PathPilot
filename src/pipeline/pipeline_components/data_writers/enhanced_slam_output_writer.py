from typing import List, Dict, Any, Optional
import os
import json
import csv
import numpy as np
from datetime import datetime
import pathlib
from plyfile import PlyElement, PlyData
from .abstract_data_writer import AbstractDataWriter
from omegaconf import DictConfig, OmegaConf


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
        self.accumulated_n_closest_points_3d = []  # Now stores arrays of n points
        self.accumulated_n_closest_points_indices = []  # Now stores arrays of n indices
        self.accumulated_n_closest_points_distances = []  # Now stores arrays of n distances
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
                "n_closest_points_distance_2d"
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
             n_closest_points_3d=None, n_closest_points_index=None, n_closest_points_distance_2d=None,
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
            n_closest_points_index: Index of the closest point
            n_closest_points_distance_2d: Current 3D distance to closest point
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
                # Convert to numpy arrays if needed and ensure proper format
                points_3d = np.array(n_closest_points_3d) if not isinstance(n_closest_points_3d, np.ndarray) else n_closest_points_3d
                distances = np.array(n_closest_points_distance_2d) if not isinstance(n_closest_points_distance_2d, np.ndarray) else n_closest_points_distance_2d
                indices = np.array(n_closest_points_index) if n_closest_points_index is not None and not isinstance(n_closest_points_index, np.ndarray) else n_closest_points_index
                
                # Store the arrays of n closest points
                self.accumulated_n_closest_points_3d.append(points_3d.tolist())
                self.accumulated_n_closest_points_distances.append(distances.tolist())
                
                # Store the indices if provided
                if indices is not None:
                    self.accumulated_n_closest_points_indices.append(indices.tolist())
                else:
                    self.accumulated_n_closest_points_indices.append(None)
        
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
        
        if self.save_closest_points and len(self.accumulated_n_closest_points_distances) > 0:
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

        segmentation = None
        if hasattr(self.final_point_cloud, "segmentation_mask_numpy"):
            segmentation = self.final_point_cloud.segmentation_mask_numpy
        
        
        ply_path = self.final_output_dir / f"{self.output_name}_pointcloud.ply"
        
        print(f"Saving point cloud with {len(points)} points to: {ply_path}")
        
        # Prepare vertex data
        if colors is not None and confidence is not None and segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2],
             colors[i, 0], colors[i, 1], colors[i, 2], confidence[i], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('confidence', 'f4'), ('segmentation', 'u1')]
        elif colors is not None and segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2],
             colors[i, 0], colors[i, 1], colors[i, 2], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('segmentation', 'u1')]
        elif confidence is not None and segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2],
             confidence[i], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('confidence', 'f4'), ('segmentation', 'u1')]
        elif segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('segmentation', 'u1')]
        elif colors is not None and confidence is not None:
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
        elif confidence is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2], confidence[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('confidence', 'f4')]
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
                "n_closest_points_3d": self.accumulated_n_closest_points_3d,
                "n_closest_points_indices": self.accumulated_n_closest_points_indices,
                "n_closest_points_distances": self.accumulated_n_closest_points_distances,
                "num_poses": len(self.accumulated_timestamps),
                "analysis_summary": {
                    "avg_n_closest_points_distance_2d": self._calculate_distance_stats("mean") if self.accumulated_n_closest_points_distances else None,
                    "min_n_closest_points_distance_2d": self._calculate_distance_stats("min") if self.accumulated_n_closest_points_distances else None,
                    "max_n_closest_points_distance_2d": self._calculate_distance_stats("max") if self.accumulated_n_closest_points_distances else None,
                }
            }
            
            closest_path = self.final_output_dir / f"{self.output_name}_closest_points.json"
            with open(closest_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:  # csv
            closest_path = self.final_output_dir / f"{self.output_name}_closest_points.csv"
            with open(closest_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'point_idx', 'closest_3d_x', 'closest_3d_y', 'closest_3d_z', 'closest_point_index', 'n_closest_points_distance_2d'])
                
                for step_i in range(len(self.accumulated_n_closest_points_3d)):
                    points_3d = self.accumulated_n_closest_points_3d[step_i]
                    distances = self.accumulated_n_closest_points_distances[step_i]
                    indices = self.accumulated_n_closest_points_indices[step_i] if step_i < len(self.accumulated_n_closest_points_indices) else None
                    
                    # Handle single point or array of points
                    for point_idx, (point, distance) in enumerate(zip(points_3d, distances)):
                        index = indices[point_idx] if indices is not None else None
                        row = [step_i, point_idx, point[0], point[1], point[2], index, distance]
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
            
        colors = None
        if hasattr(self.final_point_cloud, 'rgb_numpy'):
            colors = self.final_point_cloud.rgb_numpy
            
        confidence = None
        if hasattr(self.final_point_cloud, 'confidence_scores_numpy'):
            confidence = self.final_point_cloud.confidence_scores_numpy
        
        segmentation = None
        if hasattr(self.final_point_cloud, "segmentation_mask_numpy"):
            segmentation = self.final_point_cloud.segmentation_mask_numpy
        
        ply_path = output_dir / "pointcloud.ply"
        
        
        # Prepare vertex data
        if colors is not None and confidence is not None and segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2],
             colors[i, 0], colors[i, 1], colors[i, 2], confidence[i], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('confidence', 'f4'), ('segmentation', 'u1')]
        elif colors is not None and segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2],
             colors[i, 0], colors[i, 1], colors[i, 2], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                    ('segmentation', 'u1')]
        elif confidence is not None and segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2],
             confidence[i], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('confidence', 'f4'), ('segmentation', 'u1')]
        elif segmentation is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2], segmentation[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('segmentation', 'u1')]
        elif colors is not None and confidence is not None:
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
        elif confidence is not None:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2], confidence[i])
            for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('confidence', 'f4')]
        else:
            vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2]) for i in range(len(points))
            ]
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
            "pipeline_configuration": self._format_pipeline_config() if self.full_pipeline_config is not None else {},
            "data_summary": {
                "total_poses": len(self.accumulated_timestamps),
                "trajectory_duration": (max(self.accumulated_timestamps) - min(self.accumulated_timestamps)) if len(self.accumulated_timestamps) > 1 else 0,
                "has_floor_data": self.floor_data is not None,
                "n_closest_points_3d_count": len(self.accumulated_n_closest_points_distances)
            },
            "file_paths": results
        }
        
        if self.floor_data:
            metadata["floor_detection"] = self.floor_data
        
        if len(self.accumulated_n_closest_points_distances) > 0:
            metadata["distance_analysis"] = {
                "avg_n_closest_points_distance_2d": self._calculate_distance_stats("mean"),
                "min_n_closest_points_distance_2d": self._calculate_distance_stats("min"),
                "max_n_closest_points_distance_2d": self._calculate_distance_stats("max"),
                "std_n_closest_points_distance_2d": self._calculate_distance_stats("std")
            }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2) 

    def _format_pipeline_config(self) -> Dict[str, Any]:
        """Format the pipeline configuration for cleaner metadata output."""
        if self.full_pipeline_config is None:
            return {}
        
        # Convert DictConfig to regular dict if needed
        if isinstance(self.full_pipeline_config, DictConfig):
            config_dict = OmegaConf.to_container(self.full_pipeline_config, resolve=True)
        else:
            config_dict = self.full_pipeline_config
        
        # Create a clean copy of the configuration
        formatted_config = {}
        
        # Add pipeline structure
        if "pipeline" in config_dict:
            pipeline_config = config_dict["pipeline"]
            formatted_config["pipeline"] = {
                "components": []
            }
            
            # Format each component configuration
            for component_config in pipeline_config.get("components", []):
                component_info = {
                    "type": component_config.get("type", "Unknown"),
                    "config": component_config.get("config", {})
                }
                formatted_config["pipeline"]["components"].append(component_info)
        
        # Add any other top-level configuration keys
        for key, value in config_dict.items():
            if key != "pipeline":
                formatted_config[key] = value
        
        return formatted_config

    def _calculate_distance_stats(self, stat: str) -> float:
        """Calculate distance statistics from arrays of n closest point distances."""
        if not self.accumulated_n_closest_points_distances:
            return None
            
        # Flatten all distance arrays into a single array
        all_distances = []
        for distances in self.accumulated_n_closest_points_distances:
            if isinstance(distances, list):
                all_distances.extend(distances)
            else:
                all_distances.append(distances)
        
        if not all_distances:
            return None
            
        distances_array = np.array(all_distances)
        
        if stat == "mean":
            return float(np.mean(distances_array))
        elif stat == "min":
            return float(np.min(distances_array))
        elif stat == "max":
            return float(np.max(distances_array))
        elif stat == "std":
            return float(np.std(distances_array))
        return None 