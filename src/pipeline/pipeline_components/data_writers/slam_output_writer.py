from typing import List, Dict, Any, Optional
import os
import numpy as np
from datetime import datetime
import pathlib
from plyfile import PlyElement, PlyData
from .abstract_data_writer import AbstractDataWriter


class SLAMOutputWriter(AbstractDataWriter):
    """
    Writer component for saving SLAM outputs (point clouds and trajectories) to intermediate files.
    This enables splitting the pipeline into stages that can run in different environments.
    
    Args:
        output_dir: Directory where SLAM outputs will be written (default: "intermediate_outputs")
        output_name: Base name for output files (default: "slam_output")
        save_point_cloud: Whether to save point cloud as PLY file (default: True)
        save_trajectory: Whether to save camera trajectory as TXT file (default: True)
        create_timestamped_dir: Whether to create timestamped subdirectory (default: True)
        
    Returns:
        Dictionary containing paths to saved files
        
    Raises:
        ValueError: If required data is missing
    """
    
    def __init__(self, output_dir: str = "intermediate_outputs", 
                 output_name: str = "slam_output",
                 save_point_cloud: bool = True,
                 save_trajectory: bool = True,
                 create_timestamped_dir: bool = True) -> None:
        super().__init__(output_dir)
        self.output_name = output_name
        self.save_point_cloud = save_point_cloud
        self.save_trajectory = save_trajectory
        self.create_timestamped_dir = create_timestamped_dir
        
        # Initialize trajectory accumulation lists
        self.accumulated_timestamps = []
        self.accumulated_positions = []
        self.accumulated_quaternions = []
        self.final_point_cloud = None
        
        # Track if this is the final step
        self.is_final_step = False
        
        # Create output directory structure
        self._setup_output_directory()
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires SLAM output data."""
        inputs = ["step_nr"]  # Always need step number
        if self.save_point_cloud:
            inputs.append("point_cloud")
        if self.save_trajectory:
            inputs.extend(["camera_pose", "timestamp"])
        return inputs
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs file paths for the next stage."""
        outputs = []
        if self.save_point_cloud:
            outputs.append("point_cloud_path")
        if self.save_trajectory:
            outputs.append("trajectory_path")
        outputs.append("output_directory")
        return outputs
    
    def _setup_output_directory(self) -> None:
        """Set up the output directory structure."""
        base_dir = pathlib.Path(self._output_dir if self._output_dir else "intermediate_outputs")
        
        if self.create_timestamped_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.final_output_dir = base_dir / f"{self.output_name}_{timestamp}"
        else:
            self.final_output_dir = base_dir / self.output_name
            
        self.final_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"SLAM outputs will be saved to: {self.final_output_dir}")

    def set_final_step(self, is_final: bool = True) -> None:
        """Mark this as the final step to trigger file saving."""
        self.is_final_step = is_final
    
    def _run(self, step_nr: int, point_cloud=None, 
             camera_pose=None, timestamp=None,
             **kwargs: Any) -> Dict[str, Any]:
        """
        Accumulate SLAM outputs and save them on the final step.
        
        Args:
            step_nr: Step number within the pipeline
            point_cloud: Point cloud data entity (if save_point_cloud=True)
            camera_pose: Camera pose object (if save_trajectory=True)
            timestamp: Frame timestamp (if save_trajectory=True)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing paths to saved files (only on final step)
        """
        results = {"output_directory": str(self.final_output_dir)}
        
        # Accumulate trajectory data
        if self.save_trajectory and camera_pose is not None and timestamp is not None:
            self._accumulate_trajectory_data(camera_pose, timestamp, step_nr)
        
        # Store the latest point cloud (assuming accumulating mode gives us the full cloud)
        if self.save_point_cloud and point_cloud is not None:
            self.final_point_cloud = point_cloud
        
        # Save files only if this is marked as the final step or we detect it's the last frame
        # For now, we'll save after every frame to ensure we don't lose data
        # TODO: Implement proper final step detection
        
        # Save point cloud if we have one
        if self.save_point_cloud and self.final_point_cloud is not None:
            ply_path = self._save_point_cloud(self.final_point_cloud, step_nr)
            results["point_cloud_path"] = str(ply_path)
        
        # Save trajectory if we have accumulated data
        if self.save_trajectory and len(self.accumulated_timestamps) > 0:
            txt_path = self._save_trajectory_accumulated(step_nr)
            results["trajectory_path"] = str(txt_path)
        
        # Create a summary file with metadata
        if self.final_point_cloud is not None or len(self.accumulated_timestamps) > 0:
            self._save_metadata(results, step_nr)
        
        return results
    
    def _accumulate_trajectory_data(self, camera_pose, timestamp, step_nr: int) -> None:
        """Accumulate trajectory data from individual frames."""
        try:
            # Extract position and quaternion from the pose object
            # MAST3R uses lietorch.SE3 format with .data attribute containing [x, y, z, qx, qy, qz, qw]
            if hasattr(camera_pose, 'data'):
                # Extract SE3 pose data directly (format: x, y, z, qx, qy, qz, qw)
                pose_data = camera_pose.data.cpu().numpy().reshape(-1)
                if len(pose_data) >= 7:  # x, y, z, qx, qy, qz, qw
                    position = pose_data[:3].astype(np.float32)
                    quaternion = pose_data[3:7].astype(np.float32)
                else:
                    print(f"Warning: Unexpected pose data format at step {step_nr}, got {len(pose_data)} values")
                    return
            elif hasattr(camera_pose, 'matrix'):
                # Fallback: Extract from transformation matrix
                T = camera_pose.matrix().cpu().numpy()
                position = T[:3, 3].astype(np.float32)
                # Convert rotation matrix to quaternion
                from scipy.spatial.transform import Rotation
                quaternion = Rotation.from_matrix(T[:3, :3]).as_quat().astype(np.float32)  # [x, y, z, w]
            else:
                print(f"Warning: Unknown camera pose format at step {step_nr}: {type(camera_pose)}")
                return
            
            # Accumulate the data
            self.accumulated_timestamps.append(float(timestamp))
            self.accumulated_positions.append(position)
            self.accumulated_quaternions.append(quaternion)
            
            if step_nr % 10 == 0:  # Print every 10 steps to reduce spam
                print(f"Accumulated trajectory data for step {step_nr}, total poses: {len(self.accumulated_timestamps)}")
            
        except Exception as e:
            print(f"Error accumulating trajectory data at step {step_nr}: {e}")
            print(f"Camera pose type: {type(camera_pose)}")
            if hasattr(camera_pose, '__dict__'):
                print(f"Camera pose attributes: {list(camera_pose.__dict__.keys())}")
            if hasattr(camera_pose, 'data'):
                print(f"Pose data shape: {camera_pose.data.shape}")
                print(f"Pose data: {camera_pose.data.cpu().numpy()}")
    
    def _save_point_cloud(self, point_cloud, step_nr: int) -> pathlib.Path:
        """Save point cloud as PLY file."""
        # Extract point cloud data
        if hasattr(point_cloud, 'point_cloud_numpy'):
            points = point_cloud.point_cloud_numpy
        elif hasattr(point_cloud, 'point_cloud'):
            points = point_cloud.point_cloud
        else:
            points = point_cloud
            
        # Extract colors if available
        colors = None
        if hasattr(point_cloud, 'rgb_numpy'):
            colors = point_cloud.rgb_numpy
        elif hasattr(point_cloud, 'rgb'):
            colors = point_cloud.rgb
            
        # Extract confidence scores if available
        confidence = None
        if hasattr(point_cloud, 'confidence_scores_numpy'):
            confidence = point_cloud.confidence_scores_numpy
        elif hasattr(point_cloud, 'confidence_scores'):
            confidence = point_cloud.confidence_scores
        
        # Create PLY file path
        ply_path = self.final_output_dir / f"{self.output_name}.ply"
        
        print(f"Saving point cloud with {len(points)} points to: {ply_path}")
        
        # Prepare vertex data
        vertex_data = [
            (points[i, 0], points[i, 1], points[i, 2]) for i in range(len(points))
        ]
        
        # Add colors if available
        if colors is not None:
            vertex_data = [
                (points[i, 0], points[i, 1], points[i, 2],
                 colors[i, 0], colors[i, 1], colors[i, 2])
                for i in range(len(points))
            ]
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        else:
            vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        
        # Add confidence scores if available
        if confidence is not None:
            if colors is not None:
                vertex_data = [
                    (points[i, 0], points[i, 1], points[i, 2],
                     colors[i, 0], colors[i, 1], colors[i, 2], confidence[i])
                    for i in range(len(points))
                ]
                vertex_dtype.append(('confidence', 'f4'))
            else:
                vertex_data = [
                    (points[i, 0], points[i, 1], points[i, 2], confidence[i])
                    for i in range(len(points))
                ]
                vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('confidence', 'f4')]
        
        # Create PLY element
        vertex_array = np.array(vertex_data, dtype=vertex_dtype)
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        
        # Write PLY file
        PlyData([vertex_element]).write(str(ply_path))
        
        print(f"Successfully saved point cloud to: {ply_path}")
        return ply_path
    
    def _save_trajectory_accumulated(self, step_nr: int) -> pathlib.Path:
        """Save accumulated camera trajectory as TXT file."""
        # Create trajectory file path
        txt_path = self.final_output_dir / f"{self.output_name}.txt"
        
        print(f"Saving trajectory with {len(self.accumulated_timestamps)} poses to: {txt_path}")
        
        # Convert lists to numpy arrays
        timestamps = np.array(self.accumulated_timestamps)
        positions = np.array(self.accumulated_positions)
        quaternions = np.array(self.accumulated_quaternions)
        
        # Write trajectory file (format: timestamp x y z qx qy qz qw)
        with open(txt_path, 'w') as f:
            for i in range(len(timestamps)):
                timestamp = timestamps[i]
                pos = positions[i]
                quat = quaternions[i]
                
                f.write(f"{timestamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                       f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
        
        print(f"Successfully saved trajectory to: {txt_path}")
        return txt_path

    def _save_metadata(self, results: Dict[str, Any], step_nr: int) -> None:
        """Save metadata about the SLAM output files."""
        metadata_path = self.final_output_dir / "metadata.txt"
        
        with open(metadata_path, 'w') as f:
            f.write(f"SLAM Output Metadata\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Final step number: {step_nr}\n")
            f.write(f"Output directory: {self.final_output_dir}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - save_point_cloud: {self.save_point_cloud}\n")
            f.write(f"  - save_trajectory: {self.save_trajectory}\n")
            f.write(f"  - output_name: {self.output_name}\n")
            
            if self.save_point_cloud and "point_cloud_path" in results:
                f.write(f"Point Cloud:\n")
                f.write(f"  - file: {results['point_cloud_path']}\n")
                if self.final_point_cloud is not None:
                    if hasattr(self.final_point_cloud, 'point_cloud_numpy'):
                        points = self.final_point_cloud.point_cloud_numpy
                    elif hasattr(self.final_point_cloud, 'point_cloud'):
                        points = self.final_point_cloud.point_cloud
                    else:
                        points = self.final_point_cloud
                    f.write(f"  - num_points: {len(points)}\n")
            
            if self.save_trajectory and len(self.accumulated_timestamps) > 0:
                f.write(f"Trajectory:\n")
                f.write(f"  - file: {results.get('trajectory_path', 'N/A')}\n")
                f.write(f"  - num_poses: {len(self.accumulated_timestamps)}\n")
                f.write(f"  - time_range: {min(self.accumulated_timestamps):.3f} to {max(self.accumulated_timestamps):.3f}\n")
        
        print(f"Saved metadata to: {metadata_path}") 