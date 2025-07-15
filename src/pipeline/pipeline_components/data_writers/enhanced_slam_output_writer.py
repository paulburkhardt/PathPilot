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
    
    When closest point segment IDs are available, creates a CSV file with detected objects containing:
    step, closest_3d_x, closest_3d_y, closest_3d_z, closest_2d_distance, segment_id, class_label
    
    Args:
        output_dir: Directory where outputs will be written (default: "enhanced_slam_outputs")
        output_name: Base name for output files (default: "slam_analysis")
        save_point_cloud: Whether to save point cloud as PLY file (default: True)
        save_trajectory: Whether to save camera trajectory as TXT file (default: True)
        save_floor_data: Whether to save floor detection results (default: True)
        save_closest_points: Whether to save closest point analysis (default: True)
        save_object_mapping: Whether to save object mapping (default: True)
        save_yolo_detections: Whether to save YOLO object detection results (default: True)
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
                 save_object_mapping: bool = True,
                 save_yolo_detections: bool = True,
                 save_intermediate: bool = False,
                 intermediate_interval: int = 10,
                 create_timestamped_dir: bool = True,
                 analysis_format: str = 'json',
                 save_closest_points_segment_ids: bool = True) -> None:
        super().__init__(output_dir)
        self.output_name = output_name
        self.save_point_cloud = save_point_cloud
        self.save_trajectory = save_trajectory
        self.save_floor_data = save_floor_data
        self.save_closest_points = save_closest_points
        self.save_object_mapping = save_object_mapping
        self.save_yolo_detections = save_yolo_detections
        self.save_intermediate = save_intermediate
        self.intermediate_interval = intermediate_interval
        self.create_timestamped_dir = create_timestamped_dir
        self.analysis_format = analysis_format.lower()
        self.save_closest_points_segment_ids = save_closest_points_segment_ids
        
        if self.analysis_format not in ['json', 'csv']:
            raise ValueError("analysis_format must be 'json' or 'csv'")
        
        # Initialize data accumulation
        self.accumulated_timestamps = []
        self.accumulated_positions = []
        self.accumulated_quaternions = []
        self.accumulated_n_closest_points_3d = []  # Now stores arrays of n points
        self.accumulated_n_closest_points_indices = []  # Now stores arrays of n indices
        self.accumulated_n_closest_points_distances = []  # Now stores arrays of n distances
        # Only initialize segment IDs accumulator if we need to save them
        if self.save_closest_points_segment_ids:
            self.accumulated_n_closest_points_segment_ids = []  # Store segment IDs for each closest point
            self.accumulated_n_closest_points_class_labels = []  # Store class labels for each closest point
        self.accumulated_yolo_detections = []  # Store YOLO detections per frame
        self.accumulated_segmentation_labels = []  # Store segmentation labels per frame
        self.floor_data = None
        self.final_point_cloud = None
        self.object_mapping = None
        
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
        if self.save_object_mapping:
            inputs.extend(["objects", "object_dict"])
            # Only add segment IDs if specifically requested
            if self.save_closest_points_segment_ids:
                inputs.append("n_closest_points_segment_ids")
                inputs.append("n_closest_points_class_labels")
        if self.save_yolo_detections:
            inputs.append("yolo_detections")
        # Always try to get segmentation labels if available
        inputs.append("segmentation_labels")
        # Floor data, closest points, YOLO detections, and segmentation labels are optional and will be handled via optional_inputs
            
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
             objects=None, object_dict=None,
             n_closest_points_segment_ids=None, n_closest_points_class_labels=None, 
             yolo_detections=None, segmentation_labels=None,
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
            objects: Current frame objects
            object_dict: Object3D mapping dictionary
            n_closest_points_segment_ids: Segment IDs for the closest points (e.g., 1, 2, 3)
            n_closest_points_class_labels: Class labels for the closest points (e.g., 'Chair', 'Table')
            yolo_detections: YOLO object detection results
            segmentation_labels: Mapping of segment IDs to class labels
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
                
                # Store the segment IDs if provided and if we're configured to save them
                if self.save_closest_points_segment_ids:
                    if n_closest_points_segment_ids is not None:
                        self.accumulated_n_closest_points_segment_ids.append(n_closest_points_segment_ids)
                    else:
                        self.accumulated_n_closest_points_segment_ids.append(None)
                    
                    # Store the class labels if provided
                    if n_closest_points_class_labels is not None:
                        self.accumulated_n_closest_points_class_labels.append(n_closest_points_class_labels)
                    else:
                        self.accumulated_n_closest_points_class_labels.append(None)
        
        # Accumulate YOLO detection data
        if self.save_yolo_detections:
            yolo_data = {
                "step_nr": step_nr,
                "timestamp": float(timestamp) if timestamp is not None else None,
                "detections": yolo_detections if yolo_detections is not None else {}
            }
            self.accumulated_yolo_detections.append(yolo_data)
        
        # Accumulate segmentation labels data (always save if available)
        if segmentation_labels is not None and segmentation_labels:
            labels_data = {
                "step_nr": step_nr,
                "timestamp": float(timestamp) if timestamp is not None else None,
                "segmentation_labels": segmentation_labels
            }
            self.accumulated_segmentation_labels.append(labels_data)
        
        # Store the latest point cloud
        if self.save_point_cloud and point_cloud is not None:
            self.final_point_cloud = point_cloud
        
        # Store object mapping data when available
        if self.save_object_mapping and object_dict is not None:
            self.object_mapping = object_dict
        
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
            
        # Save closest points with segment IDs if segment IDs are available
        if any(segment_ids is not None for segment_ids in self.accumulated_n_closest_points_segment_ids):
            closest_objects_path = self._save_closest_objects_csv()
            results["closest_objects_path"] = str(closest_objects_path)
        
        if self.save_object_mapping and self.object_mapping is not None:
            object_path = self._save_object_mapping_data()
            results["object_mapping_path"] = str(object_path)
        
        if self.save_object_mapping and self.object_mapping is not None:
            object_path = self._save_object_mapping_data()
            results["object_mapping_path"] = str(object_path)
        
        if self.save_yolo_detections and len(self.accumulated_yolo_detections) > 0:
            yolo_path = self._save_yolo_detections()
            results["yolo_detections_path"] = str(yolo_path)
        
        if len(self.accumulated_segmentation_labels) > 0:
            labels_path = self._save_segmentation_labels()
            results["segmentation_labels_path"] = str(labels_path)
        
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
    
    def _save_object_mapping_data(self) -> pathlib.Path:
        """Save Object3D mapping data."""
        if self.analysis_format == 'json':
            object_path = self.final_output_dir / f"{self.output_name}_object_mapping.json"
            
            # Convert Object3D objects to serializable format
            serializable_objects = {}
            for obj_id, obj in self.object_mapping.items():
                # Save points to a separate .npy file if available
                points_file = None
                if hasattr(obj, 'points') and obj.points is not None:
                    points_file = self.final_output_dir / f"object_{obj_id}_points.npy"
                    np.save(points_file, obj.points)
                    points_file = str(points_file)
                serializable_objects[str(obj_id)] = {
                    "id": obj.id,
                    "mask_ids": list(obj.mask_id),
                    "centroid": obj.centroid.tolist() if isinstance(obj.centroid, np.ndarray) else obj.centroid,
                    "aabb": obj.aabb,
                    "cum_sum": obj.cum_sum.tolist() if isinstance(obj.cum_sum, np.ndarray) else obj.cum_sum,
                    "cum_len": obj.cum_len,
                    "description": obj.description,
                    "embeddings_shape": obj.embeddings.shape if obj.embeddings is not None else None,
                    "running_embedding_shape": obj.running_embedding.shape if obj.running_embedding is not None else None,
                    "running_embedding_weight": obj.running_embedding_weight,
                    "points_file": points_file
                }
            data = {
                "object_mapping": serializable_objects,
                "total_objects": len(self.object_mapping),
                "object_ids": list(self.object_mapping.keys())
            }
            with open(object_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:  # csv
            object_path = self.final_output_dir / f"{self.output_name}_object_mapping.csv"
            with open(object_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['object_id', 'mask_ids', 'centroid_x', 'centroid_y', 'centroid_z', 
                               'aabb_min_x', 'aabb_max_x', 'aabb_min_y', 'aabb_max_y', 'aabb_min_z', 'aabb_max_z',
                               'cum_sum_x', 'cum_sum_y', 'cum_sum_z', 'cum_len', 'description', 'points_file'])
                
                for obj_id, obj in self.object_mapping.items():
                    mask_ids_str = ','.join(map(str, obj.mask_id))
                    centroid = obj.centroid
                    aabb = obj.aabb
                    cum_sum = obj.cum_sum
                    points_file = ''
                    if hasattr(obj, 'points') and obj.points is not None:
                        points_file_path = self.final_output_dir / f"object_{obj_id}_points.npy"
                        np.save(points_file_path, obj.points)
                        points_file = str(points_file_path)
                    row = [
                        obj_id,
                        mask_ids_str,
                        centroid[0], centroid[1], centroid[2],
                        aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5],
                        cum_sum[0], cum_sum[1], cum_sum[2],
                        obj.cum_len,
                        obj.description or "",
                        points_file
                    ]
                    writer.writerow(row)
        
        print(f"Saving object mapping with {len(self.object_mapping)} objects to: {object_path}")
        return object_path
    
    def _save_closest_objects_csv(self) -> pathlib.Path:
        """Save closest points with segment IDs in CSV format."""
        closest_objects_path = self.final_output_dir / f"{self.output_name}_closest_objects.csv"
        
        print(f"Saving closest objects data to: {closest_objects_path}")
        
        with open(closest_objects_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header based on whether segment IDs are available
            if self.save_closest_points_segment_ids:
                writer.writerow(['step', 'closest_3d_x', 'closest_3d_y', 'closest_3d_z', 'closest_2d_distance', 'segment_id', 'class_label'])
            else:
                writer.writerow(['step', 'closest_3d_x', 'closest_3d_y', 'closest_3d_z', 'closest_2d_distance'])
            
            for step_i in range(len(self.accumulated_n_closest_points_3d)):
                points_3d = self.accumulated_n_closest_points_3d[step_i]
                distances = self.accumulated_n_closest_points_distances[step_i]
                
                # Get segment IDs if available
                segment_ids = None
                class_labels = None
                if self.save_closest_points_segment_ids and hasattr(self, 'accumulated_n_closest_points_segment_ids'):
                    segment_ids = self.accumulated_n_closest_points_segment_ids[step_i] if step_i < len(self.accumulated_n_closest_points_segment_ids) else None
                    class_labels = self.accumulated_n_closest_points_class_labels[step_i] if hasattr(self, 'accumulated_n_closest_points_class_labels') and step_i < len(self.accumulated_n_closest_points_class_labels) else None
                
                # Handle case where we have data
                if points_3d and distances:
                    # Write each closest point as a separate row
                    for point_idx, (point, distance) in enumerate(zip(points_3d, distances)):
                        row = [
                            step_i,                # step
                            point[0],              # closest_3d_x
                            point[1],              # closest_3d_y  
                            point[2],              # closest_3d_z
                            distance,              # closest_2d_distance
                        ]
                        
                        # Add segment ID and class label if enabled
                        if self.save_closest_points_segment_ids:
                            segment_id = segment_ids[point_idx] if segment_ids and point_idx < len(segment_ids) else -1
                            class_label = class_labels[point_idx] if class_labels and point_idx < len(class_labels) else "unknown"
                            row.append(segment_id)
                            row.append(class_label)
                        
                        writer.writerow(row)
        
        return closest_objects_path
    
    def _save_yolo_detections(self) -> pathlib.Path:
        """Save YOLO detection data."""
        if self.analysis_format == 'json':
            yolo_path = self.final_output_dir / f"{self.output_name}_yolo_detections.json"
            
            # Create summary statistics
            total_detections = 0
            class_counts = {}
            frames_with_detections = 0
            
            for frame_data in self.accumulated_yolo_detections:
                detections = frame_data.get("detections", {})
                if detections:
                    frames_with_detections += 1
                    for class_name, objects in detections.items():
                        total_detections += len(objects)
                        class_counts[class_name] = class_counts.get(class_name, 0) + len(objects)
            
            data = {
                "summary": {
                    "total_frames": len(self.accumulated_yolo_detections),
                    "frames_with_detections": frames_with_detections,
                    "total_detections": total_detections,
                    "unique_classes": len(class_counts),
                    "class_counts": class_counts
                },
                "detections_by_frame": self.accumulated_yolo_detections
            }
            
            with open(yolo_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        else:  # csv
            yolo_path = self.final_output_dir / f"{self.output_name}_yolo_detections.csv"
            with open(yolo_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step_nr', 'timestamp', 'class_name', 'bbox_x1', 'bbox_y1', 
                               'bbox_x2', 'bbox_y2', 'confidence', 'class_id', 'detection_id'])
                
                for frame_data in self.accumulated_yolo_detections:
                    step_nr = frame_data["step_nr"]
                    timestamp = frame_data["timestamp"]
                    detections = frame_data.get("detections", {})
                    
                    if not detections:
                        # Write a row with no detections
                        writer.writerow([step_nr, timestamp, '', '', '', '', '', '', '', ''])
                    else:
                        for class_name, objects in detections.items():
                            for obj in objects:
                                bbox = obj.get("bbox", [0, 0, 0, 0])
                                row = [
                                    step_nr, timestamp, class_name,
                                    bbox[0], bbox[1], bbox[2], bbox[3],
                                    obj.get("confidence", 0.0),
                                    obj.get("class_id", -1),
                                    obj.get("detection_id", -1)
                                ]
                                writer.writerow(row)
        
        return yolo_path
    
    def _save_segmentation_labels(self) -> pathlib.Path:
        """Save segmentation labels data linking segment IDs to class names."""
        if self.analysis_format == 'json':
            labels_path = self.final_output_dir / f"{self.output_name}_segmentation_labels.json"
            
            # Create comprehensive segmentation labels data
            all_classes = set()
            frames_with_labels = 0
            total_segments = 0
            
            for frame_data in self.accumulated_segmentation_labels:
                labels = frame_data.get("segmentation_labels", {})
                if labels:
                    frames_with_labels += 1
                    total_segments += len(labels)
                    all_classes.update(labels.values())
            
            data = {
                "summary": {
                    "total_frames": len(self.accumulated_segmentation_labels),
                    "frames_with_labels": frames_with_labels,
                    "total_segments": total_segments,
                    "unique_classes": sorted(list(all_classes)),
                    "class_distribution": self._calculate_class_distribution()
                },
                "labels_by_frame": self.accumulated_segmentation_labels
            }
            
            with open(labels_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        else:  # csv
            labels_path = self.final_output_dir / f"{self.output_name}_segmentation_labels.csv"
            with open(labels_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step_nr', 'timestamp', 'segment_id', 'class_label'])
                
                for frame_data in self.accumulated_segmentation_labels:
                    step_nr = frame_data["step_nr"]
                    timestamp = frame_data["timestamp"]
                    labels = frame_data.get("segmentation_labels", {})
                    
                    if not labels:
                        # Write a row indicating no labels for this frame
                        writer.writerow([step_nr, timestamp, '', ''])
                    else:
                        for segment_id, class_label in labels.items():
                            writer.writerow([step_nr, timestamp, segment_id, class_label])
        
        return labels_path
    
    def _calculate_class_distribution(self) -> Dict[str, int]:
        """Calculate the distribution of classes across all frames."""
        class_counts = {}
        
        for frame_data in self.accumulated_segmentation_labels:
            labels = frame_data.get("segmentation_labels", {})
            for class_label in labels.values():
                class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        return class_counts
    
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
                "save_object_mapping": self.save_object_mapping,
                "save_yolo_detections": self.save_yolo_detections,
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
                "n_closest_points_3d_count": len(self.accumulated_n_closest_points_distances),
                "n_closest_points_with_segment_ids_count": sum(1 for segment_ids in self.accumulated_n_closest_points_segment_ids if segment_ids is not None),
                "n_closest_points_with_class_labels_count": sum(1 for class_labels in self.accumulated_n_closest_points_class_labels if class_labels is not None) if hasattr(self, 'accumulated_n_closest_points_class_labels') else 0,
                "yolo_detections_count": len(self.accumulated_yolo_detections),
                "segmentation_labels_count": len(self.accumulated_segmentation_labels),
                "n_closest_points_3d_count": len(self.accumulated_n_closest_points_distances),
                "has_object_mapping": self.object_mapping is not None,
                "total_objects": len(self.object_mapping) if self.object_mapping is not None else 0
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
        
        if len(self.accumulated_yolo_detections) > 0:
            metadata["yolo_analysis"] = self._calculate_yolo_stats()
        
        if len(self.accumulated_segmentation_labels) > 0:
            metadata["segmentation_analysis"] = self._calculate_segmentation_stats()
        
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
    
    def _calculate_yolo_stats(self) -> Dict[str, Any]:
        """Calculate statistics from YOLO detection data."""
        total_detections = 0
        class_counts = {}
        frames_with_detections = 0
        confidence_scores = []
        detections_per_frame = []
        
        for frame_data in self.accumulated_yolo_detections:
            detections = frame_data.get("detections", {})
            frame_detection_count = 0
            
            if detections:
                frames_with_detections += 1
                for class_name, objects in detections.items():
                    object_count = len(objects)
                    total_detections += object_count
                    frame_detection_count += object_count
                    class_counts[class_name] = class_counts.get(class_name, 0) + object_count
                    
                    # Collect confidence scores
                    for obj in objects:
                        if "confidence" in obj:
                            confidence_scores.append(obj["confidence"])
            
            detections_per_frame.append(frame_detection_count)
        
        stats = {
            "total_frames": len(self.accumulated_yolo_detections),
            "frames_with_detections": frames_with_detections,
            "detection_rate": frames_with_detections / len(self.accumulated_yolo_detections) if self.accumulated_yolo_detections else 0,
            "total_detections": total_detections,
            "avg_detections_per_frame": np.mean(detections_per_frame) if detections_per_frame else 0,
            "max_detections_per_frame": max(detections_per_frame) if detections_per_frame else 0,
            "unique_classes": len(class_counts),
            "class_counts": class_counts
        }
        
        if confidence_scores:
            stats["confidence_stats"] = {
                "mean_confidence": float(np.mean(confidence_scores)),
                "min_confidence": float(np.min(confidence_scores)),
                "max_confidence": float(np.max(confidence_scores)),
                "std_confidence": float(np.std(confidence_scores))
            }
        
        return stats
    
    def _calculate_segmentation_stats(self) -> Dict[str, Any]:
        """Calculate statistics from segmentation labels data."""
        total_segments = 0
        frames_with_labels = 0
        class_counts = {}
        segments_per_frame = []
        all_classes = set()
        
        for frame_data in self.accumulated_segmentation_labels:
            labels = frame_data.get("segmentation_labels", {})
            frame_segment_count = len(labels)
            
            if frame_segment_count > 0:
                frames_with_labels += 1
                total_segments += frame_segment_count
                segments_per_frame.append(frame_segment_count)
                
                for class_label in labels.values():
                    class_counts[class_label] = class_counts.get(class_label, 0) + 1
                    all_classes.add(class_label)
        
        stats = {
            "total_frames": len(self.accumulated_segmentation_labels),
            "frames_with_labels": frames_with_labels,
            "labeling_rate": frames_with_labels / len(self.accumulated_segmentation_labels) if self.accumulated_segmentation_labels else 0,
            "total_segments": total_segments,
            "avg_segments_per_frame": np.mean(segments_per_frame) if segments_per_frame else 0,
            "max_segments_per_frame": max(segments_per_frame) if segments_per_frame else 0,
            "min_segments_per_frame": min(segments_per_frame) if segments_per_frame else 0,
            "unique_classes": len(all_classes),
            "class_list": sorted(list(all_classes)),
            "class_distribution": class_counts
        }
        
        return stats