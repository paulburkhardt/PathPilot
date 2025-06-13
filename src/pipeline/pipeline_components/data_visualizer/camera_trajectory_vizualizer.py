from typing import List, Dict, Any, Optional
import numpy as np
import rerun as rr
from .abstract_rerun_data_vizualizer import AbstractRerunDataVisualizer


class CameraTrajectoryVisualizer(AbstractRerunDataVisualizer):
    """
    Visualizer component for camera trajectories with closest point analysis.
    
    Args:
        show_trajectory_path: Show complete camera trajectory as path (default: True)
        show_view_cones: Show camera view cones if available (default: False)
        cone_length_factor: Factor for view cone length relative to distance (default: 1.5)
        max_cone_length: Maximum length for view cones in meters (default: 2.0)
        show_distance_lines: Show lines from camera to closest points (default: True)
        show_floor_analysis: Show floor-projected analysis if available (default: True)
        
    Returns:
        Empty dictionary as this is a visualization component
        
    Raises:
        ValueError: If required trajectory data is missing
    """
    
    def __init__(self, show_trajectory_path: bool = True, show_view_cones: bool = False,
                 cone_length_factor: float = 1.5, max_cone_length: float = 2.0,
                 show_distance_lines: bool = True, show_floor_analysis: bool = True) -> None:
        super().__init__()
        self.show_trajectory_path = show_trajectory_path
        self.show_view_cones = show_view_cones
        self.cone_length_factor = cone_length_factor
        self.max_cone_length = max_cone_length
        self.show_distance_lines = show_distance_lines
        self.show_floor_analysis = show_floor_analysis

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires camera and analysis data."""
        inputs = [
            "camera_positions", "camera_quaternions", "timestamps",
            "closest_point_3d", "distance_3d", "distances_array"
        ]
        
        if self.show_floor_analysis:
            inputs.extend(["closest_point_floor", "distance_floor", "floor_distances_array"])
            
        if self.show_view_cones:
            inputs.append("view_cone_mask")
            
        return inputs

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs visualizations only."""
        return []

    def _run(self, camera_positions: np.ndarray, camera_quaternions: np.ndarray,
             timestamps: np.ndarray, closest_point_3d: np.ndarray, distance_3d: np.ndarray,
             distances_array: np.ndarray,
             closest_point_floor: Optional[np.ndarray] = None,
             distance_floor: Optional[np.ndarray] = None,
             floor_distances_array: Optional[np.ndarray] = None,
             view_cone_mask: Optional[List] = None,
             **kwargs: Any) -> Dict[str, Any]:
        """
        Visualize camera trajectory with closest point analysis.
        
        Args:
            camera_positions: Nx3 array of camera positions
            camera_quaternions: Nx4 array of camera quaternions [x,y,z,w]
            timestamps: N array of timestamps
            closest_point_3d: Current or array of closest 3D points
            distance_3d: Current or array of 3D distances
            distances_array: Array of all 3D distances for plotting
            closest_point_floor: Floor-projected closest points (optional)
            distance_floor: Floor distances (optional)
            floor_distances_array: Array of all floor distances (optional)
            view_cone_mask: View cone masks for each pose (optional)
            **kwargs: Additional arguments
            
        Returns:
            Empty dictionary
        """
        print("Visualizing camera trajectory and closest point analysis...")
        
        # Log static trajectory path
        if self.show_trajectory_path:
            self._log_trajectory_path(camera_positions)
        
        # Log temporal camera poses and analysis
        self._log_temporal_analysis(
            camera_positions, camera_quaternions, timestamps,
            closest_point_3d, distance_3d, distances_array,
            closest_point_floor, distance_floor, floor_distances_array,
            view_cone_mask
        )
        
        # Log summary statistics
        self._log_summary_statistics(distances_array, floor_distances_array, timestamps)
        
        return {}

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

    def _log_trajectory_path(self, camera_positions: np.ndarray) -> None:
        """Log the complete camera trajectory as a static path."""
        rr.log("world/camera_trajectory", rr.LineStrips3D(
            strips=[camera_positions],
            colors=[255, 100, 100],  # Light red for trajectory
            radii=[0.01]
        ), static=True)
        
        print(f"Logged camera trajectory with {len(camera_positions)} poses")

    def _log_temporal_analysis(self, camera_positions: np.ndarray, camera_quaternions: np.ndarray,
                              timestamps: np.ndarray, closest_point_3d: np.ndarray,
                              distance_3d: np.ndarray, distances_array: np.ndarray,
                              closest_point_floor: Optional[np.ndarray] = None,
                              distance_floor: Optional[np.ndarray] = None,
                              floor_distances_array: Optional[np.ndarray] = None,
                              view_cone_mask: Optional[List] = None) -> None:
        """Log temporal camera poses and analysis data."""
        
        # Ensure we have arrays for iteration
        if closest_point_3d.ndim == 1:
            closest_points_3d = np.tile(closest_point_3d, (len(camera_positions), 1))
        else:
            closest_points_3d = closest_point_3d
            
        if np.isscalar(distance_3d):
            distances_3d = np.full(len(camera_positions), distance_3d)
        else:
            distances_3d = distance_3d
        
        # Handle floor data
        if closest_point_floor is not None:
            if closest_point_floor.ndim == 1:
                closest_points_floor = np.tile(closest_point_floor, (len(camera_positions), 1))
            else:
                closest_points_floor = closest_point_floor
        else:
            closest_points_floor = None
            
        if distance_floor is not None:
            if np.isscalar(distance_floor):
                distances_floor = np.full(len(camera_positions), distance_floor)
            else:
                distances_floor = distance_floor
        else:
            distances_floor = None

        # Log each camera pose with timeline
        for i, (timestamp, position, quaternion) in enumerate(
            zip(timestamps, camera_positions, camera_quaternions)
        ):
            # Set timeline
            rr.set_time_seconds("timestamp", timestamp)
            rr.set_time_sequence("frame", i)
            
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
                radii=[0.02]
            ))
            
            # Log closest point analysis
            closest_point = closest_points_3d[i]
            distance = distances_3d[i]
            
            # Log 3D closest point
            rr.log("world/closest_point_3d", rr.Points3D(
                positions=closest_point.reshape(1, 3),
                colors=[0, 255, 0],  # Green for closest point
                radii=[0.03]
            ))
            
            # Log distance line
            if self.show_distance_lines:
                rr.log("world/distance_line_3d", rr.LineStrips3D(
                    strips=[np.array([position, closest_point])],
                    colors=[255, 255, 0],  # Yellow line
                    radii=[0.005]
                ))
            
            # Log 3D distance plot
            rr.log("plots/distance_3d", rr.Scalars(scalars=[distance]))
            
            # Log floor analysis if available
            if (self.show_floor_analysis and closest_points_floor is not None and 
                distances_floor is not None):
                
                floor_point = closest_points_floor[i]
                floor_dist = distances_floor[i]
                
                # Log floor projected point
                rr.log("world/closest_point_floor", rr.Points3D(
                    positions=floor_point.reshape(1, 3),
                    colors=[255, 150, 0],  # Orange for floor-projected point
                    radii=[0.025]
                ))
                
                # Log floor distance line
                if self.show_distance_lines:
                    rr.log("world/distance_line_floor", rr.LineStrips3D(
                        strips=[np.array([position, floor_point])],
                        colors=[255, 150, 0],  # Orange line
                        radii=[0.007]
                    ))
                
                # Log floor distance plot
                rr.log("plots/distance_floor", rr.Scalars(scalars=[floor_dist]))
                
                # Log combined distance text
                rr.log("world/distance_text", rr.TextDocument(
                    f"3D: {distance:.3f}m | Floor: {floor_dist:.3f}m"
                ))
            else:
                # Log only 3D distance text
                rr.log("world/distance_text", rr.TextDocument(
                    f"Distance: {distance:.3f}m"
                ))
            
            # Log view cone if enabled and available
            if (self.show_view_cones and view_cone_mask is not None and 
                i < len(view_cone_mask) and view_cone_mask[i] is not None):
                self._log_view_cone(position, quaternion, distance)

    def _log_view_cone(self, camera_position: np.ndarray, camera_quaternion: np.ndarray,
                      distance: float) -> None:
        """Log camera view cone visualization."""
        R = self._quaternion_to_rotation_matrix(camera_quaternion)
        forward = R[:, 2]  # Camera forward direction
        
        # Create cone parameters
        cone_length = min(distance * self.cone_length_factor, self.max_cone_length)
        apex = camera_position
        
        # Create cone base circle
        angle_rad = np.radians(90.0)  # Default 90 degree cone
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

    def _log_summary_statistics(self, distances_array: np.ndarray,
                               floor_distances_array: Optional[np.ndarray],
                               timestamps: np.ndarray) -> None:
        """Log summary statistics as scalar plots."""
        print("\n=== TRAJECTORY SUMMARY STATISTICS ===")
        print(f"Total trajectory duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Number of poses: {len(timestamps)}")
        print(f"Average 3D distance to closest point: {distances_array.mean():.3f}m")
        print(f"Minimum 3D distance: {distances_array.min():.3f}m")
        print(f"Maximum 3D distance: {distances_array.max():.3f}m")
        print(f"Standard deviation 3D: {distances_array.std():.3f}m")
        
        # Log 3D statistics
        rr.log("stats/trajectory_duration", rr.Scalars(scalars=[timestamps[-1] - timestamps[0]]))
        rr.log("stats/average_distance_3d", rr.Scalars(scalars=[distances_array.mean()]))
        rr.log("stats/min_distance_3d", rr.Scalars(scalars=[distances_array.min()]))
        rr.log("stats/max_distance_3d", rr.Scalars(scalars=[distances_array.max()]))
        
        if floor_distances_array is not None and len(floor_distances_array) > 0:
            print(f"\n=== FLOOR DISTANCE STATISTICS ===")
            print(f"Average floor distance: {floor_distances_array.mean():.3f}m")
            print(f"Minimum floor distance: {floor_distances_array.min():.3f}m")
            print(f"Maximum floor distance: {floor_distances_array.max():.3f}m")
            print(f"Standard deviation floor: {floor_distances_array.std():.3f}m")
            
            # Log floor statistics
            rr.log("stats/average_distance_floor", rr.Scalars(scalars=[floor_distances_array.mean()]))
            rr.log("stats/min_distance_floor", rr.Scalars(scalars=[floor_distances_array.min()]))
            rr.log("stats/max_distance_floor", rr.Scalars(scalars=[floor_distances_array.max()])) 