# SLAM Analysis Pipeline Components

This document describes the modular pipeline components created from the `process_slam_output.py` script functionality. These components provide a flexible, configurable way to analyze SLAM outputs with point clouds and camera trajectories.

## Overview

The original `process_slam_output.py` script has been decomposed into the following modular pipeline components:

1. **Data Loaders**: Load point clouds and camera trajectories
2. **Processing Components**: Floor detection and closest point analysis  
3. **Visualizers**: Point cloud and camera trajectory visualization with Rerun

## Components

### Data Loaders

#### PLYPointCloudLoader
Loads 3D point cloud data from PLY files.

**Configuration:**
- `ply_path`: Path to PLY file
- `load_colors`: Whether to load RGB colors if available (default: true)

**Outputs:** `point_cloud`

#### TrajectoryDataLoader  
Loads camera trajectory data from TXT files.

**Configuration:**
- `trajectory_path`: Path to trajectory TXT file (format: timestamp x y z qx qy qz qw)
- `validate_format`: Whether to validate trajectory format (default: true)

**Outputs:** `camera_positions`, `camera_quaternions`, `timestamps`

### Processing Components

#### FloorDetectionComponent
Detects floor planes using RANSAC with gravity estimation from camera poses.

**Configuration:**
- `sample_ratio`: Ratio of points to sample for floor detection (default: 0.1)
- `ransac_threshold`: Distance threshold for RANSAC inlier detection (default: 0.05)
- `min_inliers`: Minimum number of inliers required for a valid plane (default: 1000)
- `n_movement_poses`: Number of first poses to use for movement direction analysis (default: 10)
- `floor_threshold`: Distance threshold for identifying floor points (default: 0.05)

**Inputs:** `point_cloud`, `camera_positions`, `camera_quaternions`
**Outputs:** `floor_normal`, `floor_offset`, `floor_threshold`, `floor_points`

#### ClosestPointFinderComponent
Finds closest points in point cloud to camera positions with optional view cone filtering.

**Configuration:**
- `use_view_cone`: Enable view cone filtering (default: false)
- `cone_angle_deg`: Half-angle of view cone in degrees (default: 90.0)
- `max_view_distance`: Maximum distance for view cone filtering (default: 10.0)
- `use_floor_distance`: Calculate horizontal distances on floor plane (default: false)

**Inputs:** `point_cloud`, `camera_positions` + conditional inputs based on config
**Outputs:** `closest_point_3d`, `distance_3d`, `distances_array` + conditional outputs

### Visualizers

#### PointCloudDataVisualizer (Enhanced)
Enhanced point cloud visualizer with floor detection and highlighting.

**Configuration:**
- `enable_floor_visualization`: Enable floor plane visualization (default: true)
- `floor_threshold`: Distance threshold for floor point identification (default: 0.05)
- `grid_size`: Size of floor grid visualization in meters (default: 2.0)
- `highlight_floor_points`: Highlight floor points in different color (default: true)

**Inputs:** `point_cloud` + optional floor data
**Outputs:** None (visualization only)

#### CameraTrajectoryVisualizer
Visualizes camera trajectories with closest point analysis and optional view cones.

**Configuration:**
- `show_trajectory_path`: Show complete camera trajectory as path (default: true)
- `show_view_cones`: Show camera view cones if available (default: false)
- `cone_length_factor`: Factor for view cone length relative to distance (default: 1.5)
- `max_cone_length`: Maximum length for view cones in meters (default: 2.0)
- `show_distance_lines`: Show lines from camera to closest points (default: true)
- `show_floor_analysis`: Show floor-projected analysis if available (default: true)

**Inputs:** Camera trajectory and analysis data
**Outputs:** None (visualization only)

## Pipeline Configuration Examples

### Basic SLAM Analysis Pipeline

```yaml
pipeline:
  components:
    - type: PLYPointCloudLoader
      config:
        ply_path: "pointcloud.ply"
        load_colors: true
    
    - type: TrajectoryDataLoader
      config:
        trajectory_path: "trajectory.txt"
        validate_format: true
    
    - type: FloorDetectionComponent
      config:
        sample_ratio: 0.1
        ransac_threshold: 0.05
        floor_threshold: 0.05
    
    - type: ClosestPointFinderComponent
      config:
        use_view_cone: false
        use_floor_distance: true
    
    - type: PointCloudDataVisualizer
      config:
        enable_floor_visualization: true
        highlight_floor_points: true
    
    - type: CameraTrajectoryVisualizer
      config:
        show_trajectory_path: true
        show_floor_analysis: true
```

### Advanced Pipeline with View Cone Filtering

```yaml
pipeline:
  components:
    - type: PLYPointCloudLoader
      config:
        ply_path: "pointcloud.ply"
    
    - type: TrajectoryDataLoader
      config:
        trajectory_path: "trajectory.txt"
    
    - type: FloorDetectionComponent
      config:
        ransac_threshold: 0.05
    
    - type: ClosestPointFinderComponent
      config:
        use_view_cone: true
        cone_angle_deg: 90.0
        max_view_distance: 10.0
        use_floor_distance: true
    
    - type: PointCloudDataVisualizer
      config:
        enable_floor_visualization: true
    
    - type: CameraTrajectoryVisualizer
      config:
        show_view_cones: true
        show_floor_analysis: true
```

## Data Flow

The typical data flow through the pipeline:

1. **PLYPointCloudLoader** → loads point cloud data
2. **TrajectoryDataLoader** → loads camera trajectory  
3. **FloorDetectionComponent** → detects floor plane from point cloud + camera poses
4. **ClosestPointFinderComponent** → finds closest points with optional filtering
5. **PointCloudDataVisualizer** → visualizes point cloud with floor highlighting
6. **CameraTrajectoryVisualizer** → visualizes trajectory with analysis results

## Features

### Floor Detection
- **Gravity Estimation**: Uses camera orientations to estimate gravity direction
- **RANSAC**: Robust plane fitting with configurable parameters
- **Validation**: Ensures planes are horizontal and support camera positions
- **Visualization**: Floor points highlighted in green, grid overlay

### Closest Point Analysis  
- **3D Distance**: Standard Euclidean distance to closest points
- **View Cone Filtering**: Only consider points within camera's field of view
- **Floor Distance**: Horizontal distance on floor plane (useful for collision detection)
- **Temporal Analysis**: Distance tracking over time with plots

### Visualization
- **Point Cloud**: Color-coded points with floor highlighting
- **Camera Trajectory**: 3D path with pose indicators
- **Distance Lines**: Visual connections between cameras and closest points
- **View Cones**: Camera field of view visualization
- **Statistics**: Summary plots and metrics
- **Timeline**: Temporal playback with Rerun

## Benefits of Modular Design

1. **Flexibility**: Mix and match components based on needs
2. **Reusability**: Components can be used in different pipeline configurations
3. **Testability**: Each component can be tested independently
4. **Configurability**: Easy parameter adjustment without code changes
5. **Extensibility**: Easy to add new components or modify existing ones

## Dependencies

- **numpy**: Numerical computations
- **scipy**: KD-tree for closest point search  
- **plyfile**: PLY file reading
- **rerun**: 3D visualization
- **pathlib**: Path handling

## Usage

```python
from src.pipeline.pipeline_builder import PipelineBuilder
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="slam_analysis_config")
def main(cfg: DictConfig) -> None:
    pipeline = PipelineBuilder.build(cfg)
    pipeline.run()

if __name__ == "__main__":
    main()
```

This modular approach provides all the functionality of the original `process_slam_output.py` script while being more flexible, maintainable, and reusable within the PathPilot pipeline architecture. 