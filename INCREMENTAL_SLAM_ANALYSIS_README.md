# Incremental SLAM Analysis Pipeline

This document describes the restructured pipeline that combines SLAM processing with real-time floor detection and closest point analysis into a unified workflow.

## Overview

The pipeline has been restructured into two main phases:

1. **Phase 1**: Incremental SLAM Analysis - Processes video frame-by-frame, building point clouds, detecting floors, and analyzing closest points in real-time
2. **Phase 2**: Visualization - Loads the comprehensive analysis results for interactive visualization

## Phase 1: Incremental SLAM Analysis

### Pipeline Components

The Phase 1 pipeline follows this execution order:

```
MAST3RSLAMVideoDataLoader → MAST3RSLAMComponent → IncrementalFloorDetectionComponent → IncrementalClosestPointFinderComponent → EnhancedSLAMOutputWriter
```

#### 1. MAST3RSLAMVideoDataLoader
- Loads video frames for SLAM processing
- Provides image data, timestamps, and camera calibration

#### 2. MAST3RSLAMComponent  
- Processes frames with MAST3R SLAM algorithm
- Builds accumulated point cloud (or refreshing mode)
- Provides camera poses and accumulated 3D reconstruction

#### 3. IncrementalFloorDetectionComponent
- **Waits for 3 frames** before starting floor detection
- Uses camera poses and accumulated point cloud to detect floor plane
- **Optional refinement**: Can refine floor plane every N frames for better accuracy
- Outputs floor normal, offset, and floor points

#### 4. IncrementalClosestPointFinderComponent
- **Waits for floor detection** to be available before starting
- Finds closest point to **current camera position** in accumulated point cloud
- Calculates both 3D distances and floor-projected distances
- Optional view cone filtering for realistic visibility constraints

#### 5. EnhancedSLAMOutputWriter
- Saves comprehensive analysis results including:
  - Final accumulated point cloud (PLY format)
  - Complete camera trajectory (TXT format)
  - Floor detection data (JSON/CSV format)
  - Closest point analysis for each frame (JSON/CSV format)
  - Optional intermediate results every N frames
  - Comprehensive metadata

### Configuration Options

#### Basic Configuration (configs/phase_1_incremental_slam_analysis.yaml)
```yaml
pipeline:
  components:
    - type: MAST3RSLAMVideoDataLoader
      config:
        video_path: Data/Videos/chair_trash_bag.mp4
        mast3r_slam_config_path: "configs/MASt3R-SLAM_configs/calib.yaml"
    
    - type: MAST3RSLAMComponent
      config:
        point_cloud_method: accumulating
        c_confidence_threshold: 1.0 
        mast3r_slam_config_path: "configs/MASt3R-SLAM_configs/calib.yaml"
    
    - type: IncrementalFloorDetectionComponent
      config:
        min_frames: 3  # Wait for 3 frames before starting
        refine_interval: 20  # Refine every 20 frames (0 = no refinement)
        max_refinement_poses: 20  # Use last 20 poses for refinement
    
    - type: IncrementalClosestPointFinderComponent
      config:
        use_floor_distance: true
        wait_for_floor: true  # Wait for floor detection before starting
    
    - type: EnhancedSLAMOutputWriter
      config:
        output_dir: "enhanced_slam_outputs"
        save_intermediate: false  # Set to true for detailed analysis
        analysis_format: "json"
```

#### Detailed Configuration with Intermediate Saving
For more detailed analysis, use `configs/phase_1_incremental_slam_analysis_with_intermediate.yaml`:
- Saves intermediate results every 5 frames
- More frequent floor refinement (every 10 frames)
- Higher sampling ratio for floor detection

### Output Structure

Phase 1 creates a timestamped directory with comprehensive outputs:

```
enhanced_slam_outputs/
└── incremental_analysis_YYYYMMDD_HHMMSS/
    ├── incremental_analysis_pointcloud.ply      # Final point cloud
    ├── incremental_analysis_trajectory.txt      # Camera trajectory
    ├── incremental_analysis_floor_data.json     # Floor detection results
    ├── incremental_analysis_closest_points.json # Closest point analysis
    ├── metadata.json                            # Comprehensive metadata
    └── intermediate/                            # Optional intermediate results
        ├── step_000005/
        │   ├── pointcloud.ply
        │   └── trajectory.txt
        └── step_000010/
            ├── pointcloud.ply
            └── trajectory.txt
```

### Data Formats

#### Trajectory File (TXT)
```
timestamp x y z qx qy qz qw
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000
```

#### Floor Data (JSON)
```json
{
  "floor_normal": [0.0, 0.0, 1.0],
  "floor_offset": -1.2,
  "detection_step": 3,
  "timestamp": 0.1
}
```

#### Closest Points Analysis (JSON)
```json
{
  "closest_points_3d": [[x1, y1, z1], [x2, y2, z2], ...],
  "distances_3d": [d1, d2, d3, ...],
  "closest_points_floor": [[x1f, y1f, z1f], ...],
  "distances_floor": [df1, df2, df3, ...],
  "analysis_summary": {
    "avg_distance_3d": 1.234,
    "min_distance_3d": 0.567,
    "max_distance_3d": 2.890
  }
}
```

## Phase 2: Visualization

Phase 2 loads the comprehensive results from Phase 1 for interactive visualization.

### Pipeline Components

```
PLYPointCloudLoader → TrajectoryDataLoader → FloorDetectionComponent → ClosestPointFinderComponent → Visualizers
```

### Configuration (configs/phase_2_visualization_from_enhanced_outputs.yaml)

Update the timestamp in the file paths to match your Phase 1 output:

```yaml
pipeline:
  components:
    - type: PLYPointCloudLoader
      config:
        ply_path: "enhanced_slam_outputs/incremental_analysis_TIMESTAMP/incremental_analysis_pointcloud.ply"
    
    - type: TrajectoryDataLoader
      config:
        trajectory_path: "enhanced_slam_outputs/incremental_analysis_TIMESTAMP/incremental_analysis_trajectory.txt"
    
    # ... visualization components
```

### Rerun Visualization Features

The visualization phase provides:
- **Point cloud rendering** with floor highlighting
- **Camera trajectory visualization** as a continuous path
- **Temporal playback** of camera poses with closest point analysis
- **Floor grid visualization** and plane highlighting
- **Distance analysis plots** showing 3D and floor distances over time
- **Interactive exploration** with Rerun's 3D viewer

## Usage

### Running Phase 1

```bash
# Basic incremental analysis
python main.py --config-name=phase_1_incremental_slam_analysis

# Detailed analysis with intermediate saving
python main.py --config-name=phase_1_incremental_slam_analysis_with_intermediate
```

### Running Phase 2

1. Update the timestamp in the Phase 2 config file
2. Run visualization:

```bash
python main.py --config-name=phase_2_visualization_from_enhanced_outputs
```

## Key Features

### Incremental Processing
- **Frame-by-frame analysis**: Each frame updates the accumulated point cloud and analyzes the current camera position
- **Waiting mechanisms**: Floor detection waits for 3 frames, closest point analysis waits for floor detection
- **Progressive refinement**: Floor plane can be refined as more data becomes available

### Flexible Output Options
- **Comprehensive saving**: All analysis data saved in structured formats
- **Intermediate results**: Optional saving of point clouds and trajectories at regular intervals
- **Multiple formats**: JSON for complex data, CSV for tabular analysis, PLY for point clouds
- **Metadata tracking**: Complete record of configuration and analysis results

### Real-time Analysis
- **Current position focus**: Closest point analysis targets the current camera position
- **Accumulated point cloud**: Uses all points gathered so far for comprehensive spatial analysis
- **Floor-aware distances**: Horizontal distances calculated on the detected floor plane

### Environment Flexibility
- **Two-stage approach**: SLAM processing and visualization can run in different environments
- **File-based communication**: Rich intermediate files enable easy data transfer
- **Visualization replay**: Complete analysis can be replayed and explored interactively

## Performance Considerations

### Floor Detection
- Floor refinement is computationally light (reduced RANSAC iterations)
- Refinement can be disabled (`refine_interval: 0`) for maximum performance
- Uses limited pose history to prevent memory growth

### Closest Point Analysis
- KD-tree rebuilding per frame is the main computational cost
- View cone filtering can reduce search space for better performance
- Floor projections use efficient 2D spatial indexing

### Memory Management
- Accumulated point clouds grow with time - monitor memory usage for long sequences
- Pose history is limited to prevent unbounded growth
- Intermediate saving helps manage disk space vs. memory trade-offs

## Troubleshooting

### Common Issues

1. **Floor detection not starting**: Check that at least 3 frames are processed
2. **Closest point analysis not starting**: Ensure floor detection has succeeded
3. **Missing timestamp in Phase 2**: Update config file paths with actual timestamp from Phase 1 output
4. **Memory issues**: Consider using "refreshing" point cloud method or reducing confidence threshold
5. **Performance issues**: Disable floor refinement or reduce intermediate saving frequency

### Debugging Options

- Enable intermediate saving to inspect point cloud growth
- Check metadata.json for analysis statistics and configuration verification
- Use visualization phase to validate floor detection and closest point results
- Monitor console output for component status and timing information 