# SLAM Pipeline Output Visualizer

A standalone Python script for visualizing SLAM (Simultaneous Localization and Mapping) pipeline outputs using the rerun-sdk. This tool provides an interactive 3D visualization of point clouds, camera trajectories, floor detection, closest point analysis, and video synchronization.

## Features

### 🎯 Core Visualization
- **Point Cloud Rendering**: Static and temporal point cloud visualization with color support
- **Camera Trajectory**: 3D camera path with temporal playback
- **Floor Detection**: Floor plane visualization with grid overlay
- **Closest Point Analysis**: Real-time proximity warnings with distance visualization
- **Video Synchronization**: Synchronized video playback with trajectory data

### 🎨 Advanced Features
- **Segmentation Support**: Color point clouds by segmentation class IDs
- **Warning System**: Multi-level proximity alerts (Normal, Caution, Strong, Critical)
- **Directional Guidance**: Human-readable direction descriptions
- **View Cones**: Camera field-of-view visualization
- **Spatial Subsampling**: Memory-efficient point cloud rendering
- **Floor Highlighting**: Automatic floor point identification and highlighting

### 📊 Interactive Elements
- **Real-time Plots**: Distance and warning level charts
- **Color-coded Warnings**: Visual proximity alerts with appropriate colors
- **Temporal Playback**: Step-through trajectory with synchronized data
- **Separate Segmentation Views**: Individual class visualization options

## Installation

### Prerequisites
- Python 3.7+
- rerun-sdk
- numpy
- plyfile

### Install Dependencies
```bash
pip install rerun-sdk numpy plyfile
```

### Optional Dependencies
For enhanced point cloud processing:
```bash
pip install open3d
```

## Usage

### Basic Usage
```bash
python visualize_slam_output.py /path/to/slam_analysis_output_dir
```

### Full Usage
```bash
python visualize_slam_output.py --color_pointcloud_by_classIds --show_segmentation_masks_separately --show-yolo-detections /path/to/slam_analysis_output_dir
```




### Command Line Options

#### Visualization Toggles
```bash
--no-point-cloud          # Disable point cloud visualization
--no-trajectory          # Disable trajectory visualization
--no-floor              # Disable floor plane visualization
--no-closest-points     # Disable closest points analysis
--no-view-cones         # Disable view cone visualization
--no-trajectory-path    # Disable static trajectory path
--no-distance-lines     # Disable distance lines to closest points
--no-highlight-floor    # Disable floor point highlighting
--no-video              # Disable video visualization
```

#### Point Cloud Options
```bash
--color_pointcloud_by_classIds     # Enable coloring by class IDs
--show_segmentation_masks_separately  # Show segmentation classes separately
--subsample-percentage 0.5         # Percentage of points to keep (0.0-1.0)
--no-subsampling                   # Disable point cloud subsampling
```

#### Warning System
```bash
--warning-distance-critical 0.2    # Critical warning threshold (meters)
--warning-distance-strong 0.4      # Strong warning threshold (meters)
--warning-distance-caution 0.6     # Caution warning threshold (meters)
--no-warnings                      # Disable proximity warning system
--no-directional-warnings          # Disable directional warnings
```

#### Visualization Parameters
```bash
--floor-threshold 0.05             # Floor point identification threshold
--grid-size 2.0                    # Floor grid size (meters)
--cone-angle 45.0                  # View cone angle (degrees)
--cone-length-factor 1.5           # View cone length factor
--max-cone-length 2.0              # Maximum cone length (meters)
```

## Examples


### Complete Visualization
```bash
python visualize_slam_output.py --color_pointcloud_by_classIds --show_segmentation_masks_separately ./outputs 
```

### Basic Visualization
```bash
python visualize_slam_output.py ./enhanced_slam_outputs/slam_analysis_20240115_143045
```

### Minimal Visualization (Point Cloud Only)
```bash
python visualize_slam_output.py ./outputs --no-view-cones --no-floor --no-video
```

### Enhanced Point Cloud with Segmentation
```bash
python visualize_slam_output.py ./outputs --color_pointcloud_by_classIds --show_segmentation_masks_separately
```

### Custom Warning Thresholds
```bash
python visualize_slam_output.py ./outputs --warning-distance-critical 0.15 --warning-distance-caution 0.8
```

### Memory-Optimized for Large Datasets
```bash
python visualize_slam_output.py ./outputs --subsample-percentage 0.8
```

### Disable Warning System
```bash
python visualize_slam_output.py ./outputs --no-warnings
```

## Data Format Support

### Input Files
The script automatically detects and loads the following file types:

#### Point Clouds
- `slam_analysis_pointcloud.ply`
- `incremental_analysis_detailed_pointcloud.ply`
- Intermediate step point clouds: `intermediate/step_*/pointcloud.ply`

#### Trajectory Data
- `slam_analysis_trajectory.txt`
- `incremental_analysis_detailed_trajectory.txt`

#### Floor Detection
- `slam_analysis_floor_data.json` / `.csv`
- `incremental_analysis_detailed_floor_data.json` / `.csv`

#### Closest Points Analysis
- `slam_analysis_closest_points.json` / `.csv`
- `incremental_analysis_detailed_closest_points.json` / `.csv`

#### Video
- Automatically loaded from pipeline configuration
- Supports rerun-compatible MP4 files

### Metadata
- `metadata.json`: Pipeline configuration and file paths

## Warning System

The visualizer includes a sophisticated proximity warning system with four levels:

| Level | Distance | Color | Description |
|-------|----------|-------|-------------|
| Normal | >0.6m | Green | Safe distance |
| Caution | 0.4-0.6m | Yellow | Moderate proximity |
| Strong | 0.2-0.4m | Orange | Close proximity |
| Critical | <0.2m | Red | Immediate danger |

### Directional Warnings
When enabled, the system provides human-readable direction descriptions:
- "Object 0.3m ahead-right"
- "Object 0.15m to your left"
- "Object 0.25m behind-below"

## Performance Optimization

### Point Cloud Subsampling
- **Percentage-based**: Configurable reduction (default: 50%)
- **Final step preservation**: Always shows 100% of points in final temporal step
- **Spatial consistency**: Maintains point distribution across space
- **Memory management**: Automatic subsampling for large datasets

### Temporal Data Handling
- **Step mapping**: Intelligent mapping between trajectory poses and point cloud steps
- **Video synchronization**: Step-based frame mapping for smooth playback
- **Memory efficiency**: Temporal point clouds only show current frame

## Troubleshooting

### Common Issues

#### Rerun Connection Failed
```bash
# Start rerun viewer manually
rerun --serve-web
# Then run the script
python visualize_slam_output.py ./outputs
```

#### Video Not Playing
```bash
# Convert video for rerun compatibility
python convert_videos_for_rerun.py Data/Videos/
```

#### Memory Issues with Large Datasets
```bash
# Reduce subsampling percentage
python visualize_slam_output.py ./outputs --subsample-percentage 0.3
# Or disable subsampling entirely
python visualize_slam_output.py ./outputs --no-subsampling
```

#### Missing Data Files
- Check that the output directory contains the expected SLAM analysis files
- Verify file naming conventions match the supported patterns
- Ensure metadata.json is present for automatic configuration

### Performance Tips
- Use `--subsample-percentage` for large point clouds
- Disable unused visualizations with `--no-*` flags
- Enable `--color_pointcloud_by_classIds` for better segmentation visualization
- Use `--show_segmentation_masks_separately` for detailed class analysis

## Configuration

### Pipeline Metadata Integration
The script automatically reads pipeline configuration from `metadata.json`:
- View cone settings from `IncrementalClosestPointFinderComponent`
- Video paths from `MAST3RSLAMVideoDataset`
- File paths and analysis parameters

### Default Settings
- Floor threshold: 0.05m
- Grid size: 2.0m
- View cone angle: 45°
- Subsampling: 50% for temporal steps
- Warning distances: 0.2m, 0.4m, 0.6m

## Output

The visualizer creates an interactive 3D scene in the rerun web viewer with:
- **World coordinate system**: All spatial data in 3D space
- **Timeline controls**: Playback controls for temporal data
- **Scene tree**: Organized visualization hierarchy
- **Real-time plots**: Distance and warning level charts
- **Text overlays**: Warning messages and status information
