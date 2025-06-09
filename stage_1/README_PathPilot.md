# PathPilot: Mast3r-Slam Output Processing Pipeline

PathPilot is a Python pipeline that processes Mast3r-Slam outputs to create interactive 3D visualizations using Rerun. It analyzes camera trajectories and finds the closest points in the reconstructed point cloud for each camera pose.

## Features

- 🎯 **Closest Point Analysis**: Finds the nearest point in the point cloud for each camera position
- 📊 **Distance Tracking**: Tracks and visualizes distances over time
- 🎥 **Interactive Timeline**: Step through the camera trajectory frame by frame
- 🌈 **Rich Visualization**: Shows point clouds, camera paths, closest points, and connecting lines
- 📈 **Statistical Analysis**: Provides summary statistics about the trajectory
- 💾 **Rerun Export**: Saves results as .rrd files for sharing and later viewing
- 🔍 **View Cone Filtering**: Optional filtering to only consider points within camera's field of view
- 🏠 **Floor Detection**: Automatic floor plane detection and coordinate system alignment
- 🎨 **Floor Visualization**: Visual highlighting of detected floor points with grid overlay

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Required Python Packages**:
   - `numpy` - Numerical computations
   - `scipy` - Spatial data structures (KD-tree)
   - `plyfile` - PLY point cloud file reading
   - `rerun-sdk` - 3D visualization and recording

## Input Data Format

PathPilot expects two files from Mast3r-Slam:

### 1. Point Cloud File (`.ply`)
- **Format**: PLY (Polygon File Format)
- **Content**: 3D points with optional RGB colors
- **Structure**: 
  ```
  x, y, z, red, green, blue
  ```

### 2. Camera Trajectory File (`.txt`)
- **Format**: Space-separated text file
- **Content**: Camera poses over time
- **Structure**: 
  ```
  timestamp x y z qx qy qz qw
  ```
  Where:
  - `timestamp`: Time in seconds
  - `x, y, z`: Camera position in world coordinates
  - `qx, qy, qz, qw`: Camera orientation as quaternion

## Usage

### Basic Usage

```bash
# Basic usage
python process_slam_output.py \
    --ply plots/one_chair/one_chair.ply \
    --trajectory plots/one_chair/one_chair.txt \
    --output one_chair_result.rrd

# With advanced features
python process_slam_output.py \
    --ply plots/one_chair/one_chair.ply \
    --trajectory plots/one_chair/one_chair.txt \
    --output one_chair_advanced.rrd \
    --use-view-cone \
    --cone-angle 60 \
    --max-view-distance 5 \
    --align-floor \
    --floor-threshold 0.05
```

### Command Line Arguments

#### Required Arguments
- `--ply, -p`: Path to PLY point cloud file (required)
- `--trajectory, -t`: Path to trajectory TXT file (required)

#### Output Options
- `--output, -o`: Output RRD file path (default: `pathpilot_output.rrd`)
- `--name, -n`: Recording name for Rerun (default: `PathPilot`)

#### Advanced Features
- `--use-view-cone`: Enable view cone filtering (only consider points within camera's field of view)
- `--cone-angle`: Half-angle of view cone in degrees (default: 90°)
- `--max-view-distance`: Maximum distance for view cone filtering in meters (default: 10m)
- `--align-floor`: Enable automatic floor detection and coordinate system alignment
- `--floor-threshold`: Distance threshold for floor plane detection in meters (default: 0.05m)

### Demo Script

For the existing `one_chair` dataset:

```bash
python demo_pathpilot.py
```

This will automatically process the `one_chair` data if available.

## Output

The pipeline generates:

1. **Interactive Rerun Visualization** (`.rrd` file) containing:
   - 3D point cloud with colors
   - Camera trajectory path (red points)
   - Closest points at each pose (green points)
   - Distance lines (yellow) connecting camera to closest points
   - Distance plot over time
   - Summary statistics

2. **Console Output** with statistics:
   - Total trajectory duration
   - Number of camera poses
   - Distance statistics (min, max, average, std dev)

## Visualization Elements

### 3D Scene
- **Point Cloud**: Original reconstruction from Mast3r-Slam with optional color information
- **Camera Path**: Red spheres showing camera positions with connecting trajectory line
- **Closest Points**: Green spheres highlighting nearest points to each camera
- **Distance Lines**: Yellow lines connecting cameras to closest points
- **Distance Text**: Real-time distance annotations at midpoints
- **Floor Visualization** (when enabled):
  - **Floor Points**: Bright green highlighting of detected floor points
  - **Floor Grid**: Yellow grid lines showing the floor plane
  - **Separate Layer**: Dedicated floor points layer for better visibility

### View Cone Visualization (when enabled)
- **Cone Wireframe**: White wireframe showing camera's field of view
- **Adaptive Sizing**: Cone length adapts to distance to closest point
- **Multiple Poses**: Individual cones for each camera position in timeline

### Timeline Controls
- **Frame Sequence**: Step through each camera pose with smooth transitions
- **Timestamp**: Navigate by actual time values from trajectory data
- **Distance Plot**: Interactive plot showing distance over time with hover details

### Statistics Panel
- **Trajectory Metrics**: Duration, pose count, movement patterns
- **Distance Analysis**: Average, min, max distances with standard deviation
- **Floor Detection Info** (when enabled): Horizontality score, camera support metrics

## Example Workflow

1. **Run Mast3r-Slam** on your video to generate:
   - `scene.ply` (point cloud)
   - `scene.txt` (trajectory)

2. **Process with PathPilot**:
   ```bash
   python process_slam_output.py -p scene.ply -t scene.txt -o result.rrd
   ```

3. **View Results**:
   ```bash
   rerun result.rrd
   ```

## Technical Details

### Algorithm Overview
1. **Data Loading**: Parse PLY and trajectory files with robust error handling
2. **Floor Detection** (optional): Automatic detection and alignment with floor plane
3. **Spatial Indexing**: Build KD-tree for efficient nearest neighbor search
4. **Distance Computation**: Find closest point for each camera pose (global or view-cone filtered)
5. **Visualization**: Create interactive Rerun recording with timeline and rich annotations

### Floor Detection Algorithm

The floor detection system uses a sophisticated approach to identify and align with the dominant horizontal plane:

#### Design Philosophy
The goal is to find the largest horizontal plane that physically supports the camera trajectory. This is typically the floor or ground plane that the camera moves above.

#### Algorithm Steps

1. **Gravity Direction Estimation**
   - Analyzes camera orientations from the first several poses
   - Extracts "down" vectors from camera coordinate frames
   - Averages these to estimate the gravity direction
   - Floor normal should be opposite to gravity (pointing upward)

2. **Candidate Point Selection**
   - Identifies points below the camera trajectory level
   - Tries both Z-axis and Y-axis as height coordinates (handles different conventions)
   - Selects the coordinate system that provides more floor candidates

3. **RANSAC Plane Fitting with Smart Scoring**
   - Samples random point triplets to define candidate planes
   - Evaluates each plane using multiple criteria:
     - **Horizontality**: How perpendicular the plane is to gravity (should be ~1.0)
     - **Inlier Count**: Number of points fitting the plane within threshold
     - **Camera Support**: Whether cameras are positioned above the plane
   - Combined scoring: `inliers × horizontality × 10 + horizontality × 1000 + camera_support × 500`

4. **Validation and Fallback**
   - Ensures detected plane is reasonably horizontal (>0.7 alignment with expected normal)
   - Verifies that cameras are positioned above the detected floor
   - Fallback: Creates horizontal plane below lowest camera if detection fails

#### Why This Approach Works
- **Physical Constraints**: Floors support cameras from below and are horizontal
- **Multi-Criteria Scoring**: Prevents selection of walls or ceiling
- **Robust to Coordinate Systems**: Works with both Z-up and Y-up conventions
- **Gravity-Aware**: Uses camera orientation to understand spatial relationships

### View Cone Filtering

When enabled, the system only considers points within the camera's field of view:

#### Implementation
1. **Camera Forward Direction**: Extracted from rotation matrix (Z-axis)
2. **Cone Geometry**: Defined by half-angle and maximum distance
3. **Point Filtering**: Uses dot product to test if points fall within cone
4. **Fallback**: Reverts to global search if no points found in view cone

#### Visualization
- White wireframe cones show the camera's field of view
- Adaptive cone length based on distance to closest point
- 8-point circular base with lines to apex

### Coordinate System Transformation

When floor alignment is enabled:

1. **Rotation Calculation**: Uses Rodrigues' formula to align floor normal with Z-axis
2. **Translation**: Moves floor plane to Z=0 level
3. **Consistent Transformation**: Applies same transformation to points, cameras, and quaternions
4. **Visualization Benefits**: Creates intuitive "floor = XY plane, height = Z" coordinate system

### Performance Optimizations
- **KD-tree Search**: O(log n) per query for efficient closest point finding
- **Spatial Filtering**: Reduces search space for floor detection
- **Early Termination**: Stops RANSAC when excellent solution found
- **Memory Management**: Samples large point clouds to control memory usage

### Coordinate Systems
- **Input**: Flexible - works with various SLAM coordinate conventions
- **Processing**: Adapts to Z-up or Y-up based on data analysis
- **Output**: Right-handed coordinate system (Y-up, Z-forward) for Rerun
- **Quaternions**: Standard `[x, y, z, w]` format throughout

## Troubleshooting

### Common Issues

1. **File Not Found Error**:
   - Check that PLY and TXT files exist and paths are correct
   - Ensure you have read permissions for input files

2. **Rerun Viewer Not Opening**:
   - Install/update Rerun: `pip install --upgrade rerun-sdk`
   - Try manual launch: `rerun your_file.rrd`
   - Check if port 9876 is available

3. **Memory Issues with Large Point Clouds**:
   - Pipeline automatically samples large point clouds for floor detection
   - Monitor system memory during processing
   - Consider using `--floor-threshold` to reduce candidates

4. **Incorrect Trajectory Format**:
   - Pipeline handles malformed lines gracefully and reports skipped lines
   - Ensure TXT file has 8 columns: `timestamp x y z qx qy qz qw`
   - Check for proper space separation and numeric values

5. **Floor Detection Issues**:
   - Algorithm reports horizontality score (should be close to 1.0)
   - Check that cameras are above detected plane in output
   - Try adjusting `--floor-threshold` if detection is too strict/loose
   - Review gravity direction estimation in console output

6. **View Cone Filtering Problems**:
   - Ensure quaternions represent valid rotations
   - Check cone angle is reasonable (30-120 degrees typical)
   - Verify max distance covers relevant scene points

### Performance Tips

- **Large Datasets**: Pipeline automatically samples to control memory usage
- **Storage**: Use SSD for faster PLY file loading
- **Memory**: Close unnecessary applications during processing
- **Floor Detection**: Disable with fewer poses (<5) as estimation becomes unreliable
- **View Cones**: Use appropriate max distance to avoid empty cones

## Development History & Design Decisions

### Algorithm Evolution

The floor detection algorithm went through several iterations to achieve robustness:

1. **Initial Approach**: Simple "points below camera" filtering
   - **Problem**: Assumed specific coordinate system orientation
   - **Issue**: Could select ceiling or walls in rotated coordinate systems

2. **Movement-Based Detection**: Used camera movement direction to infer floor plane
   - **Assumption**: Camera moves parallel to floor
   - **Problem**: Cameras can move in any direction (upward, downward, diagonal)
   - **Issue**: Often detected walls as floor when camera moved toward walls

3. **Gravity-Based Detection** (Current): Uses camera orientation to estimate gravity
   - **Innovation**: Leverages camera "down" vectors to find true gravity direction
   - **Key Insight**: Floor must be horizontal (perpendicular to gravity) and support cameras
   - **Success**: Robust to coordinate systems and camera movement patterns

### Design Philosophy

#### Physical Constraints Drive Algorithm Design
- **Floor Properties**: Horizontal, supports objects from below, largest planar surface
- **Camera Properties**: Oriented with respect to gravity, positioned above ground
- **Scene Properties**: SLAM reconstructs 3D structure preserving spatial relationships

#### Multi-Criteria Optimization
Rather than single-metric optimization, the system balances:
- **Geometric Fit**: How well points fit the plane (inlier count)
- **Physical Plausibility**: Horizontality and camera support
- **Robustness**: Fallback mechanisms and validation

#### Coordinate System Agnosticism
- **Input Flexibility**: Works with Z-up, Y-up, or arbitrary orientations
- **Automatic Detection**: Analyzes data to determine best coordinate interpretation
- **Consistent Output**: Always produces right-handed Y-up system for visualization

## Future Enhancements

### Algorithmic Improvements
- [ ] Multi-plane floor detection for multi-level environments
- [ ] Adaptive thresholding based on scene scale
- [ ] Outlier rejection in trajectory data
- [ ] Surface normal estimation for terrain following

### Visualization Features
- [ ] Multiple trajectory comparison
- [ ] Advanced distance metrics (surface distance, directional distance)
- [ ] Camera frustum visualization with depth uncertainty
- [ ] Heat maps showing point density and quality

### Performance & Scalability
- [ ] Streaming processing for real-time applications
- [ ] GPU acceleration for large point clouds
- [ ] Hierarchical level-of-detail for massive datasets
- [ ] Distributed processing for multiple sequences

### File Format Support
- [ ] Multiple trajectory formats (TUM, KITTI, EuRoC)
- [ ] Point cloud formats (PCD, LAS, XYZ)
- [ ] Export to other visualization platforms
- [ ] Integration with ROS ecosystem

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is part of the PathPilot system for robotic navigation analysis. 