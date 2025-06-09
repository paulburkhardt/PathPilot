# PathPilot: Mast3r-Slam Output Processing Pipeline

PathPilot is a Python pipeline that processes Mast3r-Slam outputs to create interactive 3D visualizations using Rerun. It analyzes camera trajectories and finds the closest points in the reconstructed point cloud for each camera pose.

## Features

- 🎯 **Closest Point Analysis**: Finds the nearest point in the point cloud for each camera position
- 📊 **Distance Tracking**: Tracks and visualizes distances over time
- 🎥 **Interactive Timeline**: Step through the camera trajectory frame by frame
- 🌈 **Rich Visualization**: Shows point clouds, camera paths, closest points, and connecting lines
- 📈 **Statistical Analysis**: Provides summary statistics about the trajectory
- 💾 **Rerun Export**: Saves results as .rrd files for sharing and later viewing

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
python process_slam_output.py \
    --ply plots/one_chair/one_chair.ply \
    --trajectory plots/one_chair/one_chair.txt \
    --output one_chair_result.rrd
```

### Command Line Arguments

- `--ply, -p`: Path to PLY point cloud file (required)
- `--trajectory, -t`: Path to trajectory TXT file (required)
- `--output, -o`: Output RRD file path (default: `pathpilot_output.rrd`)
- `--name, -n`: Recording name for Rerun (default: `PathPilot`)

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
- **Point Cloud**: Original reconstruction from Mast3r-Slam
- **Camera Path**: Red spheres showing camera positions
- **Closest Points**: Green spheres highlighting nearest points
- **Distance Lines**: Yellow lines connecting cameras to closest points
- **Distance Text**: Real-time distance annotations

### Timeline Controls
- **Frame Sequence**: Step through each camera pose
- **Timestamp**: Navigate by actual time values
- **Distance Plot**: Interactive plot showing distance over time

### Statistics Panel
- Trajectory duration
- Average/min/max distances
- Total number of poses

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
1. **Data Loading**: Parse PLY and trajectory files
2. **Spatial Indexing**: Build KD-tree for efficient nearest neighbor search
3. **Distance Computation**: Find closest point for each camera pose
4. **Visualization**: Create interactive Rerun recording with timeline

### Performance
- **KD-tree Search**: O(log n) per query for efficient closest point finding
- **Memory Usage**: Scales with point cloud size and trajectory length
- **Processing Time**: Typically seconds to minutes depending on data size

### Coordinate Systems
- Uses right-handed coordinate system (Y-up, Z-forward)
- Compatible with standard SLAM conventions
- Quaternions in `[x, y, z, w]` format

## Troubleshooting

### Common Issues

1. **File Not Found Error**:
   - Check that PLY and TXT files exist
   - Verify file paths are correct

2. **Rerun Viewer Not Opening**:
   - Install Rerun: `pip install rerun-sdk`
   - Try manually: `rerun your_file.rrd`

3. **Memory Issues with Large Point Clouds**:
   - Consider downsampling the point cloud
   - Monitor system memory usage

4. **Incorrect Trajectory Format**:
   - Ensure TXT file has 8 columns: `timestamp x y z qx qy qz qw`
   - Check for proper space separation

### Performance Tips

- For large datasets, consider subsampling the trajectory
- Use SSD storage for faster file I/O
- Close other memory-intensive applications

## Future Enhancements

- [ ] Support for multiple trajectory formats
- [ ] Point cloud filtering and downsampling options
- [ ] Advanced distance metrics (surface distance, directional distance)
- [ ] Export to other visualization formats
- [ ] Real-time processing mode
- [ ] Camera frustum visualization
- [ ] Collision detection analysis

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is part of the PathPilot system for robotic navigation analysis. 