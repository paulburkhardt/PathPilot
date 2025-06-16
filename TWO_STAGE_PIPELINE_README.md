# Two-phase SLAM Pipeline

This document describes the two-phase PathPilot pipeline system that allows you to run SLAM processing and analysis in different environments.

## Overview

The pipeline is split into two phases to accommodate different environment requirements:

- **phase 1**: MAST3R SLAM processing (requires SLAM environment)
- **phase 2**: Analysis and visualization (requires analysis environment)

Intermediate files (PLY + TXT) are saved between phases to enable environment switching.

## Architecture

```
phase 1 (SLAM Environment):
Video → MAST3RSLAMVideoDataLoader → MAST3RSLAMComponent → SLAMOutputWriter → [PLY + TXT files]

phase 2 (Analysis Environment):
[PLY + TXT files] → PLYPointCloudLoader + TrajectoryDataLoader → Analysis Components → Visualization
```

## phase 1: SLAM Extraction

### Purpose
- Process video input through MAST3R SLAM
- Generate point cloud and camera trajectory
- Save intermediate outputs for phase 2

### Components
1. **MAST3RSLAMVideoDataLoader**: Loads video data
2. **MAST3RSLAMComponent**: Performs SLAM processing
3. **SLAMOutputWriter**: Saves PLY and TXT files

### Configuration
File: `configs/phase_1_slam_extraction.yaml`

```yaml
pipeline:
  components:
    - type: MAST3RSLAMVideoDataLoader
      config:
        video_path: MASt3R-SLAM/datasets/tum/rgbd_dataset_freiburg1_room/
        mast3r_slam_config_path: "configs/MASt3R-SLAM_configs/calib.yaml"
    
    - type: MAST3RSLAMComponent
      config:
        point_cloud_method: accumulating
        c_confidence_threshold: 1.0 
        mast3r_slam_config_path: "configs/MASt3R-SLAM_configs/calib.yaml"
    
    - type: SLAMOutputWriter
      config:
        output_dir: "intermediate_outputs"
        output_name: "slam_output"
        save_point_cloud: true
        save_trajectory: true
        create_timestamped_dir: true
```

### Outputs
- `intermediate_outputs/slam_output_TIMESTAMP/slam_output.ply`: Point cloud file
- `intermediate_outputs/slam_output_TIMESTAMP/slam_output.txt`: Camera trajectory file
- `intermediate_outputs/slam_output_TIMESTAMP/metadata.txt`: Information about the outputs

## phase 2: SLAM Analysis

### Purpose
- Load saved point cloud and trajectory
- Perform floor detection and closest point analysis
- Create interactive visualizations

### Components
1. **PLYPointCloudLoader**: Loads point cloud from PLY file
2. **TrajectoryDataLoader**: Loads camera trajectory from TXT file
3. **FloorDetectionComponent**: Detects floor plane
4. **ClosestPointFinderComponent**: Finds closest points to camera
5. **PointCloudDataVisualizer**: Visualizes point cloud with floor highlighting
6. **CameraTrajectoryVisualizer**: Visualizes trajectory with analysis

### Configuration
File: `configs/phase_2_slam_analysis.yaml`

```yaml
pipeline:
  components:
    - type: PLYPointCloudLoader
      config:
        ply_path: "intermediate_outputs/slam_output_TIMESTAMP/slam_output.ply"
        load_colors: true
    
    - type: TrajectoryDataLoader
      config:
        trajectory_path: "intermediate_outputs/slam_output_TIMESTAMP/slam_output.txt"
        validate_format: true
    
    # ... analysis components
```

## Usage

### Method 1: phase Coordinator (Recommended)

The `phase_coordinator.py` script automates the two-phase process:

```bash
# Run both phases automatically
python phase_coordinator.py --auto

# Run phase 1 only
python phase_coordinator.py --phase 1

# Run phase 2 only (automatically finds latest outputs)
python phase_coordinator.py --phase 2

# See what would be executed without running
python phase_coordinator.py --dry-run --auto

# Custom configurations
python phase_coordinator.py --auto \
  --phase1-config configs/custom_phase1.yaml \
  --phase2-config configs/custom_phase2.yaml
```

### Method 2: Manual Execution

```bash
# phase 1: SLAM extraction
python main.py --config-name=phase_1_slam_extraction

# Check outputs
ls intermediate_outputs/slam_output_*/

# phase 2: Update config with actual paths and run analysis
# (Edit phase_2_slam_analysis.yaml with correct timestamp)
python main.py --config-name=phase_2_slam_analysis
```

### Method 3: Environment Switch Workflow

1. **In SLAM Environment:**
   ```bash
   # Activate SLAM environment (e.g., conda activate slam_env)
   python phase_coordinator.py --phase 1
   ```

2. **Switch Environment:**
   ```bash
   # Deactivate SLAM environment
   # Activate analysis environment (e.g., conda activate analysis_env)
   ```

3. **In Analysis Environment:**
   ```bash
   python phase_coordinator.py --phase 2
   ```

## File Formats

### Point Cloud (PLY)
- Standard PLY format with vertices
- Contains XYZ coordinates
- Optional RGB colors (if available from SLAM)
- Optional confidence scores (if available from SLAM)

### Camera Trajectory (TXT)
- One pose per line
- Format: `timestamp x y z qx qy qz qw`
- Compatible with standard SLAM trajectory formats

## Configuration Options

### SLAMOutputWriter Options
- `output_dir`: Base directory for outputs (default: "intermediate_outputs")
- `output_name`: Base name for files (default: "slam_output")
- `save_point_cloud`: Whether to save PLY file (default: true)
- `save_trajectory`: Whether to save trajectory TXT file (default: true)
- `create_timestamped_dir`: Create timestamped subdirectory (default: true)

### phase Coordinator Options
- `--phase`: Run specific phase (1 or 2)
- `--auto`: Run both phases automatically
- `--dry-run`: Show commands without executing
- `--phase1-config`: Custom phase 1 configuration
- `--phase2-config`: Custom phase 2 configuration
- `--output-dir`: Directory to search for intermediate outputs

## Directory Structure

```
PathPilot/
├── configs/
│   ├── phase_1_slam_extraction.yaml      # phase 1 configuration
│   ├── phase_2_slam_analysis.yaml        # phase 2 configuration
│   └── phase_2_slam_analysis_updated_*.yaml  # Auto-generated configs
├── intermediate_outputs/
│   └── slam_output_TIMESTAMP/
│       ├── slam_output.ply                # Point cloud
│       ├── slam_output.txt                # Camera trajectory
│       └── metadata.txt                   # Output metadata
├── phase_coordinator.py                   # phase coordination script
└── main.py                               # Main pipeline entry point
```

## Benefits

1. **Environment Isolation**: Run SLAM and analysis in different environments
2. **Resumability**: Can restart from phase 2 if analysis fails
3. **Flexibility**: Can run phases on different machines
4. **Debugging**: Can inspect intermediate outputs between phases
5. **Automation**: phase coordinator handles file path management
6. **Validation**: Automatic validation of intermediate files

## Troubleshooting

### phase 1 Issues
- Check MAST3R SLAM environment setup
- Verify video path and SLAM configuration files
- Check disk space for intermediate outputs

### phase 2 Issues
- Ensure PLY and TXT files exist in intermediate_outputs
- Check file paths in phase 2 configuration
- Verify analysis environment has required packages (scipy, rerun, etc.)

### phase Coordinator Issues
- Run with `--dry-run` to see commands
- Check that main.py is in the current directory
- Verify configuration files exist

## Example Complete Workflow

```bash
# 1. Set up for two-phase processing
ls configs/phase_1_slam_extraction.yaml
ls configs/phase_2_slam_analysis.yaml

# 2. Run complete pipeline
python phase_coordinator.py --auto

# 3. Check outputs
ls intermediate_outputs/slam_output_*/
rerun intermediate_outputs/slam_output_*/slam_output.rrd

# 4. Or run phases separately with environment switch
python phase_coordinator.py --phase 1
# ... switch environment ...
python phase_coordinator.py --phase 2
```

This two-phase approach provides maximum flexibility while maintaining the modular architecture of the PathPilot pipeline system. 