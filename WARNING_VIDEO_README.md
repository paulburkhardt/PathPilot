# Enhanced Warning Video Generator

A tool to create enhanced navigation videos with visual and audio warnings for proximity detection and obstacle avoidance, specifically designed for accessibility and navigation assistance.

## Features

- **Visual Warning Overlays**: Color-coded warning bars, text overlays, and distance meters
- **Audio Warnings**: Text-to-speech alerts synchronized with trajectory steps
- **Accessibility**: Natural language directional guidance for visually impaired users
- **Multiple Warning Levels**: Normal (green), Caution (yellow), Strong (orange), Critical (red)
- **Customizable Thresholds**: Adjustable distance thresholds for different warning levels
- **Professional Output**: High-quality video encoding with step-based synchronization
- **Consistent Timing**: Uses same synchronization approach as visualize_slam_output.py

## Installation

### Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install -r requirements_warning_video.txt
   ```

2. **System Dependencies** (Linux):
   ```bash
   # For text-to-speech (espeak)
   sudo apt-get update
   sudo apt-get install espeak espeak-data libespeak-dev
   
   # For video processing (ffmpeg)
   sudo apt-get install ffmpeg
   ```

3. **Test Audio System**:
   ```bash
   python test_audio.py
   ```

### Dependencies

- **opencv-python**: Video processing and overlay generation
- **pyttsx3**: Text-to-speech audio generation
- **moviepy**: Video and audio combination
- **numpy**: Numerical computations
- **pathlib**: Path handling

## Usage

### Basic Usage

```bash
python create_warning_video.py /path/to/slam_analysis_output_dir
```

### Advanced Options

```bash
# Custom warning thresholds
python create_warning_video.py ./outputs --warning-distance-critical 0.15 --warning-distance-strong 0.35

# Disable audio warnings
python create_warning_video.py ./outputs --no-audio

# Adjust speech settings
python create_warning_video.py ./outputs --speech-rate 120 --speech-volume 0.9

# Control audio frequency
python create_warning_video.py ./outputs --audio-interval 3.0
```

### Complete Example

```bash
python create_warning_video.py \
    ./enhanced_slam_outputs/slam_analysis_20240115_143045 \
    --warning-distance-critical 0.15 \
    --warning-distance-strong 0.35 \
    --warning-distance-caution 0.7 \
    --speech-rate 140 \
    --audio-interval 2.5
```

## Command Line Options

### Required Arguments

- `output_dir`: Path to SLAM analysis output directory containing trajectory and closest points data

### Warning System Parameters

- `--warning-distance-critical FLOAT`: Distance threshold for critical warnings in meters (default: 0.2)
- `--warning-distance-strong FLOAT`: Distance threshold for strong warnings in meters (default: 0.4)
- `--warning-distance-caution FLOAT`: Distance threshold for caution warnings in meters (default: 0.6)

### Audio Parameters

- `--no-audio`: Disable audio warnings completely
- `--speech-rate INT`: Speech rate for audio warnings in words per minute (default: 150)
- `--speech-volume FLOAT`: Speech volume from 0.0 to 1.0 (default: 0.8)
- `--audio-interval FLOAT`: Minimum interval between audio warnings in seconds (default: 2.0)

## Output

The tool generates:

1. **Enhanced Video**: `enhanced_warning_video.mp4` with visual overlays
2. **Video with Audio**: `enhanced_warning_video_with_audio.mp4` (if audio enabled)

### Visual Elements

- **Color Bar**: Left side vertical bar showing current warning level
- **Warning Text**: Top center with distance and directional information
- **Distance Meter**: Bottom right showing distance scale and thresholds
- **Warning Colors**: Green (normal), Yellow (caution), Orange (strong), Red (critical)

### Audio Elements

- **Natural Language**: "Caution. Obstacle 0.5 meters ahead-left"
- **Escalating Urgency**: More urgent tone for closer obstacles
- **Spatial Guidance**: Directional information (ahead, left, right, above, below)

## Technical Details

### Synchronization

The tool uses **step-based synchronization** matching the approach in `visualize_slam_output.py`:

1. Trajectory poses are mapped evenly across video frames
2. Each SLAM analysis step corresponds to a specific video frame
3. Audio warnings are positioned based on step timing in the video

This ensures consistent timing between the real-time visualizer and the enhanced video output.

### Warning Levels

| Level | Distance | Color | Audio Prefix |
|-------|----------|-------|--------------|
| Normal | > 0.6m | Green | "Clear" |
| Caution | 0.4-0.6m | Yellow | "Caution" |
| Strong | 0.2-0.4m | Orange | "Warning" |
| Critical | < 0.2m | Red | "Critical Warning" |

### Directional Analysis

The system uses camera-relative coordinates to provide natural directional guidance:

- **Forward/Backward**: Based on camera's facing direction
- **Left/Right**: Relative to camera orientation
- **Above/Below**: Relative to camera height
- **Combined Directions**: "ahead-left", "behind-above", etc.

## Troubleshooting

### Audio Issues

1. **Test Audio System**:
   ```bash
   python test_audio.py
   ```

2. **Common Solutions**:
   ```bash
   # Install TTS engine (Linux)
   sudo apt-get install espeak espeak-data
   
   # Install Python TTS
   pip install pyttsx3
   
   # Install video processing
   pip install moviepy
   sudo apt-get install ffmpeg
   ```

3. **Audio Not Working**:
   - Check if speakers/headphones are connected
   - Verify system audio settings
   - Test with `--no-audio` flag to generate visual-only video
   - Run `test_audio.py` to diagnose TTS issues

### Video Issues

1. **Video Not Found**:
   - Ensure the original video file exists in the path specified in pipeline configuration
   - Check if video file is in a supported format (MP4, AVI, MOV)

2. **Synchronization Issues**:
   - Verify trajectory and closest points data lengths match
   - Check that SLAM analysis completed successfully
   - Ensure trajectory timestamps are valid

3. **Large File Sizes**:
   - Video processing creates uncompressed intermediate files
   - Final output is properly compressed with H.264 codec
   - Consider reducing video resolution if disk space is limited

### Performance

- **Memory Usage**: Proportional to video length and resolution
- **Processing Time**: ~1-2x video duration for encoding
- **Disk Space**: Temporary files may use 2-3x final video size during processing

## Integration

This tool is designed to work alongside:

- **`visualize_slam_output.py`**: Real-time analysis and visualization
- **SLAM Pipeline**: Generates trajectory and closest points data
- **Video Data**: Original navigation videos from the SLAM dataset

The enhanced videos can be used for:

- **Navigation Training**: Teaching obstacle avoidance patterns
- **Accessibility**: Audio-guided navigation for visually impaired users
- **Analysis**: Reviewing dangerous situations and near-misses
- **Documentation**: Creating accessible records of navigation sessions

## Examples

### Warning Messages

- **Normal**: "Distance: 1.2m (ahead-right)"
- **Caution**: "⚠️ CAUTION: Object 0.5m ahead-left"
- **Strong**: "⚠️ WARNING: Object 0.3m to your right"
- **Critical**: "🚨 CRITICAL: Object 0.15m ahead!"

### Audio Output

- **Caution**: "Caution. Obstacle 0.5 meters ahead-left"
- **Strong**: "Warning! Obstacle 0.3 meters to your right"
- **Critical**: "Critical warning! Obstacle 0.15 meters ahead!"

## Support

For issues and questions:

1. Check this README for common solutions
2. Run `test_audio.py` to diagnose audio problems
3. Verify input data integrity (trajectory, closest points, video)
4. Test with different command line options to isolate issues

The tool provides detailed console output to help diagnose problems during processing. 