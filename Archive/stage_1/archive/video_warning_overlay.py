#!/usr/bin/env python3
"""
PathPilot Video Warning Overlay System
Overlays real-time warning information on the video based on trajectory and point cloud data
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.distance import cdist
import os
from datetime import datetime
import rerun as rr

print("="*60)
print("    PathPilot Video Warning Overlay System")
print("="*60)
print()

# ============================================================================
# STEP 1: Load trajectory data and point cloud
# ============================================================================
print("🔍 STEP 1: Loading trajectory and point cloud data...")

def load_trajectory_data(filepath):
    """Load trajectory data from the one_chair.txt file"""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    timestamp = float(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    data.append([timestamp, x, y, z, qw, qx, qy, qz])
        return np.array(data)
    except Exception as e:
        print(f"❌ Error loading trajectory: {e}")
        return None

def load_point_cloud(filepath):
    """Load point cloud from .ply file"""
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        print(f"✅ Loaded point cloud with {len(points)} points")
        return points
    except Exception as e:
        print(f"❌ Error loading point cloud: {e}")
        return None

# Load data
trajectory_data = load_trajectory_data('../plots/one_chair/one_chair.txt')
chair_points = load_point_cloud('../plots/one_chair/one_chair.ply')

if trajectory_data is None or chair_points is None:
    print("❌ Failed to load required data")
    exit(1)

print(f"✅ Loaded {len(trajectory_data)} trajectory points")
print(f"✅ Loaded point cloud with {len(chair_points)} points")
print()

# ============================================================================
# STEP 2: Define warning system parameters
# ============================================================================
print("⚙️  STEP 2: Setting up warning parameters...")

warning_thresholds = {
    'danger': 0.15,    # Very close - immediate danger
    'warning': 0.30,   # Close - warning needed  
    'caution': 0.50,   # Approaching - caution
}

# Warning colors (BGR format for OpenCV)
warning_colors = {
    'DANGER': (0, 0, 255),      # Red
    'WARNING': (0, 165, 255),   # Orange
    'CAUTION': (0, 255, 255),   # Yellow
    'SAFE': (0, 255, 0),        # Green
}

print(f"✅ Warning system configured")
print()

# ============================================================================
# STEP 3: Calculate distances for all trajectory points
# ============================================================================
print("📏 STEP 3: Pre-calculating distances...")

def calculate_min_distance_to_pointcloud(position, pointcloud):
    """Calculate minimum distance from position to point cloud"""
    if len(pointcloud) == 0:
        return float('inf'), None
    
    # Calculate distances to all points in the cloud
    distances = cdist([position], pointcloud, metric='euclidean')[0]
    
    min_distance = np.min(distances)
    closest_point_idx = np.argmin(distances)
    closest_point = pointcloud[closest_point_idx]
    
    return min_distance, closest_point

def get_warning_level(distance, thresholds):
    """Determine warning level based on distance"""
    if distance <= thresholds['danger']:
        return 'DANGER'
    elif distance <= thresholds['warning']:
        return 'WARNING'
    elif distance <= thresholds['caution']:
        return 'CAUTION'
    else:
        return 'SAFE'

# Pre-calculate all distances
trajectory_warnings = []
print("Calculating distances for all trajectory points...")

for i, point in enumerate(trajectory_data):
    if i % 10 == 0:
        print(f"  Processing {i+1}/{len(trajectory_data)}")
    
    timestamp = point[0]
    position = point[1:4]
    
    distance, closest_point = calculate_min_distance_to_pointcloud(position, chair_points)
    level = get_warning_level(distance, warning_thresholds)
    
    trajectory_warnings.append({
        'timestamp': timestamp,
        'position': position,
        'distance': distance,
        'level': level,
        'closest_point': closest_point
    })

print(f"✅ Pre-calculated warnings for {len(trajectory_warnings)} points")
print()

# ============================================================================
# STEP 4: Video processing functions
# ============================================================================
print("🎥 STEP 4: Setting up video processing...")

def get_warning_for_time(timestamp, trajectory_warnings):
    """Get the warning information for a specific timestamp"""
    # Find the closest trajectory point to this timestamp
    best_match = None
    min_time_diff = float('inf')
    
    for warning in trajectory_warnings:
        time_diff = abs(warning['timestamp'] - timestamp)
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            best_match = warning
    
    return best_match

def draw_warning_overlay(frame, warning_info, frame_time):
    """Draw warning overlay on the video frame"""
    height, width = frame.shape[:2]
    
    if warning_info is None:
        return frame
    
    level = warning_info['level']
    distance = warning_info['distance']
    color = warning_colors[level]
    
    # Create overlay
    overlay = frame.copy()
    
    # Warning banner at top
    banner_height = 80
    cv2.rectangle(overlay, (0, 0), (width, banner_height), color, -1)
    
    # Warning text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Main warning message
    if level == 'DANGER':
        main_text = "🚨 DANGER - STOP!"
        sub_text = f"Chair only {distance:.3f}m away"
    elif level == 'WARNING':
        main_text = "⚠️ WARNING - SLOW DOWN"
        sub_text = f"Chair {distance:.3f}m away"
    elif level == 'CAUTION':
        main_text = "⚡ CAUTION"
        sub_text = f"Chair {distance:.3f}m ahead"
    else:
        main_text = "✅ SAFE"
        sub_text = f"Chair {distance:.3f}m away"
    
    # Calculate text positions
    (text_width, text_height), _ = cv2.getTextSize(main_text, font, font_scale, thickness)
    main_x = (width - text_width) // 2
    main_y = 35
    
    (sub_width, sub_height), _ = cv2.getTextSize(sub_text, font, 0.8, 2)
    sub_x = (width - sub_width) // 2
    sub_y = 65
    
    # Draw text with black outline for visibility
    cv2.putText(overlay, main_text, (main_x, main_y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(overlay, main_text, (main_x, main_y), font, font_scale, (255, 255, 255), thickness)
    
    cv2.putText(overlay, sub_text, (sub_x, sub_y), font, 0.8, (0, 0, 0), 4)
    cv2.putText(overlay, sub_text, (sub_x, sub_y), font, 0.8, (255, 255, 255), 2)
    
    # Distance indicator on the right side
    indicator_x = width - 200
    indicator_y = 120
    indicator_width = 150
    indicator_height = 20
    
    # Background for distance indicator
    cv2.rectangle(overlay, (indicator_x, indicator_y), 
                 (indicator_x + indicator_width, indicator_y + indicator_height), 
                 (50, 50, 50), -1)
    
    # Distance bar (red to green based on distance)
    max_distance = 1.0  # Maximum distance for visualization
    distance_ratio = min(distance / max_distance, 1.0)
    bar_width = int(indicator_width * distance_ratio)
    
    # Color interpolation from red to green
    if distance_ratio < 0.5:
        bar_color = (0, int(255 * distance_ratio * 2), 255)  # Red to yellow
    else:
        bar_color = (0, 255, int(255 * (2 - distance_ratio * 2)))  # Yellow to green
    
    cv2.rectangle(overlay, (indicator_x, indicator_y), 
                 (indicator_x + bar_width, indicator_y + indicator_height), 
                 bar_color, -1)
    
    # Distance text
    distance_text = f"{distance:.3f}m"
    cv2.putText(overlay, distance_text, (indicator_x, indicator_y + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Timestamp
    time_text = f"Time: {frame_time:.2f}s"
    cv2.putText(overlay, time_text, (10, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay with original frame
    alpha = 0.8 if level in ['DANGER', 'WARNING'] else 0.6
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return result

print("✅ Video processing functions ready")
print()

# ============================================================================
# STEP 5: Process video with warning overlays
# ============================================================================
print("🎬 STEP 5: Processing video with warning overlays...")

# Top-down visualization removed as requested

def process_video_with_warnings(input_path, output_path, trajectory_warnings):
    """Process video and add warning overlays"""
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("\nProcessing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate current timestamp
        current_time = frame_count / fps
        
        # Get warning for current time
        warning_info = get_warning_for_time(current_time, trajectory_warnings)
        
        # Add warning overlay
        frame_with_warning = draw_warning_overlay(frame, warning_info, current_time)
        
        # Write frame
        out.write(frame_with_warning)
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 30 == 0:  # Every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"✅ Video processing complete!")
    print(f"   Output saved to: {output_path}")
    
    return True


def create_rerun_3d_visualization(trajectory_warnings, chair_points, output_name="pathpilot_3d_viz"):
    """Create a 3D visualization using Rerun showing trajectory and closest points over time"""
    
    # Create output path for the .rrd file
    output_path = f"./one_chair/{output_name}.rrd"
    
    # Initialize Rerun with web serving configuration
    rr.init(output_name)
    rr.serve_web()  # Use web server instead of saving to file
    rr.save(output_path)  # Still save to file as backup
    
    print(f"\n🎯 Creating 3D Rerun visualization...")
    print(f"   Starting web server for real-time viewing...")
    print(f"   Also saving to file: {output_path}")
    
    # Log the complete point cloud data
    if len(chair_points) > 0:
        print(f"   Logging complete point cloud with {len(chair_points)} points...")
        
        # Create height-based coloring for better visualization
        z_values = chair_points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        z_range = z_max - z_min
        
        # Color points based on height: blue (low) to red (high)
        colors = []
        for z in z_values:
            if z_range > 0:
                normalized_height = (z - z_min) / z_range
                # Create gradient from blue (floor) to brown (furniture) to red (high)
                if normalized_height < 0.3:  # Lower points - bluish
                    colors.append([int(100 + normalized_height * 155), int(100 + normalized_height * 155), 255])
                elif normalized_height < 0.7:  # Middle points - brownish
                    colors.append([139, 69, 19])
                else:  # Higher points - reddish
                    colors.append([255, int(100 - (normalized_height - 0.7) * 100), 0])
            else:
                colors.append([139, 69, 19])  # Default brown if no height variation
        
        rr.log(
            "world/complete_pointcloud",
            rr.Points3D(
                positions=chair_points,
                colors=colors,
                radii=0.008,  # Slightly smaller for better visibility with many points
            ),
        )
        
        # Also log point cloud statistics
        rr.log("pointcloud_stats/total_points", rr.Scalar(len(chair_points)))
        rr.log("pointcloud_stats/height_range", rr.Scalar(z_range))
        rr.log("pointcloud_stats/min_height", rr.Scalar(z_min))
        rr.log("pointcloud_stats/max_height", rr.Scalar(z_max))
        
        # Log bounding box for reference
        x_min, x_max = chair_points[:, 0].min(), chair_points[:, 0].max()
        y_min, y_max = chair_points[:, 1].min(), chair_points[:, 1].max()
        
        # Create bounding box wireframe
        bbox_corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        
        # Define edges of bounding box
        bbox_edges = [
            # Bottom face
            [bbox_corners[0], bbox_corners[1]], [bbox_corners[1], bbox_corners[2]],
            [bbox_corners[2], bbox_corners[3]], [bbox_corners[3], bbox_corners[0]],
            # Top face  
            [bbox_corners[4], bbox_corners[5]], [bbox_corners[5], bbox_corners[6]],
            [bbox_corners[6], bbox_corners[7]], [bbox_corners[7], bbox_corners[4]],
            # Vertical edges
            [bbox_corners[0], bbox_corners[4]], [bbox_corners[1], bbox_corners[5]],
            [bbox_corners[2], bbox_corners[6]], [bbox_corners[3], bbox_corners[7]]
        ]
        
        rr.log(
            "world/bounding_box",
            rr.LineStrips3D(
                strips=bbox_edges,
                colors=[128, 128, 128],  # Gray wireframe
                radii=0.002,
            ),
        )
    
    # Log the complete trajectory path
    trajectory_positions = np.array([w['position'] for w in trajectory_warnings])
    rr.log(
        "world/trajectory_path",
        rr.LineStrips3D(
            strips=[trajectory_positions],
            colors=[150, 150, 150],  # Light gray
            radii=0.005,
        ),
    )
    
    # Create timeline visualization (simplified)
    print("   Logging timeline data...")
    
    for i, warning in enumerate(trajectory_warnings):
        if i % 20 == 0:  # Log every 20th point to reduce load
            print(f"   Processing {i+1}/{len(trajectory_warnings)}")
        
        timestamp = warning['timestamp']
        position = warning['position']
        closest_point = warning['closest_point']
        level = warning['level']
        distance = warning['distance']
        
        # Set timeline
        rr.set_time_seconds("timeline", timestamp)
        
        # Color based on warning level
        if level == 'DANGER':
            person_color = [255, 0, 0]  # Red
        elif level == 'WARNING':
            person_color = [255, 165, 0]  # Orange
        elif level == 'CAUTION':
            person_color = [255, 255, 0]  # Yellow
        else:
            person_color = [0, 255, 0]  # Green
        
        # Log current person position
        rr.log(
            "world/person_position",
            rr.Points3D(
                positions=[position],
                colors=[person_color],
                radii=0.05,
            ),
        )
        
        # Log closest point if it exists
        if closest_point is not None:
            rr.log(
                "world/closest_point",
                rr.Points3D(
                    positions=[closest_point],
                    colors=[255, 255, 0],  # Yellow
                    radii=0.03,
                ),
            )
            
            # Log connection line between person and closest point
            rr.log(
                "world/distance_line",
                rr.LineStrips3D(
                    strips=[np.array([position, closest_point])],
                    colors=[255, 255, 0],  # Yellow
                    radii=0.01,
                ),
            )
        
        # Log basic statistics only
        rr.log("stats/distance", rr.Scalar(distance))
        rr.log("stats/warning_level", rr.TextLog(f"Level: {level}"))
    
    print(f"✅ 3D Rerun visualization created!")
    print(f"   🌐 Web server started - open browser to view live visualization")
    print(f"   📁 File also saved to: {output_path}")
    print(f"   💡 Alternative viewing: rerun {output_path}")
    
    return True

# Process the video
input_video = "../Data/Videos/one_chair.mp4"
output_video = "./one_chair/one_chair_with_warnings.mp4"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_video), exist_ok=True)

print("🎬 Creating visualizations:")
print(f"   1. Warning overlay video: {output_video}")
print(f"   2. Interactive 3D Rerun visualization (web server)")
print()

# Create the warning overlay video
success1 = process_video_with_warnings(input_video, output_video, trajectory_warnings)

# Create the 3D Rerun visualization
success2 = create_rerun_3d_visualization(trajectory_warnings, chair_points, "pathpilot_chair_navigation_2")

if success1 and success2:
    print()
    print("="*60)
    print("✅ All Visualizations Created Successfully!")
    print("="*60)
    print(f"📁 Warning overlay video: {output_video}")
    print(f"🎯 Interactive 3D Rerun visualization: Web server started")
    print()
    
    # Generate summary statistics
    level_counts = {}
    for warning in trajectory_warnings:
        level = warning['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("📊 Warning Summary:")
    total_points = len(trajectory_warnings)
    for level, count in level_counts.items():
        percentage = (count / total_points) * 100
        emoji = "🚨" if level == "DANGER" else "⚠️" if level == "WARNING" else "⚡" if level == "CAUTION" else "✅"
        print(f"   {emoji} {level}: {count} moments ({percentage:.1f}%)")
    
    min_distance = min(w['distance'] for w in trajectory_warnings)
    avg_distance = np.mean([w['distance'] for w in trajectory_warnings])
    
    print(f"\n📏 Distance Statistics:")
    print(f"   Minimum distance: {min_distance:.4f}m")
    print(f"   Average distance: {avg_distance:.4f}m")
    
    # Closest point statistics
    closest_points = [w['closest_point'] for w in trajectory_warnings if w['closest_point'] is not None]
    if closest_points:
        closest_points_array = np.array(closest_points)
        avg_closest_height = np.mean(closest_points_array[:, 2])
        print(f"   Average height of closest points: {avg_closest_height:.3f}m")
    
    if min_distance < 0.1:
        print(f"   🚨 CRITICAL: Very close approach detected!")
    elif level_counts.get('DANGER', 0) > 0:
        print(f"   ⚠️  WARNING: {level_counts['DANGER']} danger situations in video")
    else:
        print(f"   ✅ SAFE: No immediate danger situations detected")
    
    if success2:
        print(f"\n🎯 3D Visualization Features:")
        print(f"   • Complete point cloud visualization (ALL {len(chair_points)} points)")
        print(f"   • Height-based color coding for better depth perception")
        print(f"   • Bounding box wireframe for spatial reference")
        print(f"   • Real-time person position tracking")
        print(f"   • Closest point highlighting")
        print(f"   • Dynamic distance measurements and warning levels")
        print(f"   • Interactive timeline navigation")
        print(f"   • Full 3D rotation and zoom capabilities")
        print(f"   • Point cloud statistics panel")
        print(f"\n🎮 Visualization Legend:")
        print(f"   • Blue points = Lower/floor level points")
        print(f"   • Brown points = Mid-level furniture points")
        print(f"   • Red points = Higher elevation points")
        print(f"   • Gray wireframe = Scene bounding box")
        print(f"   • Gray line = Complete trajectory path")
        print(f"   • Colored sphere = Person position (color = warning level)")
        print(f"   • Yellow point = Current closest point")
        print(f"   • Yellow line = Distance connection")
        print(f"\n📊 Point Cloud Stats:")
        print(f"   • Total points: {len(chair_points)}")
        z_values = chair_points[:, 2]
        print(f"   • Height range: {z_values.min():.3f}m to {z_values.max():.3f}m")
        print(f"   • Point density: Enhanced visualization with height coloring")
         
else:
    print("❌ Visualization processing failed!")