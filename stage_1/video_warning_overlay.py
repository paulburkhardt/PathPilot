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
    """Calculate minimum distance from position to point cloud accounting for person height and excluding floor"""
    if len(pointcloud) == 0:
        return float('inf'), None
    
    # Person height threshold (1.80m)
    person_height = 1.80
    
    # Floor height threshold - filter out points that are likely floor
    # Assuming floor is around z=0, we'll exclude points below a small threshold
    floor_threshold = 0.1  # 10cm above floor level
    
    # Filter out floor points
    non_floor_points = pointcloud[pointcloud[:, 2] > floor_threshold]
    
    if len(non_floor_points) == 0:
        return float('inf'), None  # No obstacles above floor level
    
    # Extract position coordinates
    x, y, z = position
    
    # If trajectory point is below person height, only consider horizontal distance
    if z <= person_height:
        # Use only x,y coordinates for distance calculation
        position_2d = np.array([x, y])
        pointcloud_2d = non_floor_points[:, :2]  # Only x,y coordinates from filtered point cloud
        distances = cdist([position_2d], pointcloud_2d, metric='euclidean')[0]
    else:
        # For points above person height, use full 3D distance
        distances = cdist([position], non_floor_points, metric='euclidean')[0]
    
    min_distance = np.min(distances)
    closest_point_idx = np.argmin(distances)
    closest_point = non_floor_points[closest_point_idx]
    
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

def create_visualization_frame(trajectory_warnings, chair_points, current_time, frame_size=(800, 600)):
    """Create a top-down view visualization frame showing trajectory and closest points"""
    
    # Create a blank frame
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    
    # Get current warning info
    warning_info = get_warning_for_time(current_time, trajectory_warnings)
    
    if warning_info is None:
        return frame
    
    current_position = warning_info['position']
    closest_point = warning_info['closest_point']
    level = warning_info['level']
    distance = warning_info['distance']
    
    # Filter out floor points for visualization
    floor_threshold = 0.1
    non_floor_points = chair_points[chair_points[:, 2] > floor_threshold]
    
    # Calculate bounds for visualization
    all_points = np.vstack([trajectory_data[:, 1:4], non_floor_points])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    
    # Add some padding
    padding = 0.5
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    # Scale to frame
    x_scale = (frame_size[0] - 40) / (x_max - x_min)
    y_scale = (frame_size[1] - 40) / (y_max - y_min)
    scale = min(x_scale, y_scale)
    
    def world_to_screen(x, y):
        screen_x = int((x - x_min) * scale + 20)
        screen_y = int((y - y_min) * scale + 20)
        return screen_x, screen_y
    
    # Draw point cloud (chair points)
    for point in non_floor_points:
        sx, sy = world_to_screen(point[0], point[1])
        if 0 <= sx < frame_size[0] and 0 <= sy < frame_size[1]:
            cv2.circle(frame, (sx, sy), 2, (100, 100, 100), -1)  # Gray points
    
    # Draw trajectory path
    for i in range(len(trajectory_data) - 1):
        p1 = trajectory_data[i]
        p2 = trajectory_data[i + 1]
        sx1, sy1 = world_to_screen(p1[1], p1[2])
        sx2, sy2 = world_to_screen(p2[1], p2[2])
        cv2.line(frame, (sx1, sy1), (sx2, sy2), (150, 150, 150), 1)
    
    # Draw current position
    sx, sy = world_to_screen(current_position[0], current_position[1])
    color = warning_colors[level]
    cv2.circle(frame, (sx, sy), 8, color, -1)
    cv2.circle(frame, (sx, sy), 12, (255, 255, 255), 2)
    
    # Draw closest point if it exists
    if closest_point is not None:
        closest_sx, closest_sy = world_to_screen(closest_point[0], closest_point[1])
        cv2.circle(frame, (closest_sx, closest_sy), 6, (0, 255, 255), -1)  # Yellow
        cv2.circle(frame, (closest_sx, closest_sy), 8, (255, 255, 255), 2)
        
        # Draw line between current position and closest point
        cv2.line(frame, (sx, sy), (closest_sx, closest_sy), (0, 255, 255), 2)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(frame, "Top-Down View: Trajectory & Closest Points", (10, 25), 
               font, 0.7, (255, 255, 255), 2)
    
    # Current info
    info_y = frame_size[1] - 80
    cv2.putText(frame, f"Time: {current_time:.2f}s", (10, info_y), 
               font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Level: {level}", (10, info_y + 15), 
               font, 0.5, color, 1)
    cv2.putText(frame, f"Distance: {distance:.3f}m", (10, info_y + 30), 
               font, 0.5, (255, 255, 255), 1)
    
    if closest_point is not None:
        cv2.putText(frame, f"Closest Point: ({closest_point[0]:.2f}, {closest_point[1]:.2f}, {closest_point[2]:.2f})", 
                   (10, info_y + 45), font, 0.4, (0, 255, 255), 1)
    
    # Legend
    legend_x = frame_size[0] - 200
    cv2.putText(frame, "Legend:", (legend_x, 30), font, 0.5, (255, 255, 255), 1)
    cv2.circle(frame, (legend_x + 10, 45), 4, (100, 100, 100), -1)
    cv2.putText(frame, "Chair points", (legend_x + 25, 50), font, 0.4, (255, 255, 255), 1)
    cv2.circle(frame, (legend_x + 10, 65), 6, (0, 255, 0), -1)
    cv2.putText(frame, "Current position", (legend_x + 25, 70), font, 0.4, (255, 255, 255), 1)
    cv2.circle(frame, (legend_x + 10, 85), 4, (0, 255, 255), -1)
    cv2.putText(frame, "Closest point", (legend_x + 25, 90), font, 0.4, (255, 255, 255), 1)
    
    return frame

def create_visualization_video(output_path, trajectory_warnings, chair_points, fps=30.0):
    """Create a visualization video showing trajectory and closest points"""
    
    frame_size = (800, 600)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # Calculate video duration based on trajectory
    max_time = max(w['timestamp'] for w in trajectory_warnings)
    total_frames = int(max_time * fps)
    
    print(f"\n📊 Creating visualization video...")
    print(f"   Duration: {max_time:.2f} seconds")
    print(f"   Total frames: {total_frames}")
    
    for frame_num in range(total_frames):
        current_time = frame_num / fps
        
        # Create visualization frame
        vis_frame = create_visualization_frame(trajectory_warnings, chair_points, current_time, frame_size)
        
        # Write frame
        out.write(vis_frame)
        
        # Progress indicator
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"  Visualization progress: {progress:.1f}%")
    
    out.release()
    print(f"✅ Visualization video saved to: {output_path}")
    
    return True

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

# Process the video
input_video = "../Data/Videos/one_chair.mp4"
output_video = "./one_chair/one_chair_with_warnings_height_accounted.mp4"
visualization_video = "./one_chair/one_chair_closest_points_visualization.mp4"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_video), exist_ok=True)

print("🎬 Creating two videos:")
print(f"   1. Warning overlay video: {output_video}")
print(f"   2. Closest points visualization: {visualization_video}")
print()

# Create the warning overlay video
success1 = process_video_with_warnings(input_video, output_video, trajectory_warnings)

# Create the visualization video showing closest points
success2 = create_visualization_video(visualization_video, trajectory_warnings, chair_points)

if success1 and success2:
    print()
    print("="*60)
    print("✅ Both Videos Created Successfully!")
    print("="*60)
    print(f"📁 Warning overlay video: {output_video}")
    print(f"📊 Closest points visualization: {visualization_video}")
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
        
else:
    print("❌ Video processing failed!") 