#!/usr/bin/env python3
"""
PathPilot Warning System using Point Cloud Data
Calculates distances from trajectory to actual chair geometry from .ply file
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from scipy.spatial.distance import cdist

print("="*60)
print("    PathPilot Point Cloud Warning System")
print("="*60)
print()

# ============================================================================
# STEP 1: Load trajectory data from one_chair.txt
# ============================================================================
print("🔍 STEP 1: Loading trajectory data...")

def load_trajectory_data(filepath):
    """Load trajectory data from the one_chair.txt file"""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # timestamp + position + quaternion
                    timestamp = float(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    # Quaternion: qw, qx, qy, qz
                    qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    data.append([timestamp, x, y, z, qw, qx, qy, qz])
        return np.array(data)
    except Exception as e:
        print(f"❌ Error loading trajectory: {e}")
        return None

trajectory_data = load_trajectory_data('../plots/one_chair/one_chair.txt')

if trajectory_data is not None:
    print(f"✅ Loaded {len(trajectory_data)} trajectory points")
    print(f"   Time range: {trajectory_data[0,0]:.2f}s to {trajectory_data[-1,0]:.2f}s")
    print(f"   Position range:")
    print(f"     X: [{trajectory_data[:,1].min():.3f}, {trajectory_data[:,1].max():.3f}] m")
    print(f"     Y: [{trajectory_data[:,2].min():.3f}, {trajectory_data[:,2].max():.3f}] m") 
    print(f"     Z: [{trajectory_data[:,3].min():.3f}, {trajectory_data[:,3].max():.3f}] m")
else:
    print("❌ Failed to load trajectory data")
    exit(1)

print()

# ============================================================================
# STEP 2: Load point cloud from one_chair.ply
# ============================================================================
print("🪑 STEP 2: Loading chair point cloud...")

def load_point_cloud(filepath):
    """Load point cloud from .ply file using Open3D"""
    try:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        print(f"✅ Loaded point cloud with {len(points)} points")
        print(f"   Point cloud bounds:")
        print(f"     X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}] m")
        print(f"     Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}] m")
        print(f"     Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}] m")
        
        return points, colors
    except Exception as e:
        print(f"❌ Error loading point cloud: {e}")
        return None, None

chair_points, chair_colors = load_point_cloud('../plots/one_chair/one_chair.ply')

if chair_points is None:
    print("❌ Failed to load point cloud")
    exit(1)

print()

# ============================================================================
# STEP 3: Define warning system parameters
# ============================================================================
print("⚙️  STEP 3: Setting up warning parameters...")

warning_thresholds = {
    'danger': 0.15,    # Very close - immediate danger
    'warning': 0.30,   # Close - warning needed
    'caution': 0.50,   # Approaching - caution
    'safe': 1.0        # Safe distance
}

print(f"✅ Warning thresholds set:")
for level, distance in warning_thresholds.items():
    print(f"   {level.upper()}: {distance}m")
print()

# ============================================================================
# STEP 4: Calculate distances from trajectory to point cloud
# ============================================================================
print("📏 STEP 4: Calculating distances to chair point cloud...")

def calculate_min_distance_to_pointcloud(position, pointcloud):
    """Calculate minimum distance from a position to any point in the point cloud"""
    if len(pointcloud) == 0:
        return float('inf')
    
    # Calculate distances to all points in the cloud
    distances = cdist([position], pointcloud, metric='euclidean')[0]
    return np.min(distances)

def get_warning_level(distance, thresholds):
    """Determine warning level based on distance"""
    if distance <= thresholds['danger']:
        return 'DANGER', '🚨', 'red'
    elif distance <= thresholds['warning']:
        return 'WARNING', '⚠️', 'orange'
    elif distance <= thresholds['caution']:
        return 'CAUTION', '⚡', 'yellow'
    else:
        return 'SAFE', '✅', 'green'

# Process all trajectory points
print("Processing trajectory points...")
results = []

for i, point in enumerate(trajectory_data):
    if i % 10 == 0:  # Progress indicator
        print(f"  Processing point {i+1}/{len(trajectory_data)}")
    
    timestamp = point[0]
    position = point[1:4]  # x, y, z coordinates
    
    # Calculate minimum distance to chair point cloud
    min_distance = calculate_min_distance_to_pointcloud(position, chair_points)
    
    # Get warning level
    level, emoji, color = get_warning_level(min_distance, warning_thresholds)
    
    results.append({
        'timestamp': timestamp,
        'position': position,
        'distance': min_distance,
        'level': level,
        'emoji': emoji,
        'color': color
    })

print(f"✅ Processed {len(results)} trajectory points")

# Count warning levels
level_counts = {}
for result in results:
    level = result['level']
    level_counts[level] = level_counts.get(level, 0) + 1

print("\n📊 Warning level distribution:")
for level, count in level_counts.items():
    percentage = (count / len(results)) * 100
    emoji = next(r['emoji'] for r in results if r['level'] == level)
    print(f"   {emoji} {level}: {count} points ({percentage:.1f}%)")
print()

# ============================================================================
# STEP 5: Identify critical moments
# ============================================================================
print("🚨 STEP 5: Identifying critical moments...")

critical_moments = []
for i, result in enumerate(results):
    if result['level'] in ['DANGER', 'WARNING']:
        critical_moments.append({
            'index': i,
            'timestamp': result['timestamp'],
            'distance': result['distance'],
            'level': result['level'],
            'position': result['position']
        })

if critical_moments:
    print(f"⚠️  Found {len(critical_moments)} critical moments:")
    # Sort by distance (closest first)
    sorted_moments = sorted(critical_moments, key=lambda x: x['distance'])
    for i, moment in enumerate(sorted_moments[:5]):  # Show top 5 most critical
        print(f"   {i+1}. Time: {moment['timestamp']:6.2f}s, "
              f"Distance: {moment['distance']:.4f}m, "
              f"Level: {moment['level']}, "
              f"Pos: [{moment['position'][0]:.3f}, {moment['position'][1]:.3f}, {moment['position'][2]:.3f}]")
else:
    print("✅ No critical moments detected - safe trajectory!")
print()

# ============================================================================
# STEP 6: Create visualizations
# ============================================================================
print("📊 STEP 6: Creating visualizations...")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: 3D trajectory with point cloud
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Sample point cloud for visualization (too many points slow down plotting)
sample_indices = np.random.choice(len(chair_points), min(5000, len(chair_points)), replace=False)
sampled_chair_points = chair_points[sample_indices]

ax1.scatter(sampled_chair_points[:, 0], sampled_chair_points[:, 1], sampled_chair_points[:, 2], 
           c='brown', s=1, alpha=0.3, label='Chair Point Cloud')

# Plot trajectory with color coding
positions = np.array([r['position'] for r in results])
colors = [r['color'] for r in results]

ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
           c=colors, s=20, alpha=0.8, label='Trajectory')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Trajectory with Chair Point Cloud')
ax1.legend()

# Plot 2: Distance over time
ax2 = fig.add_subplot(2, 2, 2)
timestamps = [r['timestamp'] for r in results]
distances = [r['distance'] for r in results]

ax2.plot(timestamps, distances, 'b-', linewidth=2, label='Distance to chair')
ax2.axhline(y=warning_thresholds['danger'], color='red', linestyle='--', alpha=0.7, label='Danger')
ax2.axhline(y=warning_thresholds['warning'], color='orange', linestyle='--', alpha=0.7, label='Warning')
ax2.axhline(y=warning_thresholds['caution'], color='gold', linestyle='--', alpha=0.7, label='Caution')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Distance to Chair (m)')
ax2.set_title('Distance to Chair Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Top-down view (X-Y plane)
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(sampled_chair_points[:, 0], sampled_chair_points[:, 1], 
           c='brown', s=2, alpha=0.5, label='Chair (top view)')
ax3.scatter(positions[:, 0], positions[:, 1], c=colors, s=30, alpha=0.8, label='Trajectory')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_title('Top-Down View')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Warning level distribution
ax4 = fig.add_subplot(2, 2, 4)
levels = list(level_counts.keys())
counts = list(level_counts.values())
colors_bar = ['green' if l=='SAFE' else 'yellow' if l=='CAUTION' else 'orange' if l=='WARNING' else 'red' for l in levels]

bars = ax4.bar(levels, counts, color=colors_bar, alpha=0.7)
ax4.set_ylabel('Number of Points')
ax4.set_title('Warning Level Distribution')
ax4.grid(True, alpha=0.3)

# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('./one_chair/pointcloud_warning_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualizations saved as 'pointcloud_warning_analysis.png'")
print()

# ============================================================================
# STEP 7: Generate warning messages
# ============================================================================
print("📢 STEP 7: Sample real-time warning messages...")

def generate_warning_message(result):
    """Generate contextual warning message"""
    level = result['level']
    distance = result['distance']
    emoji = result['emoji']
    
    if level == 'DANGER':
        return f"{emoji} IMMEDIATE DANGER! Chair only {distance:.3f}m away - STOP NOW!"
    elif level == 'WARNING':
        return f"{emoji} WARNING: Chair {distance:.3f}m away - Slow down immediately!"
    elif level == 'CAUTION':
        return f"{emoji} CAUTION: Chair detected {distance:.3f}m ahead - Be careful!"
    else:
        return f"{emoji} Safe navigation - Chair {distance:.3f}m away"

# Show sample messages throughout trajectory
print("Sample warning messages during navigation:")
sample_step = max(1, len(results) // 10)  # Show ~10 samples
for i in range(0, len(results), sample_step):
    result = results[i]
    message = generate_warning_message(result)
    print(f"[{result['timestamp']:6.2f}s] {message}")
print()

# ============================================================================
# STEP 8: Safety analysis and recommendations
# ============================================================================
print("📋 STEP 8: Safety analysis and recommendations...")

# Calculate key metrics
min_distance = min(distances)
avg_distance = np.mean(distances)
median_distance = np.median(distances)
danger_count = level_counts.get('DANGER', 0)
warning_count = level_counts.get('WARNING', 0)
caution_count = level_counts.get('CAUTION', 0)
safe_count = level_counts.get('SAFE', 0)
total_points = len(results)

# Calculate safety score
unsafe_points = danger_count + warning_count
safety_score = ((total_points - unsafe_points) / total_points) * 100

print(f"📊 Trajectory Safety Analysis:")
print(f"   Total trajectory points: {total_points}")
print(f"   Minimum distance to chair: {min_distance:.4f}m")
print(f"   Average distance to chair: {avg_distance:.4f}m")
print(f"   Median distance to chair: {median_distance:.4f}m")
print(f"   Danger situations: {danger_count} ({danger_count/total_points*100:.1f}%)")
print(f"   Warning situations: {warning_count} ({warning_count/total_points*100:.1f}%)")
print(f"   Caution situations: {caution_count} ({caution_count/total_points*100:.1f}%)")
print(f"   Safe navigation: {safe_count} ({safe_count/total_points*100:.1f}%)")
print(f"   Overall safety score: {safety_score:.1f}%")
print()

print("🎯 Safety Recommendations:")
if safety_score >= 95:
    print("   ✅ EXCELLENT: Very safe navigation with minimal risk")
elif safety_score >= 85:
    print("   🟢 GOOD: Generally safe but monitor close approaches")
elif safety_score >= 70:
    print("   🟡 MODERATE: Some safety concerns - consider path adjustment")
elif safety_score >= 50:
    print("   🟠 CONCERNING: Multiple risky situations - path needs improvement")
else:
    print("   🔴 DANGEROUS: High collision risk - immediate path revision required")

if min_distance < 0.05:
    print("   🚨 CRITICAL: Extremely close approach detected - potential collision!")
elif min_distance < 0.10:
    print("   ⚠️  HIGH RISK: Very close approach - review navigation algorithm")

if danger_count > 0:
    print(f"   🚨 {danger_count} danger situations require immediate attention")
if warning_count > total_points * 0.1:  # More than 10% warnings
    print("   ⚠️  High frequency of warnings suggests path planning issues")

print()
print("="*60)
print("✅ Point Cloud Warning System Analysis Complete!")
print("="*60) 