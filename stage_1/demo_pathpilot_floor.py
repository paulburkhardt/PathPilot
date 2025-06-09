#!/usr/bin/env python3
"""
Demo script for PathPilot with floor distance calculation
Shows both 3D distances and horizontal floor distances for collision detection
"""

import subprocess
import pathlib
import sys

def main():
    # Check if required files exist
    ply_file = pathlib.Path("../plots/two_chairs_and_trash/two_chairs_and_trash.ply")
    traj_file = pathlib.Path("../plots/two_chairs_and_trash/two_chairs_and_trash.txt")
    
    if not ply_file.exists():
        print(f"Error: PLY file not found: {ply_file}")
        print("Please ensure the one_chair dataset is available in plots/one_chair/")
        sys.exit(1)
    
    if not traj_file.exists():
        print(f"Error: Trajectory file not found: {traj_file}")
        print("Please ensure the one_chair dataset is available in plots/one_chair/")
        sys.exit(1)
    
    print("=== PathPilot Demo with Floor Distance Calculation ===")
    print("This demo shows both 3D and horizontal floor distances")
    print("Floor distances are useful for collision detection as they show")
    print("horizontal clearance regardless of object height.\n")
    
    # Run PathPilot with floor distance calculation
    cmd = [
        "python", "process_slam_output.py",
        "--ply", str(ply_file),
        "--trajectory", str(traj_file),
        "--output", "two_chairs_and_trash/two_chairs_and_trash_floor_demo.rrd",
        "--name", "PathPilot_Floor_Demo",
        "--align-floor",           # Enable floor detection
        "--use-floor-distance",    # Enable floor distance calculation
        "--use-view-cone",         # Enable view cone filtering
        "--cone-angle", "60",      # 60-degree half-angle
        "--max-view-distance", "5", # 5-meter max distance
        "--floor-threshold", "0.05" # 5cm threshold for floor detection
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "="*50)
        print("Demo completed successfully!")
        print("Output file: one_chair_floor_demo.rrd")
        print("\nTo view the result:")
        print("rerun one_chair_floor_demo.rrd")
        print("\nIn the viewer, you'll see:")
        print("- Green points: 3D closest points")
        print("- Orange points: Floor-projected closest points")
        print("- Yellow lines: 3D distance lines")
        print("- Orange lines: Floor distance lines")
        print("- Two distance plots: '3D' and 'Floor'")
        print("- Floor visualization with green highlights and yellow grid")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running PathPilot: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: process_slam_output.py not found in current directory")
        sys.exit(1)

if __name__ == "__main__":
    main() 