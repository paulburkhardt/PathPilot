#!/usr/bin/env python3
"""
Demo script for PathPilot pipeline using the one_chair dataset.
This script demonstrates how to process Mast3r-Slam outputs.
"""

import subprocess
import pathlib
import sys


def main():
    """Run the PathPilot pipeline on the one_chair dataset."""
    
    # Define paths to the Mast3r-Slam output files
    base_dir = pathlib.Path("plots/one_chair")
    ply_file = base_dir / "one_chair.ply"
    trajectory_file = base_dir / "one_chair.txt"
    output_file = "one_chair_pathpilot.rrd"
    
    # Check if input files exist
    if not ply_file.exists():
        print(f"Error: PLY file not found at {ply_file}")
        print("Please run Mast3r-Slam first to generate the point cloud.")
        return 1
    
    if not trajectory_file.exists():
        print(f"Error: Trajectory file not found at {trajectory_file}")
        print("Please run Mast3r-Slam first to generate the camera trajectory.")
        return 1
    
    print("=== PathPilot Demo: Processing one_chair dataset ===\n")
    print(f"Point cloud file: {ply_file}")
    print(f"Trajectory file: {trajectory_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Build the command
    cmd = [
        sys.executable, "process_slam_output.py",
        "--ply", str(ply_file),
        "--trajectory", str(trajectory_file),
        "--output", output_file,
        "--name", "OneChair_PathPilot"
    ]
    
    print("Running PathPilot pipeline...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the pipeline
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("=== Pipeline Output ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== Warnings/Errors ===")
            print(result.stderr)
        
        print(f"\n=== Success! ===")
        print(f"PathPilot visualization saved to: {output_file}")
        print(f"To view the result, run: rerun {output_file}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error running pipeline: {e}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return 1
    
    except FileNotFoundError:
        print("Error: Could not find process_slam_output.py script")
        print("Make sure you're running this from the correct directory")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 