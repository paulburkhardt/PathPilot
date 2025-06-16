#!/usr/bin/env python3
"""
PathPilot phase Coordinator
Coordinates running the two-phase SLAM pipeline with automatic file path management.

phase 1: MAST3R SLAM processing (video → point cloud + trajectory)
phase 2: SLAM analysis (point cloud + trajectory → visualization + analysis)
"""

import argparse
import pathlib
import subprocess
import yaml
import glob
import sys
from datetime import datetime


def find_latest_slam_output(base_dir: str = "intermediate_outputs") -> tuple[str, str]:
    """
    Find the latest SLAM output directory and return PLY and TXT file paths.
    
    Args:
        base_dir: Base directory to search for SLAM outputs
        
    Returns:
        Tuple of (ply_path, txt_path)
        
    Raises:
        FileNotFoundError: If no SLAM outputs are found
    """
    base_path = pathlib.Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Output directory {base_dir} does not exist. Run phase 1 first.")
    
    # Find all slam_output_* directories
    slam_dirs = list(base_path.glob("slam_output_*"))
    
    if not slam_dirs:
        raise FileNotFoundError(f"No SLAM output directories found in {base_dir}. Run phase 1 first.")
    
    # Sort by creation time (newest first)
    slam_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_dir = slam_dirs[0]
    
    # Find PLY and TXT files
    ply_files = list(latest_dir.glob("*.ply"))
    txt_files = list(latest_dir.glob("*.txt"))
    
    if not ply_files:
        raise FileNotFoundError(f"No PLY file found in {latest_dir}")
    if not txt_files or all('metadata' in f.name for f in txt_files):
        raise FileNotFoundError(f"No trajectory TXT file found in {latest_dir}")
    
    # Get the trajectory file (not metadata)
    trajectory_files = [f for f in txt_files if 'metadata' not in f.name]
    
    ply_path = str(ply_files[0])
    txt_path = str(trajectory_files[0])
    
    print(f"Found latest SLAM output in: {latest_dir}")
    print(f"  PLY file: {ply_path}")
    print(f"  TXT file: {txt_path}")
    
    return ply_path, txt_path


def update_phase2_config(config_path: str, ply_path: str, txt_path: str) -> str:
    """
    Update phase 2 configuration with actual file paths.
    
    Args:
        config_path: Path to phase 2 configuration file
        ply_path: Path to PLY file
        txt_path: Path to trajectory TXT file
        
    Returns:
        Path to updated configuration file
    """
    # Read the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update file paths
    for component in config['pipeline']['components']:
        if component['type'] == 'PLYPointCloudLoader':
            component['config']['ply_path'] = ply_path
        elif component['type'] == 'TrajectoryDataLoader':
            component['config']['trajectory_path'] = txt_path
    
    # Create updated config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_config_path = f"configs/phase_2_slam_analysis_updated_{timestamp}.yaml"
    
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Updated phase 2 configuration saved to: {updated_config_path}")
    return updated_config_path


def run_phase(phase: int, config_path: str, dry_run: bool = False) -> bool:
    """
    Run a pipeline phase.
    
    Args:
        phase: phase number (1 or 2)
        config_path: Path to configuration file
        dry_run: If True, only show the command that would be run
        
    Returns:
        True if successful, False otherwise
    """
    cmd = ["python", "main.py", f"--config-name={pathlib.Path(config_path).stem}"]
    
    print(f"\n{'='*50}")
    print(f"Running phase {phase}")
    print(f"Configuration: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    if dry_run:
        print("DRY RUN: Command would be executed here")
        return True
    
    try:
        # Change to the directory containing main.py if needed
        result = subprocess.run(cmd, check=True, cwd=".")
        print(f"\nphase {phase} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nphase {phase} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\nError: Could not find main.py or python executable")
        print("Make sure you're running this script from the PathPilot root directory")
        return False


def main():
    parser = argparse.ArgumentParser(description="PathPilot Two-phase SLAM Pipeline Coordinator")
    parser.add_argument("--phase", type=int, choices=[1, 2], 
                       help="Run specific phase (1: SLAM extraction, 2: analysis)")
    parser.add_argument("--phase1-config", default="configs/phase_1_slam_extraction.yaml",
                       help="Configuration file for phase 1 (default: configs/phase_1_slam_extraction.yaml)")
    parser.add_argument("--phase2-config", default="configs/phase_2_slam_analysis.yaml", 
                       help="Configuration file for phase 2 (default: configs/phase_2_slam_analysis.yaml)")
    parser.add_argument("--output-dir", default="intermediate_outputs",
                       help="Directory to search for intermediate outputs (default: intermediate_outputs)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show commands that would be run without executing them")
    parser.add_argument("--auto", action="store_true",
                       help="Run both phases automatically")
    
    args = parser.parse_args()
    
    # Validate configuration files exist
    if args.phase in [None, 1] or args.auto:
        if not pathlib.Path(args.phase1_config).exists():
            print(f"Error: phase 1 configuration file not found: {args.phase1_config}")
            sys.exit(1)
    
    if args.phase in [None, 2] or args.auto:
        if not pathlib.Path(args.phase2_config).exists():
            print(f"Error: phase 2 configuration file not found: {args.phase2_config}")
            sys.exit(1)
    
    try:
        if args.auto:
            # Run both phases automatically
            print("Running both phases automatically...")
            
            # phase 1
            success = run_phase(1, args.phase1_config, args.dry_run)
            if not success:
                print("phase 1 failed. Aborting.")
                sys.exit(1)
            
            if not args.dry_run:
                # Find the generated outputs
                ply_path, txt_path = find_latest_slam_output(args.output_dir)
                
                # Update phase 2 configuration
                updated_config = update_phase2_config(args.phase2_config, ply_path, txt_path)
            else:
                updated_config = args.phase2_config
            
            # phase 2
            success = run_phase(2, updated_config, args.dry_run)
            if not success:
                print("phase 2 failed.")
                sys.exit(1)
                
        elif args.phase == 1:
            # Run phase 1 only
            success = run_phase(1, args.phase1_config, args.dry_run)
            if not success:
                sys.exit(1)
            
            if not args.dry_run:
                print(f"\nphase 1 complete! To run phase 2:")
                print(f"python {sys.argv[0]} --phase 2")
                
        elif args.phase == 2:
            # Run phase 2 only
            if not args.dry_run:
                # Find the latest outputs and update config
                ply_path, txt_path = find_latest_slam_output(args.output_dir)
                updated_config = update_phase2_config(args.phase2_config, ply_path, txt_path)
            else:
                updated_config = args.phase2_config
                
            success = run_phase(2, updated_config, args.dry_run)
            if not success:
                sys.exit(1)
        else:
            # Show usage
            print("PathPilot Two-phase SLAM Pipeline")
            print("=================================")
            print()
            print("Usage examples:")
            print(f"  {sys.argv[0]} --auto                    # Run both phases automatically")
            print(f"  {sys.argv[0]} --phase 1                # Run phase 1 (SLAM extraction)")
            print(f"  {sys.argv[0]} --phase 2                # Run phase 2 (analysis)")
            print(f"  {sys.argv[0]} --dry-run --auto         # Show what would be executed")
            print()
            print("phase 1: Video → SLAM processing → saves PLY + TXT files")
            print("phase 2: PLY + TXT files → floor detection + analysis + visualization")
            print()
            print("Use --help for all available options")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 