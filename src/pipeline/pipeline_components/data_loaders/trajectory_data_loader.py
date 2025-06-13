from typing import List, Dict, Any, Tuple
import numpy as np
import pathlib
from .abstract_data_loader import AbstractDataLoader


class TrajectoryDataLoader(AbstractDataLoader):
    """
    Data loader component for camera trajectory files.
    Loads trajectory data from TXT files with format: timestamp x y z qx qy qz qw
    
    Args:
        trajectory_path: Path to trajectory TXT file
        validate_format: Whether to validate trajectory format (default: True)
        
    Returns:
        Dictionary containing trajectory data
        
    Raises:
        FileNotFoundError: If trajectory file doesn't exist
        ValueError: If trajectory file format is invalid
    """
    
    def __init__(self, trajectory_path: str, validate_format: bool = True) -> None:
        super().__init__()
        self.trajectory_path = pathlib.Path(trajectory_path)
        self.validate_format = validate_format
        
        # Load trajectory data during initialization
        self._load_trajectory_data()
        
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component is a data loader and doesn't require inputs."""
        return []

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs trajectory data."""
        return ["camera_positions", "camera_quaternions", "timestamps"]

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Load and return trajectory data.
        
        Args:
            **kwargs: Unused arguments
            
        Returns:
            Dictionary containing trajectory data
        """
        return {
            "camera_positions": self.camera_positions,
            "camera_quaternions": self.camera_quaternions,
            "timestamps": self.timestamps
        }

    def _load_trajectory_data(self) -> None:
        """Load trajectory data from file."""
        if not self.trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_path}")
            
        print(f"Loading camera trajectory from: {self.trajectory_path}")
        
        # Parse trajectory file
        timestamps, positions, quaternions = self._parse_trajectory_file()
        
        # Store as instance variables
        self.timestamps = timestamps
        self.camera_positions = positions
        self.camera_quaternions = quaternions
        
        print(f"Loaded trajectory with {len(positions)} poses")
        print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} seconds")
        print(f"Position range: [{positions.min(axis=0)}, {positions.max(axis=0)}]")

    def _parse_trajectory_file(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse camera trajectory from TXT file.
        
        Expected format: timestamp x y z qx qy qz qw
        
        Returns:
            timestamps: Array of timestamps
            positions: Nx3 array of camera positions
            quaternions: Nx4 array of camera orientations (x,y,z,w)
        """
        # Read the file line by line to handle formatting issues
        valid_lines = []
        with open(self.trajectory_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Split the line into values
                values = line.split()
                
                # Only keep lines with exactly 8 values
                if len(values) == 8:
                    try:
                        # Try to convert to float to validate
                        float_values = [float(v) for v in values]
                        valid_lines.append(float_values)
                    except ValueError:
                        if self.validate_format:
                            print(f"Warning: Skipping line {line_num} - invalid float values")
                else:
                    if self.validate_format:
                        print(f"Warning: Skipping line {line_num} - has {len(values)} columns instead of 8")
        
        if not valid_lines:
            raise ValueError("No valid trajectory data found in file")
        
        # Convert to numpy array
        data = np.array(valid_lines, dtype=np.float64)
        
        timestamps = data[:, 0]
        positions = data[:, 1:4].astype(np.float32)
        quaternions = data[:, 4:8].astype(np.float32)
        
        # Validate trajectory data
        if self.validate_format:
            self._validate_trajectory_data(timestamps, positions, quaternions, len(valid_lines), line_num)
        
        return timestamps, positions, quaternions

    def _validate_trajectory_data(self, timestamps: np.ndarray, positions: np.ndarray,
                                 quaternions: np.ndarray, valid_lines: int, total_lines: int) -> None:
        """Validate loaded trajectory data."""
        # Check for monotonic timestamps
        if not np.all(np.diff(timestamps) >= 0):
            print("Warning: Timestamps are not monotonically increasing")
        
        # Check for reasonable position values
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            raise ValueError("Invalid position values found (NaN or Inf)")
            
        # Check for reasonable quaternion values
        if np.any(np.isnan(quaternions)) or np.any(np.isinf(quaternions)):
            raise ValueError("Invalid quaternion values found (NaN or Inf)")
        
        # Check quaternion normalization
        quaternion_norms = np.linalg.norm(quaternions, axis=1)
        if not np.allclose(quaternion_norms, 1.0, atol=1e-3):
            print("Warning: Some quaternions are not properly normalized")
            # Normalize quaternions
            quaternions = quaternions / quaternion_norms[:, np.newaxis]
        
        # Report data quality
        skipped_lines = total_lines - valid_lines
        if skipped_lines > 0:
            print(f"Skipped {skipped_lines} invalid lines out of {total_lines} total lines")
        
        print(f"Trajectory validation passed:")
        print(f"  - Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"  - Average frame rate: {len(timestamps) / (timestamps[-1] - timestamps[0]):.1f} Hz")
        print(f"  - Position bounds: {positions.min(axis=0)} to {positions.max(axis=0)}")

    def __iter__(self):
        """Make this component iterable for pipeline usage."""
        # For trajectory data, we typically want to yield the complete trajectory
        # rather than individual poses, so we yield once
        yield {
            "camera_positions": self.camera_positions,
            "camera_quaternions": self.camera_quaternions,
            "timestamps": self.timestamps
        } 