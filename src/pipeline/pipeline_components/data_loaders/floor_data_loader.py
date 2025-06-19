from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import json
import pathlib
from ..abstract_pipeline_component import AbstractPipelineComponent


class FloorDataLoader(AbstractPipelineComponent):
    """
    Component for loading pre-computed floor detection data from Phase 1 outputs.
    Supports both CSV and JSON formats.
    
    Args:
        floor_data_path: Path to the floor data file (CSV or JSON)
        data_format: Format of the data file ('auto', 'csv', or 'json')
        validate_data: Whether to validate the loaded data (default: True)
    
    Returns:
        Dictionary containing loaded floor detection results
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the data format is invalid or data validation fails
    """
    
    def __init__(self, floor_data_path: str, data_format: str = 'auto', 
                 validate_data: bool = True) -> None:
        super().__init__()
        self.floor_data_path = pathlib.Path(floor_data_path)
        self.data_format = data_format
        self.validate_data = validate_data

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component doesn't require any inputs from the bucket."""
        return []

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs pre-computed floor detection results."""
        return ["floor_normal", "floor_offset", "floor_data"]

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Load floor detection data from file.
        
        Returns:
            Dictionary containing floor detection results
        """
        if not self.floor_data_path.exists():
            raise FileNotFoundError(f"Floor data file not found: {self.floor_data_path}")
        
        # Determine data format
        if self.data_format == 'auto':
            if self.floor_data_path.suffix.lower() == '.json':
                format_type = 'json'
            elif self.floor_data_path.suffix.lower() == '.csv':
                format_type = 'csv'
            else:
                raise ValueError(f"Cannot auto-detect format for file: {self.floor_data_path}")
        else:
            format_type = self.data_format
        
        print(f"Loading floor detection data from: {self.floor_data_path}")
        print(f"Data format: {format_type}")
        
        # Load data based on format
        if format_type == 'json':
            results = self._load_json_data()
        elif format_type == 'csv':
            results = self._load_csv_data()
        else:
            raise ValueError(f"Unsupported data format: {format_type}")
        
        # Validate data if requested
        if self.validate_data:
            self._validate_data(results)
        
        # Print summary
        floor_normal = results["floor_normal"]
        floor_offset = results["floor_offset"]
        print(f"Loaded floor plane: normal={floor_normal}, offset={floor_offset:.3f}")
        
        return results

    def _load_json_data(self) -> Dict[str, Any]:
        """Load floor detection data from JSON file."""
        with open(self.floor_data_path, 'r') as f:
            data = json.load(f)
        
        # Extract floor parameters
        floor_normal = np.array(data["floor_normal"])
        floor_offset = float(data["floor_offset"])
        
        results = {
            "floor_normal": floor_normal,
            "floor_offset": floor_offset,
            "floor_data": data  # Keep original data for reference
        }
        
        return results

    def _load_csv_data(self) -> Dict[str, Any]:
        """Load floor detection data from CSV file."""
        df = pd.read_csv(self.floor_data_path)
        
        if len(df) == 0:
            raise ValueError("Floor data CSV file is empty")
        
        # Use the last row (most recent detection)
        last_row = df.iloc[-1]
        
        # Extract floor parameters
        floor_normal = np.array([
            last_row['normal_x'],
            last_row['normal_y'], 
            last_row['normal_z']
        ])
        floor_offset = float(last_row['offset'])
        
        results = {
            "floor_normal": floor_normal,
            "floor_offset": floor_offset,
            "floor_data": {
                "floor_normal": floor_normal.tolist(),
                "floor_offset": floor_offset,
                "detection_step": int(last_row['detection_step']) if 'detection_step' in last_row else None,
                "timestamp": float(last_row['timestamp']) if 'timestamp' in last_row else None
            }
        }
        
        return results

    def _validate_data(self, results: Dict[str, Any]) -> None:
        """Validate the loaded floor detection data."""
        if "floor_normal" not in results or "floor_offset" not in results:
            raise ValueError("Missing required floor detection data")
        
        floor_normal = results["floor_normal"]
        
        # Check floor normal is a 3D vector
        if len(floor_normal) != 3:
            raise ValueError("Floor normal should be a 3D vector")
        
        # Check floor normal is normalized (approximately)
        norm = np.linalg.norm(floor_normal)
        if abs(norm - 1.0) > 0.1:  # Allow some tolerance
            print(f"Warning: Floor normal is not normalized (norm={norm:.3f})")
        
        # Check floor offset is a valid number
        floor_offset = results["floor_offset"]
        if not isinstance(floor_offset, (int, float)) or np.isnan(floor_offset):
            raise ValueError("Floor offset should be a valid number")
        
        print("Floor data validation passed") 