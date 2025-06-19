from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import json
import pathlib
from ..abstract_pipeline_component import AbstractPipelineComponent


class ClosestPointsDataLoader(AbstractPipelineComponent):
    """
    Component for loading pre-computed closest points data from Phase 1 outputs.
    Supports both CSV and JSON formats.
    
    Args:
        closest_points_path: Path to the closest points data file (CSV or JSON)
        data_format: Format of the data file ('auto', 'csv', or 'json')
        validate_data: Whether to validate the loaded data (default: True)
    
    Returns:
        Dictionary containing loaded closest point analysis results
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the data format is invalid or data validation fails
    """
    
    def __init__(self, closest_points_path: str, data_format: str = 'auto', 
                 validate_data: bool = True) -> None:
        super().__init__()
        self.closest_points_path = pathlib.Path(closest_points_path)
        self.data_format = data_format
        self.validate_data = validate_data

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component doesn't require any inputs from the bucket."""
        return []

    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs pre-computed closest point analysis results."""
        return [
            "closest_point_3d", "closest_point_index", "distance_3d", 
            "distances_array", "closest_point_floor", "distance_floor", 
            "projected_point", "floor_distances_array", "view_cone_mask"
        ]

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Load closest points data from file.
        
        Returns:
            Dictionary containing closest point analysis results
        """
        if not self.closest_points_path.exists():
            raise FileNotFoundError(f"Closest points data file not found: {self.closest_points_path}")
        
        # Determine data format
        if self.data_format == 'auto':
            if self.closest_points_path.suffix.lower() == '.json':
                format_type = 'json'
            elif self.closest_points_path.suffix.lower() == '.csv':
                format_type = 'csv'
            else:
                raise ValueError(f"Cannot auto-detect format for file: {self.closest_points_path}")
        else:
            format_type = self.data_format
        
        print(f"Loading closest points data from: {self.closest_points_path}")
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
        num_poses = len(results["distances_array"])
        print(f"Loaded closest points data for {num_poses} camera poses")
        print(f"3D distance range: {results['distances_array'].min():.3f} to {results['distances_array'].max():.3f}m")
        
        if "floor_distances_array" in results and len(results["floor_distances_array"]) > 0:
            print(f"Floor distance range: {results['floor_distances_array'].min():.3f} to {results['floor_distances_array'].max():.3f}m")
        
        return results

    def _load_json_data(self) -> Dict[str, Any]:
        """Load closest points data from JSON file."""
        with open(self.closest_points_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists to numpy arrays
        closest_points_3d = np.array(data["closest_points_3d"])
        distances_3d = np.array(data["distances_3d"])
        
        results = {
            "closest_point_3d": closest_points_3d,
            "distance_3d": distances_3d,
            "distances_array": distances_3d
        }
        
        # Load floor data if available
        if "closest_points_floor" in data and "distances_floor" in data:
            closest_points_floor = np.array(data["closest_points_floor"])
            distances_floor = np.array(data["distances_floor"])
            
            results.update({
                "closest_point_floor": closest_points_floor,
                "distance_floor": distances_floor,
                "projected_point": closest_points_floor,  # Same as closest_point_floor
                "floor_distances_array": distances_floor
            })
        
        # Generate dummy indices (not saved in JSON format)
        results["closest_point_index"] = np.arange(len(distances_3d))
        
        # Generate dummy view cone mask (not saved in JSON format)
        # Since we don't have the original view cone data, provide None for each pose
        results["view_cone_mask"] = [None] * len(distances_3d)
        
        return results

    def _load_csv_data(self) -> Dict[str, Any]:
        """Load closest points data from CSV file."""
        df = pd.read_csv(self.closest_points_path)
        
        # Extract 3D closest points and distances
        closest_points_3d = df[['closest_3d_x', 'closest_3d_y', 'closest_3d_z']].values
        distances_3d = df['distance_3d'].values
        
        results = {
            "closest_point_3d": closest_points_3d,
            "distance_3d": distances_3d,
            "distances_array": distances_3d,
            "closest_point_index": df['step'].values  # Use step numbers as indices
        }
        
        num_poses = len(distances_3d)
        
        # Load floor data if available
        floor_columns = ['closest_floor_x', 'closest_floor_y', 'closest_floor_z', 'distance_floor']
        if all(col in df.columns for col in floor_columns):
            # Filter out rows where floor data is missing (NaN)
            floor_mask = df[floor_columns].notna().all(axis=1)
            
            if floor_mask.any():
                closest_points_floor = df.loc[floor_mask, ['closest_floor_x', 'closest_floor_y', 'closest_floor_z']].values
                distances_floor = df.loc[floor_mask, 'distance_floor'].values
                
                # For compatibility, we need to align floor data with 3D data
                # If some poses don't have floor data, fill with NaN arrays
                full_closest_points_floor = np.full((num_poses, 3), np.nan)
                full_distances_floor = np.full(num_poses, np.nan)
                
                floor_indices = df.index[floor_mask]
                full_closest_points_floor[floor_indices] = closest_points_floor
                full_distances_floor[floor_indices] = distances_floor
                
                results.update({
                    "closest_point_floor": full_closest_points_floor,
                    "distance_floor": full_distances_floor,
                    "projected_point": full_closest_points_floor,  # Same as closest_point_floor
                    "floor_distances_array": full_distances_floor
                })
        
        # Generate dummy view cone mask (not saved in CSV format)
        # Since we don't have the original view cone data, provide None for each pose
        results["view_cone_mask"] = [None] * num_poses
        
        return results

    def _validate_data(self, results: Dict[str, Any]) -> None:
        """Validate the loaded data for consistency."""
        if "distances_array" not in results or len(results["distances_array"]) == 0:
            raise ValueError("No valid closest points data found")
        
        num_poses = len(results["distances_array"])
        
        # Check 3D data consistency
        if len(results["closest_point_3d"]) != num_poses:
            raise ValueError("Mismatch between number of closest points and distances")
        
        if results["closest_point_3d"].shape[1] != 3:
            raise ValueError("Closest points should have 3 coordinates (x, y, z)")
        
        # Check floor data consistency if present
        if "floor_distances_array" in results and len(results["floor_distances_array"]) > 0:
            if len(results["floor_distances_array"]) != num_poses:
                raise ValueError("Mismatch between 3D and floor distance arrays")
            
            if "closest_point_floor" in results and len(results["closest_point_floor"]) != num_poses:
                raise ValueError("Mismatch between floor closest points and distances")
        
        print("Data validation passed") 