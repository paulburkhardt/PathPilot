from typing import List, Dict, Any
import os
import numpy as np
from .abstract_data_writer import AbstractDataWriter
import pickle


class DatabaseDataWriter(AbstractDataWriter):
    """
    Writer component for saving point database to PLY files.
    
    Args:
        output_dir: Directory where point cloud files will be written.
    
    Returns:
        -
    
    Raises:
        ValueError: If input point database is invalid
    """
    
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component reads point database."""
        return ["database"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component doesn't add anything to the bucket."""
        return []
    
    def _run(
        self, 
        step_nr: int,
        database,
        ) -> Dict[str, Any]:
        """
        Write point cloud data to a PLY file.
        
        Args:
            step_nr: Step number within the pipeline
            database: PointCloudDataEntity or Nx3 numpy array of point cloud coordinates
        Returns:
            Empty dictionary as no data is added to bucket
        Raises:
            ValueError: If database is not a valid format
        """

        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.output_dir, f"step_{step_nr}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "rtree_index_.pkl")

        try:
            with open(output_path, 'wb') as f:
                pickle.dump(database, f)
            print(f"R-tree index saved to {output_path}")
        except Exception as e:
            print(f"Error saving R-tree index: {e}")
        
        return {}