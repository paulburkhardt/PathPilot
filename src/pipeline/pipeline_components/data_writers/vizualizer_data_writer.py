from typing import List, Dict, Any
import os
import numpy as np
from .abstract_data_writer import AbstractDataWriter
import rerun as rr

class VizualizerDataWriter(AbstractDataWriter):
    """
    Writer component for saving point cloud data to PLY files.
    
    Args:
        output_dir: Directory where point cloud files will be written.
    
    Returns:
        -
    
    Raises:
        ValueError: If input point cloud data is invalid
    """
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component reads point cloud data."""
        return []
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component doesn't add anything to the bucket."""
        return []
    
    def _run(
        self, 
        step_nr: int,
        vizualizer: rr.Viewer,
    ) -> Dict[str, Any]:
        """
        Write point cloud data to a PLY file.
        
        Args:
            step_nr: Step number within the pipeline
            point_cloud: Nx3 numpy array of point cloud coordinates
        Returns:
            Empty dictionary as no data is added to bucket
        Raises:
            ValueError: If point_cloud is not a valid Nx3 array
        """
        if vizualizer is None:
            raise ValueError("vizualizer must be provided")
            
        output_path = os.path.join(self.output_dir,f"step_{step_nr}", "vizualizer.rrd")
        rr.save(output_path)
        
        return {}
