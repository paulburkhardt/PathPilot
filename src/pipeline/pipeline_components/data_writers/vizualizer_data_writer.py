from typing import List, Dict, Any
import os
import numpy as np
from datetime import datetime
from .abstract_data_writer import AbstractDataWriter
import rerun as rr

class VizualizerDataWriter(AbstractDataWriter):
    """
    Writer component for saving visualization data to RRD files.
    
    Args:
        output_dir: Directory where visualization files will be written.
    
    Returns:
        -
    
    Raises:
        ValueError: If input visualization data is invalid
    """
    
    def __init__(self, output_dir: str = None) -> None:
        super().__init__(output_dir)
        # Create timestamped parent directory
        if self._output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._timestamped_output_dir = os.path.join(self._output_dir, f"run_{timestamp}")
        else:
            self._timestamped_output_dir = None
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component reads visualization data."""
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
        Write visualization data to an RRD file.
        
        Args:
            step_nr: Step number within the pipeline
            vizualizer: Rerun viewer object to save
        Returns:
            Empty dictionary as no data is added to bucket
        Raises:
            ValueError: If vizualizer is not provided
        """
        if vizualizer is None:
            raise ValueError("vizualizer must be provided")
            
        # Create output directory with timestamp if it doesn't exist
        base_output_dir = self._timestamped_output_dir if self._timestamped_output_dir else self.output_dir
        output_dir = os.path.join(base_output_dir, f"step_{step_nr}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "vizualizer.rrd")
        rr.save(output_path)
        
        return {}
