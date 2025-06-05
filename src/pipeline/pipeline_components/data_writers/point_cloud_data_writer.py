from typing import List, Dict, Any
import os
import numpy as np
from .abstract_data_writer import AbstractDataWriter

class PointCloudDataWriter(AbstractDataWriter):
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
        return ["step_nr","point_cloud"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component doesn't add anything to the bucket."""
        return []
    
    def _run(
        self, 
        step_nr: int,
        point_cloud: np.ndarray,
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
        if not isinstance(point_cloud, np.ndarray) or point_cloud.shape[1] != 3:
            raise ValueError("point_cloud must be a Nx3 numpy array")
            
        output_path = os.path.join(self.output_dir,f"step_{step_nr}", "pointcloud.ply")
        
        # Write PLY file
        with open(output_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            
            # Write vertices
            for point in point_cloud:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        
        return {}
