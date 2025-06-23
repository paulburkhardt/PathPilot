from typing import List, Dict, Any
import os
import numpy as np
from .abstract_data_writer import AbstractDataWriter
from src.pipeline.data_entities.point_cloud_data_entity import PointCloudDataEntity

from plyfile import PlyData, PlyElement

class PointCloudDataWriter(AbstractDataWriter):
    """
    Writer component for saving point cloud data to PLY files.
    
    Args:
        output_dir: Directory where point cloud files will be written.
        only_save_last: Only saves the output if its the last frame
    Returns:
        -
    
    Raises:
        ValueError: If input point cloud data is invalid
    """

    def __init__(
            self, 
            output_dir: str, 
            only_save_last: bool = False
        ):

        super().__init__(output_dir)
        self.only_save_last = only_save_last
    
    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component reads point cloud data."""
        return ["step_nr","total_steps","point_cloud"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component doesn't add anything to the bucket."""
        return []
    
    def _save_ply(
            self,
            filename:str, 
            points:np.ndarray, 
            colors:np.ndarray
        ):
        """
        Save points and colors as .ply file

        Args:
            filename: Path to the output PLY file.
            points: Nx3 numpy array of point coordinates.
            colors: Nx3 numpy array of RGB color values (0-255).
        """

        colors = colors.astype(np.uint8)
        # Combine XYZ and RGB into a structured array
        pcd = np.empty(
            len(points),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        pcd["x"], pcd["y"], pcd["z"] = points.T
        pcd["red"], pcd["green"], pcd["blue"] = colors.T
        vertex_element = PlyElement.describe(pcd, "vertex")
        ply_data = PlyData([vertex_element], text=False)
        ply_data.write(filename)

    def _run(
        self, 
        step_nr: int,
        total_steps: int,
        point_cloud: PointCloudDataEntity,
    ) -> Dict[str, Any]:
        """
        Write point cloud data to a PLY file.
        
        Args:
            step_nr: Step number within the pipeline
            total_steps: total number of steps
            point_cloud: Point cloud data entity
        Returns:
            Empty dictionary as no data is added to bucket
        Raises:
            ValueError: If point_cloud is not a valid Nx3 array
        """

        save = True    
        if self.only_save_last:
            if step_nr != total_steps-1:
                save = False

        if save:

            output_dir = os.path.join(self.output_dir,f"step_{step_nr}")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, "pointcloud.ply")

            points_and_colors = point_cloud.as_numpy(with_rgb=True)
            points = points_and_colors[:,0:3]
            colors = points_and_colors[:,3:]
            self._save_ply(output_path,points,colors)

        return {}