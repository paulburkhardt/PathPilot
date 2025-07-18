from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

from utils import CameraPose


class DataManager():
    """Manages trajectory and object data."""

    def __init__(self):

        self.trajectory = None
        self.objects_df = None

        self.data_loaded = False

    def load_data(self,data_dir:Path):
        data_dir = Path(data_dir)
        self.trajectory = self.load_trajectory(data_dir / "incremental_analysis_detailed_trajectory.txt")
        self.objects_df = pd.read_csv(data_dir / "incremental_analysis_detailed_objects.csv")

        self.data_loaded = True

    def load_trajectory(self, trajectory_file: Path) -> Dict[int, CameraPose]:
        """Load camera trajectory data."""
        trajectory = {}
        data = np.loadtxt(trajectory_file)
        
        for i, row in enumerate(data):
            position = row[1:4]
            quaternion = row[4:8]
            trajectory[i] = CameraPose.from_position_and_quaternion(position, quaternion)
        
        return trajectory
        
    def get_frame_data(self, frame_idx: int):
        """Get camera pose and objects for a specific frame."""
        
        camera_pose = self.trajectory.get(frame_idx)
        frame_objects = self.objects_df[self.objects_df.step_nr == frame_idx]
        return camera_pose, frame_objects
