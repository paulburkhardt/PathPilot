from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import numpy as np

@dataclass
class CameraPose:
    """Camera pose with position and orientation."""
    position: np.ndarray
    rotation: Rotation
    
    @classmethod
    def from_position_and_quaternion(cls, position: np.ndarray, quaternion: np.ndarray):
        return cls(
            position=position,
            rotation=Rotation.from_quat(quaternion)
        )
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        vec_world = point - self.position
        return self.rotation.inv().apply(vec_world)