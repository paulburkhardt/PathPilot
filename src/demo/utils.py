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

    def world_to_cam(
        self,
        point_world
    ):
        return self.rotation.inv().apply(point_world-self.position)


    def cam_to_world(
        self,
        point_cam
    ):
        return self.rotation.apply(point_cam) + self.position
    

    def world_to_string_direction(
            self, 
            point_world
    ):
        point_cam = self.world_to_cam(point_world)

        x, y, z = point_cam  # In camera coordinates

        # Ignore vertical direction (y), only use x (right) and z (forward)
        angle = np.arctan2(x, z)  # angle in radians, where:
                                # x = right, z = forward

        angle_deg = np.degrees(angle)
        # Normalize to [0, 360)
        angle_deg = (angle_deg + 360) % 360

        # Divide into 8 zones (45° each)
        if 337.5 <= angle_deg or angle_deg < 22.5:
            return "front"
        elif 22.5 <= angle_deg < 67.5:
            return "front-right"
        elif 67.5 <= angle_deg < 112.5:
            return "right"
        elif 112.5 <= angle_deg < 157.5:
            return "back-right"
        elif 157.5 <= angle_deg < 202.5:
            return "back"
        elif 202.5 <= angle_deg < 247.5:
            return "back-left"
        elif 247.5 <= angle_deg < 292.5:
            return "left"
        elif 292.5 <= angle_deg < 337.5:
            return "front-left"

