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


        # Create direction description
        direction_parts = []
        
        # Front/Back (Z-axis)
        has_front_back = False
        if abs(z) > 0.1:  # Threshold to avoid noise
            has_front_back = True
            if z > 0:
                direction_parts.append("ahead")
            else:
                direction_parts.append("behind")
        
        # Left/Right (X-axis)
        has_left_right = False
        if abs(x) > 0.1:
            has_left_right = True
            if x > 0:
                direction_parts.append("right")
            else:
                direction_parts.append("left")
        
        # Up/Down (Y-axis)
        has_up_down = False
        if abs(y) > 0.1:
            has_up_down = True
            if y > 0:
                direction_parts.append("below")
            else:
                direction_parts.append("above")
        
        # Combine directions intelligently
        if has_left_right and has_front_back and has_up_down:
            return "-".join(direction_parts)
        elif len(direction_parts)==2:
            return "-".join(direction_parts)
        elif len(direction_parts)==1:
            return direction_parts[0]
        else:
            return "at almost exactly your location"

        if len(direction_parts) == 0:
            return "directly in front"
        elif len(direction_parts) == 1:
            return f"to your {direction_parts[0]}"
        elif len(direction_parts) == 2:
            # For two directions, combine them naturally
            if "ahead" in direction_parts or "behind" in direction_parts:
                front_back = next((d for d in direction_parts if d in ["ahead", "behind"]), "")
                other = next((d for d in direction_parts if d not in ["ahead", "behind"]), "")
                return f"{front_back}-{other}"
            else:
                return f"{direction_parts[0]}-{direction_parts[1]}"
        else:
            # Three directions - combine all
            return f"{direction_parts[0]}-{direction_parts[1]}-{direction_parts[2]}"



        #OLD DEPRECATED
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

