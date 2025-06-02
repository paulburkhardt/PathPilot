import numpy as np
from typing import Dict, Any
from scipy.spatial import cKDTree
from lietorch import SE3
from utils import L2_distance, calculate_angle

class Object3D:
    def __init__(self, obj_id, position, scale = None, description = None):
        self.id = obj_id
        self.position = np.array(position)
        self.description = description
        self.scale = scale 



class DataBase:
    def __init__(self, config: Dict[str, Any]):
        self.tree = None
        self.objects = []
        self.radius = config['radius']

    def add_object(self, object: Object3D):
        self.objects.append(object)
        self.tree = cKDTree(self.objects,3)

    def get_data(self, camera_pose: SE3):
        camera_forward = camera_pose.rotation() @ np.array([0, 0, self.radius])  # or [0, 0, -1] depending on convention
        camera_position = camera_pose.translation()
        position_of_interest = camera_forward + camera_position
        # Convert camera forward to homogeneous coordinates
        distances, indices = self.tree.query(camera_forward,k=10)
        nearest_objects = []
        distances_to_camera = []
        position_descriptions = []
        for i in indices:
            object = self.objects[i]
            if L2_distance(object.position, position_of_interest) > self.radius:
                angle_to_camera = calculate_angle(camera_forward, )
                distance_to_camera = L2_distance(object.position, camera_position)
                position_description = self.describe_position(camera_position, 
                                                              position_of_interest, 
                                                              object.position)
                nearest_objects.append(object)
        distances_to_camera.sort()
        nearest_objects = nearest_objects[distances_to_camera.index(distances_to_camera[0])]

        return nearest_objects, distances_to_camera
    
    def get_direction(self, camera_pose: SE3, object_position: np.ndarray):
        camera_to_object = object_position - camera_pose.translation()
        camera_to_object = camera_to_object / np.linalg.norm(camera_to_object)
        camra_right = camera_pose.rotation() @ np.array([1, 0, 0])
        camera_forward = camera_pose.rotation() @ np.array([0, 0, 1])
        x = np.dot(camera_to_object, camera_forward)
        y = np.dot(camera_to_object, camra_right)
        angle_deg = np.degrees(np.arctan2(y, x))
        return angle_deg
    
 
