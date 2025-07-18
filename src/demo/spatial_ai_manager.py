from typing import List, Any
from dotenv import load_dotenv
import os
from openai import OpenAI

from utils import CameraPose

class SpatialAIManager:
    """ 
    Allows to chat with the spatial data.
    """

    def __init__(self):

        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file.")
        
        self.client = OpenAI(api_key = self.api_key)
        
    def chat_with_spatial_data(
        self,
        prompt:str,
        camera_pose: CameraPose,
        objects: List[Any]
    )-> str:
        """
        Executes the prompt against the spatial data.

        Args:
            prompt: prompt to execute
            camera_pose: pose of the camera
            objects: Objects in the frame

        Returns:
            Answer by spatialAi 
        """

        system_prompt = """
        You are a helpful assistant to a visually impaired person. You are going to help the visually impaired person to understand his environment based on the given data.
        You will be provided a camera pose(translation + rotation) and a list of objects:
        {closest_3d_x: <x coordinate of the closest point of the object>,closest_3d_y: <y coordinate of the closest point of the object>,closest_3d_z: <z coordinate of the closest point of the object>,closest_2d_distance: <distance to that object in 2d (this is the relevant distance here)>,class_label: <Label describing the class of the object>}
        {position: <x,y,z coordinates of the camera>, rotation: <quaternion of the camera rotation>}

        and the visually impaired person is going to ask questions concerning this data. 
        Please answer truthfully and based on the provided objects and camera pose. Please also answer concisely with at most 100 characters.

        example:
            PROMPT: "I am very exhausted and would like to sit down. Can you guide me?"
            CAMERA_POSE: {position: [0.2,0.7,0.5], rotation: [0.7071, 0.0, 0.7071, 0.0]}
            OBJECTS: [
                {closest_3d_x: 1.2,closest_3d_y: 0.7, closest_3d_z: 0.9,closest_2d_distance: 0.93,class_label: Table}
                {closest_3d_x: 1.5,closest_3d_y: 0.68,closest_3d_z: 1.1,closest_2d_distance: 1.3, class_label: Chair}
            ]
            
            OUTPUT: "Sure, I will guide you! There is a chair to your at 1.3 meters to your front-right. But be careful, there also is a table in the same direction at a closer distance."
        """

        # Convert the dataset to a list of dicts matching the required structure
        objects_as_text = "[]"
        if len(objects):
            objects_as_text = "\n".join([
                "{closest_3d_x: "+ str(row.closest_3d_x) + ", closest_3d_y: "+ str(row.closest_3d_y)+ ", closest_3d_z: "+ str(row.closest_3d_z)+ ", closest_2d_distance: " + str(row.closest_2d_distance) + ", class_label: " +str(row.class_label) + "}"
                for _,row in objects.iterrows()
            ])
        camera_pose_as_text = "{}"
        if camera_pose is not None:
            camera_pose_as_text = "{position: "+ str(camera_pose.position.tolist())+", quaternion: "+ str(camera_pose.rotation.as_quat().tolist()) + " }"

        print("Asking Spatial AI...")
        output = "Dummy output"
        #response = self.client.responses.parse(
        #            model="gpt-4o-2024-08-06",
        #            input=[
        #                {"role": "system", "content": system_prompt},
        #                {
        #                    "role": "user",
        #                    "content": "PROMPT: "+ prompt +"\n" +"CAMERA POSE: "+ camera_pose_as_text + "\n" + "OBJECTS: " + objects_as_text ,
        #                },
        #            ]
        #        )
        #output = response.output_text
        print("Done.")

        return output