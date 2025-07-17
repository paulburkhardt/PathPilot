"""
Interactive video player with synchronized warning signals.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
import pygame
import time
from scipy.spatial.transform import Rotation
from gtts import gTTS
import tempfile
import os
import threading
import speech_recognition as sr
import io
from openai import OpenAI

# Constants for distance-based warning intervals (in meters : seconds)
DISTANCE_WARNING_MAP = {
    (0.0, 0.8): 0.0,    # Continuous beeping
    (0.8, 1.0): 0.1,    # 0.1s pause between beeps
    (1.0, 1.5): 0.4,    # 0.4s pause between beeps
    (1.5, 2.0): 0.8     # 0.8s pause between beeps
}

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


class SpatialAIManager:
    """ 
    Allows to chat with the spatial data.
    """

    def __init__(self):

        #self.client = OpenAI(api_key = "")
        pass

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

        SYSTEM_PROMPT = """
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
        #                {"role": "system", "content": SYSTEM_PROMPT},
        #                {
        #                    "role": "user",
        #                    "content": "PROMPT: "+ prompt +"\n" +"CAMERA POSE: "+ camera_pose_as_text + "\n" + "OBJECTS: " + objects_as_text ,
        #                },
        #            ]
        #        )
        #output = response.output_text
        print("Done.")

        return output    




class SpeechRecognitionManager:
    """
    Handles all user input via microphone
    """

    DEBUG = False

    def __init__(self):
        
        self.speech_recognizer = sr.Recognizer()

    def record_audio(self)->Any:
        """
        Uses the microphone to record audio
        """

        with sr.Microphone() as source:
            print("Adjusting for noise...")
            self.speech_recognizer.adjust_for_ambient_noise(source)
            print("Done. Start speaking now.")
            audio = self.speech_recognizer.listen(source)
            print("Finished listening.")

        if self.DEBUG:
            wav_data = audio.get_wav_data()
            sound = pygame.mixer.Sound(file=io.BytesIO(wav_data))
            sound.play()

        return audio
    
    def transcribe_audio(self,audio)->str:
        """transcribes the audio"""

        try:
            # Recognize speech using Google Web Speech API
            text = self.speech_recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
    

class AudioManager:
    """
    Handles all audio-related functionality in a separate thread.
    
    Args:
        -
    Returns:
        -
    Raises:
        -
    """
    def __init__(self):
        """
        Initialize the AudioManager with sound loading and thread setup.
        
        Args:
            -
        Returns:
            -
        Raises:
            -
        """
        pygame.mixer.init()
        self.load_beep_sounds()
        self.audio_thread = None
        self.running = False
        
        # Current state
        self.beep = None
        self.interval = float('inf')
        self.last_beep_time = 0
        
    def load_beep_sounds(self):
        """
        Load or generate beep sounds for left and right warnings.
        
        Args:
            -
        Returns:
            -
        Raises:
            -
        """
        self.beep_left = pygame.mixer.Sound("static/left_beep.wav")
        self.beep_right = pygame.mixer.Sound("static/right_beep.wav")
        
    def get_beep_interval(self, distance: float) -> float:
        """
        Get the interval between beeps based on distance.
        
        Args:
            distance: Distance to the object in meters
        Returns:
            float: Time interval between beeps in seconds
        Raises:
            -
        """
        for (min_dist, max_dist), interval in DISTANCE_WARNING_MAP.items():
            if min_dist <= distance < max_dist:
                return interval
        return float('inf')  # No beeping for distances outside the ranges
        
    def set_state(self, distance: float, is_left: bool):
        """
        Update the warning state with new distance and direction.
        
        Args:
            distance: Distance to the object in meters
            is_left: True if object is on the left, False if on the right
        Returns:
            -
        Raises:
            -
        """
        self.beep = self.beep_left if is_left else self.beep_right
        self.interval = self.get_beep_interval(distance)
            
    def start(self):
        """
        Start the audio playback thread.
        
        Args:
            -
        Returns:
            -
        Raises:
            -
        """
        if not self.running:
            self.running = True
            self.audio_thread = threading.Thread(target=self._audio_loop)
            self.audio_thread.daemon = True  # Thread will stop when main program exits
            self.audio_thread.start()
            
    def stop(self):
        """
        Stop the audio playback thread.
        
        Args:
            -
        Returns:
            -
        Raises:
            -
        """
        self.running = False
        if self.audio_thread:
            self.audio_thread.join()
            
    def _audio_loop(self):
        """
        Main audio playback loop running in background thread.
        
        Args:
            -
        Returns:
            -
        Raises:
            -
        """
        while self.running:
             if self.beep is not None:
                if self.interval == float("inf"):
                    pass
                elif self.interval == 0.0:
                    self.beep.play()
                else:
                    self.beep.play()
                    time.sleep(self.interval)
            
    def speak_text(self, text: str):
        """
        Convert text to speech and play it.
        
        Args:
            text: The text to be converted to speech
        Returns:
            -
        Raises:
            -
        """
        # Create temp files in current directory
        temp_mp3 = tempfile.mktemp(suffix='.mp3', dir='.')
        try:
            # Generate and save speech
            tts = gTTS(text=text, lang='en')
            tts.save(temp_mp3)
            
            # Load and play with pygame
            pygame.mixer.music.load(temp_mp3)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        finally:
            # Clean up temp file
            try:
                os.remove(temp_mp3)
            except Exception:
                pass  # Ignore cleanup errors

class DataManager:
    """Manages trajectory and object data."""
    def __init__(self, data_dir: Path):
        self.trajectory = self.load_trajectory(data_dir / "incremental_analysis_detailed_trajectory.txt")
        self.objects_df = pd.read_csv(data_dir / "incremental_analysis_detailed_objects.csv")
        
    def load_trajectory(self, trajectory_file: Path) -> Dict[int, CameraPose]:
        """Load camera trajectory data."""
        trajectory = {}
        data = np.loadtxt(trajectory_file)
        
        for i, row in enumerate(data):
            position = row[1:4]
            quaternion = row[4:8]
            trajectory[i] = CameraPose.from_position_and_quaternion(position, quaternion)
        
        return trajectory
        
    def get_frame_data(self, frame_idx: int) -> Tuple[Optional[CameraPose], pd.DataFrame]:
        """Get camera pose and objects for a specific frame."""
        camera_pose = self.trajectory.get(frame_idx)
        frame_objects = self.objects_df[self.objects_df.step_nr == frame_idx]
        return camera_pose, frame_objects

class VideoPlayer:
    """Interactive video player with synchronized warnings."""
    def __init__(self, video_path: str, data_dir: Path):
        self.cap = cv2.VideoCapture(video_path)
        self.data_manager = DataManager(Path(data_dir))
        self.audio_manager = AudioManager()
        self.speech_recognition_manager = SpeechRecognitionManager()
        self.spatial_ai_manager = SpatialAIManager()
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_playing = False
        self.current_frame = 0
        
    def run(self):
        """Main video playback loop."""
        cv2.namedWindow("Video Player")
        frame_time = 1.0 / self.fps  # Time per frame in seconds
        last_frame_time = time.time()
        
        while True:
            current_time = time.time()
            if self.is_playing:
                # Check if enough time has passed for next frame
                if current_time - last_frame_time >= frame_time:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                        
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.process_frame(frame)
                    last_frame_time = current_time
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar

                #stop/start video player
                self.is_playing = not self.is_playing

                #stop/start audio player
                if self.audio_manager.running:
                    self.audio_manager.stop()
                else:
                    self.audio_manager.start()

            elif key == ord('e'):  # Explain scene
                self.explain_scene()

            elif key == ord("a"):
                audio = self.speech_recognition_manager.record_audio()
                text = self.speech_recognition_manager.transcribe_audio(audio)
                camera_pose, objects = self.data_manager.get_frame_data(self.current_frame)
                ai_answer = self.spatial_ai_manager.chat_with_spatial_data(text,camera_pose,objects)
                self.audio_manager.speak_text(ai_answer)

                
        self.cleanup()
        
    def process_frame(self, frame):
        """Process current frame and update warnings."""
        camera_pose, objects = self.data_manager.get_frame_data(self.current_frame)
        
        if camera_pose is not None and not objects.empty:
            # Find closest object and its direction
            closest_obj = objects.loc[objects.closest_2d_distance.idxmin()]
            obj_pos = np.array([closest_obj.closest_3d_x, 
                              closest_obj.closest_3d_y, 
                              closest_obj.closest_3d_z])
            
            # Get object position in camera coordinates
            obj_cam = camera_pose.transform_point(obj_pos)
            is_left = obj_cam[0] < 0  # x-coordinate determines left/right
            
            # Play appropriate warning sound
            self.audio_manager.set_state(closest_obj.closest_2d_distance, is_left)
            
        cv2.imshow("Video Player", frame)
        

    def _get_direction(self, vec_cam) -> str:
        """
        Determines the direction of the object relative to the camera position and orientation.

        Args:
            vec_cam: vector in camera frame
        Returns:
            Direction string: "front", "right", "left", "back", "front-right",
            "front-left", "back-left", or "back-right".
        """
        
        # In camera coordinates:
        # x points right
        # y points down
        # z points forward
        x, z = vec_cam[0], vec_cam[2]  # project onto xz plane
        angle = np.arctan2(x, z) * 180 / np.pi  # angle in degrees

        # Return direction based on angle
        if -22.5 <= angle < 22.5:
            return "front"
        elif 22.5 <= angle < 67.5:
            return "front-right"
        elif 67.5 <= angle < 112.5:
            return "right"
        elif 112.5 <= angle < 157.5:
            return "back-right"
        elif angle >= 157.5 or angle < -157.5:
            return "back"
        elif -157.5 <= angle < -112.5:
            return "back-left"
        elif -112.5 <= angle < -67.5:
            return "left"
        else:  # -67.5 <= angle < -22.5
            return "front-left"

        

    def explain_scene(self):
        """Generate and speak description of current scene."""
        if not self.is_playing:  # Only works when paused
            camera_pose, objects = self.data_manager.get_frame_data(self.current_frame)
            
            if camera_pose is not None and not objects.empty:
                descriptions = []
                for _, obj in objects.iterrows():
                    obj_pos = np.array([obj.closest_3d_x, obj.closest_3d_y, obj.closest_3d_z])
                    pos_cam = camera_pose.transform_point(obj_pos)
                    direction = self._get_direction(pos_cam)
                    
                    descriptions.append(
                        f"A {obj.class_label.lower()} at {obj.closest_2d_distance:.1f} meters to your {direction}"
                    )
                
                scene_description = ". ".join(descriptions) + "."
                self.audio_manager.speak_text(scene_description)
            else:
                self.audio_manager.speak_text("There are no objects in close proximity.")
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.audio_manager.stop()  # Stop the audio thread

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive video player with proximity warnings")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("data_dir", help="Path to directory containing trajectory and object data")
    
    args = parser.parse_args()
    
    player = VideoPlayer(args.video_path, args.data_dir)
    player.run()

if __name__ == "__main__":
    main()
