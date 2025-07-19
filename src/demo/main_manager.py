from pathlib import Path
import numpy as np

from video_player import VideoPlayer
from data_manager import DataManager
from audio_player import AudioPlayer
from speech_recognizer import SpeechRecognizer
from spatial_ai_manager import SpatialAIManager

class MainManager():

    def __init__(self):

        self.video_player = VideoPlayer(self)
        self.data_manager = DataManager()
        self.audio_player = AudioPlayer()
        self.speech_recognizer = SpeechRecognizer()
        self.spatial_ai_manager = SpatialAIManager()

        self.last_processed_frame_idx = None
        self.is_playing = False

    @property
    def step_idx(self):
        """
        The Video player is the determining factor for the current step
        """
        return self.video_player.step_idx
    
    @property
    def camera_pose(self):
        return self.data_manager.get_frame_data(self.step_idx)[0]
    
    @property
    def objects(self):
        return self.data_manager.get_frame_data(self.step_idx)[1]

    def load_data(
            self,
            data_dir:Path
        ):
        
        try:
            self.data_manager.load_data(data_dir=data_dir)
            return True
        except:
            return False

    def load_video(
        self,
        video_file:Path
    ):
        self.video_player.load_video(video_file)

    def _start_players(self):
        self.video_player.start()
        self.audio_player.start()

    def _stop_players(self):
        self.video_player.stop()
        self.audio_player.stop()

    def start(self):
        
        if not self.is_playing:
            self._start_players()
            self.is_playing = True
        else:
            print("Tried to start while running!")

    def stop(self):
        if self.is_playing:
            self._stop_players()
        self.is_playing = False
        print("Stopped.")

    def process_frame(self):
        """
        Process the current frame
        """

        camera_pose = self.camera_pose
        objects = self.objects
        if camera_pose is not None and not objects.empty:
            # Find closest object and its direction
            closest_obj = objects.loc[objects.closest_2d_distance.idxmin()]
            obj_pos = np.array([closest_obj.closest_3d_x, 
                              closest_obj.closest_3d_y, 
                              closest_obj.closest_3d_z])
            
            # Get object position in camera coordinates
            obj_cam = camera_pose.world_to_cam(obj_pos)
            is_left = obj_cam[0] < 0  # x-coordinate determines left/right
            
            # Play appropriate warning sound
            self.audio_player.set_state(closest_obj.closest_2d_distance, is_left)
        else:
            self.audio_player.set_state(float("inf"),True)

    def record_user_audio_and_transcribe(
        self
    )->str:
        return self.speech_recognizer.record_and_transcribe()

    def chat_with_spatial_data(
        self,
        prompt:str
    ):
        
        return self.spatial_ai_manager.chat_with_spatial_data(
            prompt=prompt,
            camera_pose=self.camera_pose,
            objects=self.objects
        )

    def explain_scene(self):
        """Generate and speak description of current scene."""
        camera_pose, objects = self.camera_pose,self.objects
        
        if camera_pose is not None and not objects.empty:
            descriptions = []
            for _, obj in objects.iterrows():
                
                obj_pos = np.array([obj.closest_3d_x, obj.closest_3d_y, obj.closest_3d_z])
                direction = camera_pose.world_to_string_direction(
                    obj_pos
                )
                
                descriptions.append(
                    f"A {obj.class_label.lower()} at {obj.closest_2d_distance:.1f} meters to your {direction}"
                )
            
            scene_description = ". ".join(descriptions) + "."
            self.audio_player.speak_text(scene_description)
        else:
            self.audio_player.speak_text("There are no objects in close proximity.")