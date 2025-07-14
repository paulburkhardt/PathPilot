"""
Interactive video player with synchronized warning signals.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import pygame
import threading
from queue import Queue
import time
from scipy.spatial.transform import Rotation
from gtts import gTTS
import tempfile
import os
from pydub import AudioSegment
from pydub.playback import play

# Constants for distance-based warning intervals (in meters : seconds)
DISTANCE_WARNING_MAP = {
    (0.0, 0.2): 0.0,    # Continuous beeping
    (0.2, 0.4): 0.1,    # 0.1s pause between beeps
    (0.4, 0.6): 0.4,    # 0.4s pause between beeps
    (0.6, 0.8): 0.8     # 0.8s pause between beeps
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

class AudioManager:
    """Handles all audio-related functionality."""
    def __init__(self):
        pygame.mixer.init()
        self.load_beep_sounds()
        self.audio_queue = Queue()
        self.is_playing = False
        self.audio_thread = None
        
    def load_beep_sounds(self):
        """Load or generate beep sounds for left and right warnings."""
        self.beep_left = pygame.mixer.Sound("static/left_beep.wav")
        self.beep_right = pygame.mixer.Sound("static/right_beep.wav")
        
    def get_beep_interval(self, distance: float) -> float:
        """Get the interval between beeps based on distance."""
        for (min_dist, max_dist), interval in DISTANCE_WARNING_MAP.items():
            if min_dist <= distance < max_dist:
                return interval
        return float('inf')  # No beeping for distances outside the ranges
        
    def play_warning(self, distance: float, is_left: bool):
        """Play warning beep with appropriate timing."""
        beep = self.beep_left if is_left else self.beep_right
        interval = self.get_beep_interval(distance)
        if interval == 0:  # Continuous beeping
            beep.play(-1)  # Loop indefinitely
        else:
            beep.play()
            time.sleep(interval)
            
    def speak_text(self, text: str):
        """Convert text to speech and play it."""
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            tts.save(f.name)
            sound = AudioSegment.from_mp3(f.name)
            play(sound)
            os.unlink(f.name)

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
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.is_playing = False
        self.current_frame = 0
        
    def run(self):
        """Main video playback loop."""
        cv2.namedWindow("Video Player")
        
        while True:
            if self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.process_frame(frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                self.is_playing = not self.is_playing
            elif key == ord('e'):  # Explain scene
                self.explain_scene()
                
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
            self.audio_manager.play_warning(closest_obj.closest_2d_distance, is_left)
            
        cv2.imshow("Video Player", frame)
        
    def explain_scene(self):
        """Generate and speak description of current scene."""
        if not self.is_playing:  # Only works when paused
            camera_pose, objects = self.data_manager.get_frame_data(self.current_frame)
            
            if camera_pose is not None and not objects.empty:
                descriptions = []
                for _, obj in objects.iterrows():
                    obj_pos = np.array([obj.closest_3d_x, obj.closest_3d_y, obj.closest_3d_z])
                    pos_cam = camera_pose.transform_point(obj_pos)
                    direction = "left" if pos_cam[0] < 0 else "right"
                    
                    descriptions.append(
                        f"A {obj.class_label.lower()} at {obj.closest_2d_distance:.1f} meters to your {direction}"
                    )
                
                scene_description = ". ".join(descriptions) + "."
                self.audio_manager.speak_text(scene_description)
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

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
