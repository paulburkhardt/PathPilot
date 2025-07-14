"""
Generate warning videos from SLAM analysis CSV output.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from gtts import gTTS
import tempfile
import shutil
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

@dataclass
class CameraPose:
    """
    Represents a camera pose with position and orientation.
    
    Args:
        position: 3D position vector
        rotation: Rotation object from scipy.spatial.transform
    """
    position: np.ndarray
    rotation: Rotation
    
    @classmethod
    def from_position_and_quaternion(cls, position: np.ndarray, quaternion: np.ndarray):
        """
        Create a CameraPose from position and quaternion (x,y,z,w format).
        """
        return cls(
            position=position,
            rotation=Rotation.from_quat(quaternion)
        )
    
    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """
        Transform a point from world coordinates to camera coordinates.
        """
        # Vector from camera to point in world coordinates
        vec_world = point - self.position
        # Transform to camera coordinates
        return self.rotation.inv().apply(vec_world)

def load_trajectory(trajectory_file: str) -> Dict[int, CameraPose]:
    """
    Load camera trajectory from enhanced SLAM output.
    
    Args:
        trajectory_file: Path to the trajectory file
    Returns:
        Dictionary mapping steps to CameraPose objects
    Raises:
        FileNotFoundError: If trajectory file doesn't exist
    """
    trajectory = {}
    data = np.loadtxt(trajectory_file)
    
    for i, row in enumerate(data):
            
        position = row[1:4]  # x, y, z
        quaternion = row[4:8]  # qx, qy, qz, qw
        
        # Create CameraPose object
        trajectory[i] = CameraPose.from_position_and_quaternion(position, quaternion)
    
    return trajectory

class ObjectWarningGenerator:
    """
    Generates warning messages based on detected objects from CSV data.

    Args:
        warning_cooldown: Number of frames to wait before repeating a warning.
        critical_distance: Maximum distance to generate warnings for objects.
                         Objects further than this will be ignored.
    """
    def __init__(self, warning_cooldown: int = 30, critical_distance: float = float('inf')) -> None:
        self.warning_cooldown = warning_cooldown
        self.critical_distance = critical_distance
        # Dictionary to track when each object was last warned about
        # Key: (class_label, direction), Value: frame index of last warning
        self.warning_buffer = {}

    def _get_direction(self, object_pos: np.ndarray, camera_pose: CameraPose) -> str:
        """
        Determines the direction of the object relative to the camera position and orientation.

        Args:
            object_pos: 3D position of the object in world coordinates.
            camera_pose: Camera pose with position and orientation.
        Returns:
            Direction string: "front", "right", "left", "back", "front-right",
            "front-left", "back-left", or "back-right".
        """
        # Transform object position to camera coordinates
        vec_cam = camera_pose.transform_point(object_pos)
        
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

    def process_objects(self, objects_df: pd.DataFrame, frame_idx: int, frame_camera_pose: CameraPose) -> str:
        """
        Process all objects in the current frame and generate a warning message.
        
        Args:
            objects_df: DataFrame containing objects for the current frame
            frame_idx: Current frame index
            frame_camera_pose: Camera pose for that frame
        Returns:
            Warning message as string
        """
        if objects_df.empty:
            return ""

        objects_to_warn = []
        for _, row in objects_df.iterrows():
            segment_id = row.segment_id
            distance = row.closest_2d_distance

            if distance > self.critical_distance:
                continue

            last_warning = self.warning_buffer.get(segment_id, -self.warning_cooldown)
            if frame_idx - last_warning >= self.warning_cooldown:
                self.warning_buffer[segment_id] = frame_idx
                object_pos = np.array([row.closest_3d_x, row.closest_3d_y, row.closest_3d_z])
                direction = self._get_direction(object_pos, frame_camera_pose)
                objects_to_warn.append((row.class_label, distance, direction))

        if not objects_to_warn:
            return ""

        messages = []
        for obj, dist, direction in objects_to_warn:
            messages.append(f"There is a {obj.lower()} at {dist:.2f} meters to your {direction}.")
        return " ".join(messages)

class AudioOverlay:
    """
    Converts warning messages to audio and overlays them onto the video.

    Args:
        -
    Returns:
        -
    Raises:
        -
    """
    def __init__(self) -> None:
        pass

    def warnings_to_audio(self, warnings: List[Tuple[int, str]], video_fps: float, video_duration: float) -> str:
        """
        Converts warnings to a single audio file, timed to video.

        Args:
            warnings: List of (frame_idx, message).
            video_fps: Frames per second of the video.
            video_duration: Duration of the video in seconds.
        Returns:
            Path to generated audio file.
        Raises:
            -
        """
        temp_dir = tempfile.mkdtemp()
        audio_segments = []
        last_end = 0.0
        for frame_idx, msg in warnings:
            t = frame_idx / video_fps
            audio_path = os.path.join(temp_dir, f"audio_{frame_idx}.wav")
            tts = gTTS(text=msg, lang='en')
            mp3_path = os.path.join(temp_dir, f"audio_{frame_idx}.mp3")
            tts.save(mp3_path)
            # Convert mp3 to wav using pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(audio_path, format="wav")
            audio_segments.append((t, audio_path))
        # Concatenate audio segments with silence in between
        from pydub import AudioSegment
        final_audio = AudioSegment.silent(duration=int(video_duration * 1000))
        for t, audio_path in audio_segments:
            seg = AudioSegment.from_wav(audio_path)
            start_ms = int(t * 1000)
            final_audio = final_audio.overlay(seg, position=start_ms)
        out_path = os.path.join(temp_dir, "final_audio.wav")
        final_audio.export(out_path, format="wav")
        return out_path

    def overlay_audio_on_video(self, video_path: str, audio_path: str, output_path: str) -> None:
        """
        Overlays the audio onto the video.

        Args:
            video_path: Path to original video.
            audio_path: Path to generated audio.
            output_path: Path to save the modified video.
        Returns:
            -
        Raises:
            ValueError: If output_path doesn't have a valid video extension
        """
        # Ensure output path has a video extension
        if not any(output_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov']):
            output_path = output_path + '.mp4'  # Add default extension if none provided

        try:
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            
            # Create a composite video clip and set its audio
            final_clip = CompositeVideoClip([video])
            final_clip.audio = audio
            
            # Write the result with the appropriate codec based on extension
            ext = os.path.splitext(output_path)[1].lower()
            if ext == '.mp4':
                codec = 'libx264'
                audio_codec = 'aac'
            elif ext == '.avi':
                codec = 'mpeg4'
                audio_codec = 'mp3'
            elif ext == '.mkv':
                codec = 'libx264'
                audio_codec = 'libvorbis'
            else:  # .mov or others
                codec = 'libx264'
                audio_codec = 'aac'

            final_clip.write_videofile(
                output_path,
                codec=codec,
                audio_codec=audio_codec,
                preset='medium',
                ffmpeg_params=['-strict', '-2']
            )
        finally:
            # Clean up
            try:
                video.close()
            except:
                pass
            try:
                audio.close()
            except:
                pass
            try:
                final_clip.close()
            except:
                pass

def main(
    video_path: str,
    analysis_csv: str,
    trajectory_txt: str,
    output_video_path: str,
    warning_cooldown: int = 30,
    frame_subsample: int = 1,
    critical_distance: float = float('inf')  # Now optional, filters by distance if provided
) -> None:
    """
    Generate warning video from SLAM analysis CSV data.

    Args:
        video_path: Path to input video
        analysis_csv: Path to SLAM analysis CSV
        trajectory_txt: Path to trajectory file
        output_video_path: Path for output video
        warning_cooldown: Frames to wait before repeating warnings
        frame_subsample: Process every nth frame
        critical_distance: Optional distance threshold for warnings
    """
    # Load and process CSV data
    df = pd.read_csv(analysis_csv)
    trajectory = load_trajectory(trajectory_txt)

    generator = ObjectWarningGenerator(warning_cooldown, critical_distance)
    audio_overlay = AudioOverlay()

    # Load video info
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Process each frame and generate warnings
    warnings = []
    for step in sorted(df.step_nr.unique()):
        # Adjust frame index to match original video timing
        frame_idx = step * frame_subsample
        
        # Get camera pose for current frame
        camera_pose = trajectory[step]
        
        # Get objects for current frame
        frame_objects = df[df.step_nr == step]
        
        # Generate warning message if any objects detected
        message = generator.process_objects(frame_objects, frame_idx, camera_pose)
        if message:
            warnings.append((frame_idx, message))

    # Generate and overlay audio
    audio_path = audio_overlay.warnings_to_audio(warnings, fps, duration)
    audio_overlay.overlay_audio_on_video(video_path, audio_path, output_video_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate video with audio warnings from SLAM analysis CSV.")
    parser.add_argument("video_path", help="Path to the input video.")
    parser.add_argument("analysis_csv", help="Path to the SLAM analysis CSV file.")
    parser.add_argument("trajectory_txt", help="Path to the camera trajectory file.")
    parser.add_argument("output_video_path", help="Path to save the modified video.")
    parser.add_argument(
        "--warning-cooldown",
        "-c",
        type=int,
        default=30,
        help="Number of frames to wait before repeating a warning about the same object (default: 30)"
    )
    parser.add_argument(
        "--frame-subsample",
        "-s",
        type=int,
        default=1,
        help="Only process every nth frame, e.g., 2 means every second frame (default: 1)"
    )
    parser.add_argument(
        "--critical-distance",
        "-d",
        type=float,
        default=float('inf'),
        help="Only warn about objects closer than this distance (default: no limit)"
    )
    args = parser.parse_args()
    main(
        args.video_path,
        args.analysis_csv,
        args.trajectory_txt,
        args.output_video_path,
        warning_cooldown=args.warning_cooldown,
        frame_subsample=args.frame_subsample,
        critical_distance=args.critical_distance
    )