#!/usr/bin/env python3
"""
Enhanced Warning Video Generator

Creates an enhanced version of the original video with:
- Visual warning overlays (text and color bars)
- Audio warnings for navigation assistance
- Accessibility features for visually impaired users

Usage:
    python create_warning_video.py /path/to/slam_analysis_output_dir [options]
"""

import argparse
import json
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import subprocess
import tempfile
import os
from dataclasses import dataclass
import math

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Audio warnings will be disabled.")

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Some video processing features will be limited.")


@dataclass
class WarningFrame:
    """Data structure for a single frame's warning information."""
    timestamp: float
    distance: float
    direction: str
    warning_level: str
    position: np.ndarray
    quaternion: np.ndarray
    closest_point: np.ndarray


class WarningVideoGenerator:
    """
    Generator for enhanced warning videos with visual and audio alerts.
    """
    
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """
        Initialize the warning video generator.
        
        Args:
            output_dir: Path to SLAM analysis output directory
            config: Configuration dictionary with options
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.data = {}
        self.warning_frames: List[WarningFrame] = []
        
        # Warning system configuration
        self.warning_thresholds = {
            'critical': config.get('warning_distance_critical', 0.2),
            'strong': config.get('warning_distance_strong', 0.4),
            'caution': config.get('warning_distance_caution', 0.6)
        }
        
        # Visual styling
        self.colors = {
            'normal': (0, 255, 0),      # Green (BGR for OpenCV)
            'caution': (0, 255, 255),   # Yellow
            'strong': (0, 165, 255),    # Orange
            'critical': (0, 0, 255)     # Red
        }
        
        # Audio setup
        self.tts_engine = None
        if TTS_AVAILABLE and config.get('enable_audio', True):
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', config.get('speech_rate', 150))
                self.tts_engine.setProperty('volume', config.get('speech_volume', 0.8))
            except Exception as e:
                print(f"Warning: Could not initialize TTS engine: {e}")
                self.tts_engine = None
        
        # Validate output directory
        if not self.output_dir.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
    
    def load_slam_data(self) -> None:
        """Load SLAM analysis data and original video information."""
        print("Loading SLAM analysis data...")
        
        # Load metadata
        self._load_metadata()
        
        # Load trajectory data
        self._load_trajectory()
        
        # Load closest points data
        self._load_closest_points()
        
        # Get video path from metadata
        self._get_video_path()
        
        # Process data into warning frames
        self._process_warning_frames()
    
    def _load_metadata(self) -> None:
        """Load metadata.json if available."""
        metadata_path = self.output_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.data['metadata'] = json.load(f)
            print(f"Loaded metadata from {metadata_path}")
        else:
            print("No metadata.json found")
            self.data['metadata'] = {}
    
    def _load_trajectory(self) -> None:
        """Load camera trajectory data."""
        # Try multiple possible filenames
        possible_names = [
            "slam_analysis_trajectory.txt",
            "incremental_analysis_detailed_trajectory.txt"
        ]
        
        traj_file = None
        for name in possible_names:
            candidate = self.output_dir / name
            if candidate.exists():
                traj_file = candidate
                break
        
        if traj_file and traj_file.exists():
            try:
                trajectory_data = np.loadtxt(traj_file)
                self.data['trajectory'] = {
                    'timestamps': trajectory_data[:, 0],
                    'positions': trajectory_data[:, 1:4],
                    'quaternions': trajectory_data[:, 4:8]  # [qx, qy, qz, qw]
                }
                print(f"Loaded trajectory with {len(self.data['trajectory']['positions'])} poses")
            except Exception as e:
                print(f"Failed to load trajectory from {traj_file}: {e}")
                raise
        else:
            raise FileNotFoundError("No trajectory file found")
    
    def _load_closest_points(self) -> None:
        """Load closest points analysis data."""
        # Try JSON first, then CSV
        closest_files = [
            self.output_dir / "slam_analysis_closest_points.json",
            self.output_dir / "slam_analysis_closest_points.csv",
            self.output_dir / "incremental_analysis_detailed_closest_points.json",
            self.output_dir / "incremental_analysis_detailed_closest_points.csv"
        ]
        
        for closest_file in closest_files:
            if closest_file.exists():
                try:
                    if closest_file.suffix == '.json':
                        with open(closest_file, 'r') as f:
                            closest_data = json.load(f)
                        self.data['closest_points'] = {
                            'points_3d': np.array(closest_data['n_closest_points_3d']),
                            'distances': np.array(closest_data['n_closest_points_distances'])
                        }
                    else:  # CSV
                        with open(closest_file, 'r') as f:
                            lines = f.readlines()
                        
                        if len(lines) > 1:
                            step_data = {}
                            for line in lines[1:]:
                                parts = line.strip().split(',')
                                if len(parts) >= 7:
                                    step = int(parts[0])
                                    point_idx = int(parts[1])
                                    
                                    if point_idx == 0:  # First closest point
                                        step_data[step] = {
                                            'point': [float(parts[2]), float(parts[3]), float(parts[4])],
                                            'distance': float(parts[6])
                                        }
                            
                            points_3d = []
                            distances = []
                            steps = []
                            for step in sorted(step_data.keys()):
                                points_3d.append([step_data[step]['point']])
                                distances.append([step_data[step]['distance']])
                                steps.append(step)
                            
                            self.data['closest_points'] = {
                                'points_3d': np.array(points_3d, dtype=object),
                                'distances': np.array(distances, dtype=object),
                                'steps': np.array(steps)  # Include step information
                            }
                    
                    print(f"Loaded closest points data from {closest_file}")
                    return
                except Exception as e:
                    print(f"Failed to load closest points data from {closest_file}: {e}")
        
        raise FileNotFoundError("No closest points data found")
    
    def _get_video_path(self) -> None:
        """Extract video path from pipeline configuration."""
        video_path = None
        
        # Try to get from metadata
        if 'pipeline_configuration' in self.data['metadata']:
            pipeline_config = self.data['metadata']['pipeline_configuration']
            if 'pipeline' in pipeline_config and 'components' in pipeline_config['pipeline']:
                for component in pipeline_config['pipeline']['components']:
                    if component.get('type') == 'MAST3RSLAMVideoDataset':
                        video_path = component.get('config', {}).get('video_path')
                        break
        
        if not video_path:
            raise ValueError("Could not find video path in metadata")
        
        # Resolve path
        if not Path(video_path).is_absolute():
            # Try relative to current directory, then relative to output directory
            video_file = Path(video_path)
            if not video_file.exists():
                video_file = self.output_dir.parent / video_path
                if not video_file.exists():
                    script_dir = Path(__file__).parent
                    video_file = script_dir / video_path
        else:
            video_file = Path(video_path)
        
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        self.data['video_path'] = str(video_file)
        print(f"Found video: {video_file}")
    
    def _process_warning_frames(self) -> None:
        """Process SLAM data into warning frames for video generation."""
        print("Processing warning frames...")
        
        trajectory = self.data['trajectory']
        closest_points = self.data['closest_points']
        
        timestamps = trajectory['timestamps']
        positions = trajectory['positions']
        quaternions = trajectory['quaternions']
        
        # Prepare closest points data (same logic as visualize_slam_output.py)
        closest_points_data = self._prepare_closest_points_data(len(positions))
        
        for i in range(len(positions)):
            # Get closest point data
            if closest_points_data and i < len(closest_points_data['points']):
                closest_point = closest_points_data['points'][i]
                distance = closest_points_data['distances'][i]
                
                # Check if this pose has valid closest point data
                has_valid_data = True
                if 'valid_mask' in closest_points_data:
                    has_valid_data = closest_points_data['valid_mask'][i]
                
                if not (has_valid_data and distance > 0):
                    closest_point = positions[i]  # Fallback
                    distance = 0.0
            else:
                closest_point = positions[i]  # Fallback
                distance = 0.0
            
            # Calculate warning level
            warning_level = self._calculate_warning_level(distance)
            
            # Calculate direction
            direction = self._calculate_direction_to_point(
                positions[i], quaternions[i], closest_point
            )
            
            # Create warning frame with step index instead of timestamp
            warning_frame = WarningFrame(
                timestamp=i,  # Use step index for synchronization
                distance=distance,
                direction=direction,
                warning_level=warning_level,
                position=positions[i],
                quaternion=quaternions[i],
                closest_point=closest_point
            )
            
            self.warning_frames.append(warning_frame)
        
        print(f"Processed {len(self.warning_frames)} warning frames")
    
    def _prepare_closest_points_data(self, num_poses: int) -> Optional[Dict[str, np.ndarray]]:
        """Prepare closest points data for visualization (same logic as visualize_slam_output.py)."""
        if 'closest_points' not in self.data:
            return None
            
        closest_data = self.data['closest_points']
        
        # Handle different data formats
        if 'points_3d' in closest_data:
            points_3d = closest_data['points_3d']
            distances = closest_data['distances']
            steps = closest_data.get('steps', None)
            
            # If we have step information, map data to trajectory poses
            if steps is not None:
                print(f"Mapping {len(steps)} closest point entries to {num_poses} trajectory poses using step indices")
                
                # Create arrays for all poses, filled with None initially
                mapped_points = [None] * num_poses
                mapped_distances = [None] * num_poses
                
                # Map data based on step indices
                for i, step in enumerate(steps):
                    if step < num_poses and len(points_3d[i]) > 0:
                        mapped_points[step] = points_3d[i][0]  # First closest point
                        mapped_distances[step] = distances[i][0]  # First distance
                
                # Fill missing data with fallback values
                final_points = []
                final_distances = []
                
                for i in range(num_poses):
                    if mapped_points[i] is not None:
                        final_points.append(mapped_points[i])
                        final_distances.append(mapped_distances[i])
                    else:
                        # Use trajectory position as fallback for missing data
                        final_points.append(self.data['trajectory']['positions'][i])
                        final_distances.append(0.0)
                
                return {
                    'points': np.array(final_points),
                    'distances': np.array(final_distances),
                    'valid_mask': np.array([mapped_points[i] is not None for i in range(num_poses)])
                }
            
            else:
                # Fallback to old behavior if no step information
                if len(points_3d) != num_poses:
                    print(f"Warning: Closest points data length ({len(points_3d)}) doesn't match trajectory length ({num_poses})")
                    # Take the minimum length
                    min_len = min(len(points_3d), num_poses)
                    points_3d = points_3d[:min_len]
                    distances = distances[:min_len]
                
                # Extract first closest point and distance for each pose
                first_closest_points = []
                first_distances = []
                
                for i in range(len(points_3d)):
                    if len(points_3d[i]) > 0:
                        first_closest_points.append(points_3d[i][0])
                        first_distances.append(distances[i][0])
                    else:
                        # Use trajectory position as fallback
                        first_closest_points.append(self.data['trajectory']['positions'][i])
                        first_distances.append(0.0)
                
                return {
                    'points': np.array(first_closest_points),
                    'distances': np.array(first_distances)
                }
        
        return None
    
    def _calculate_warning_level(self, distance: float) -> str:
        """Calculate warning level based on distance."""
        if distance <= self.warning_thresholds['critical']:
            return 'critical'
        elif distance <= self.warning_thresholds['strong']:
            return 'strong'
        elif distance <= self.warning_thresholds['caution']:
            return 'caution'
        else:
            return 'normal'
    
    def _calculate_direction_to_point(self, camera_position: np.ndarray, 
                                    camera_quaternion: np.ndarray, 
                                    closest_point: np.ndarray) -> str:
        """Calculate direction from camera to closest point."""
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = camera_quaternion
        
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm > 0:
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        # Calculate vector from camera to closest point
        to_point = closest_point - camera_position
        
        # Transform to camera coordinate system
        camera_coords = R.T @ to_point
        
        # Determine direction
        x, y, z = camera_coords
        direction_parts = []
        
        # Front/Back
        if abs(z) > 0.1:
            if z > 0:
                direction_parts.append("ahead")
            else:
                direction_parts.append("behind")
        
        # Left/Right
        if abs(x) > 0.1:
            if x > 0:
                direction_parts.append("right")
            else:
                direction_parts.append("left")
        
        # Up/Down
        if abs(y) > 0.1:
            if y > 0:
                direction_parts.append("below")
            else:
                direction_parts.append("above")
        
        # Combine directions
        if len(direction_parts) == 0:
            return "directly ahead"
        elif len(direction_parts) == 1:
            return f"to your {direction_parts[0]}"
        else:
            return "-".join(direction_parts)
    
    def _map_trajectory_to_video_frames(self, num_poses: int, total_video_frames: int) -> Dict[int, int]:
        """Map trajectory steps to video frames using step-based division (same as visualize_slam_output.py)."""
        print(f"Step-based mapping: {num_poses} trajectory poses to {total_video_frames} video frames")
        
        # Map trajectory steps evenly across video frames
        frame_mapping = {}
        
        if num_poses <= 1:
            # Special case: only one pose, use first frame
            frame_mapping[0] = 0
            return frame_mapping
        
        for i in range(num_poses):
            # Calculate progress through trajectory (0.0 to 1.0)
            progress = i / (num_poses - 1)
            
            # Map to video frame index
            video_frame_idx = int(progress * (total_video_frames - 1))
            video_frame_idx = min(video_frame_idx, total_video_frames - 1)  # Clamp to valid range
            
            # Store mapping from trajectory step to video frame
            frame_mapping[i] = video_frame_idx
        
        # Log mapping statistics
        unique_frames = len(set(frame_mapping.values()))
        print(f"Mapped trajectory steps to {unique_frames} unique video frames")
        print(f"Video frame step size: {total_video_frames / num_poses:.2f} frames per trajectory step")
        
        return frame_mapping
    
    def _get_warning_frame_for_video_frame(self, video_frame_idx: int, step_to_frame_mapping: Dict[int, int]) -> Optional[WarningFrame]:
        """Get the appropriate warning frame for a given video frame index."""
        # Find the trajectory step that maps to this video frame (or the closest one)
        best_step = 0
        best_distance = float('inf')
        
        for step_idx, mapped_frame_idx in step_to_frame_mapping.items():
            distance = abs(mapped_frame_idx - video_frame_idx)
            if distance < best_distance:
                best_distance = distance
                best_step = step_idx
        
        # Return the warning frame for the best matching step
        if best_step < len(self.warning_frames):
            return self.warning_frames[best_step]
        
        return None
    
    def generate_enhanced_video(self) -> str:
        """Generate enhanced video with warning overlays and audio."""
        print("Generating enhanced video...")
        
        input_video_path = self.data['video_path']
        output_video_path = self.output_dir / f"enhanced_warning_video.mp4"
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input video: {width}x{height} at {fps}fps, {total_frames} frames")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Generate audio warnings
        audio_files = []
        if self.config.get('enable_audio', True):
            audio_files = self._generate_audio_warnings()
        
        # Map trajectory steps to video frames (same approach as visualize_slam_output.py)
        num_trajectory_steps = len(self.warning_frames)
        step_to_frame_mapping = self._map_trajectory_to_video_frames(num_trajectory_steps, total_frames)
        
        frame_idx = 0
        
        print("Processing video frames with step-based synchronization...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find corresponding warning frame using step-based mapping
            warning_frame = self._get_warning_frame_for_video_frame(frame_idx, step_to_frame_mapping)
            
            if warning_frame:
                # Add visual overlays
                frame = self._add_visual_overlays(frame, warning_frame)
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing complete: {output_video_path}")
        
        # Add audio if available
        if audio_files and MOVIEPY_AVAILABLE:
            final_output = self._combine_video_with_audio(str(output_video_path), audio_files)
            return final_output
        
        return str(output_video_path)
    
    def _add_visual_overlays(self, frame: np.ndarray, warning_frame: WarningFrame) -> np.ndarray:
        """Add visual warning overlays to a frame."""
        height, width = frame.shape[:2]
        
        # Get colors for current warning level
        color = self.colors[warning_frame.warning_level]
        
        # Draw warning color bar (left side)
        bar_width = 20
        bar_height = height // 4
        bar_x = 10
        bar_y = (height - bar_height) // 2
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), color, -1)
        
        # Add color bar label
        cv2.putText(frame, warning_frame.warning_level.upper(), 
                   (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Generate warning message
        message = self._generate_warning_message(warning_frame)
        
        # Draw warning text (top of screen)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)
        
        # Draw background rectangle
        bg_height = text_height + baseline + 20
        cv2.rectangle(frame, (0, 0), (width, bg_height), (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (0, 0), (width, bg_height), color, 3)  # Colored border
        
        # Draw text
        text_x = (width - text_width) // 2
        text_y = text_height + 10
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Draw distance meter (bottom right)
        self._draw_distance_meter(frame, warning_frame)
        
        return frame
    
    def _draw_distance_meter(self, frame: np.ndarray, warning_frame: WarningFrame) -> None:
        """Draw a distance meter visualization."""
        height, width = frame.shape[:2]
        
        # Meter parameters
        meter_width = 200
        meter_height = 30
        meter_x = width - meter_width - 20
        meter_y = height - meter_height - 20
        
        # Background
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), 
                     (255, 255, 255), 2)
        
        # Distance scale (0-2 meters)
        max_distance = 2.0
        distance = min(warning_frame.distance, max_distance)
        fill_width = int((distance / max_distance) * meter_width)
        
        # Fill meter with appropriate color
        color = self.colors[warning_frame.warning_level]
        if fill_width > 0:
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_width, meter_y + meter_height), 
                         color, -1)
        
        # Add distance text
        distance_text = f"{warning_frame.distance:.2f}m"
        cv2.putText(frame, distance_text, (meter_x, meter_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add threshold markers
        for threshold_name, threshold_value in self.warning_thresholds.items():
            if threshold_value <= max_distance:
                marker_x = meter_x + int((threshold_value / max_distance) * meter_width)
                cv2.line(frame, (marker_x, meter_y), (marker_x, meter_y + meter_height), 
                        (255, 255, 255), 2)
    
    def _generate_warning_message(self, warning_frame: WarningFrame) -> str:
        """Generate warning message for display."""
        distance_str = f"{warning_frame.distance:.2f}m"
        
        if warning_frame.warning_level == 'normal':
            return f"Distance: {distance_str} ({warning_frame.direction})"
        
        warning_prefixes = {
            'caution': "CAUTION: ",
            'strong': "WARNING: ",
            'critical': "CRITICAL: "
        }
        
        prefix = warning_prefixes[warning_frame.warning_level]
        
        if warning_frame.warning_level == 'critical':
            return f"{prefix}Object {distance_str} {warning_frame.direction}!"
        else:
            return f"{prefix}Object {distance_str} {warning_frame.direction}"
    
    def _generate_audio_warnings(self) -> List[str]:
        """Generate audio warning files."""
        if not self.tts_engine:
            return []
        
        print("Generating audio warnings...")
        
        audio_files = []
        temp_dir = Path(tempfile.mkdtemp())
        
        # Group warnings by step intervals to avoid too frequent audio
        audio_step_interval = max(1, int(self.config.get('audio_interval', 2.0) * 30 / len(self.warning_frames)))  # Rough conversion
        last_audio_step = -audio_step_interval
        
        for i, warning_frame in enumerate(self.warning_frames):
            # Only generate audio for warnings and at intervals
            if (warning_frame.warning_level != 'normal' and 
                i >= last_audio_step + audio_step_interval):
                
                # Generate audio message
                audio_message = self._generate_audio_message(warning_frame)
                
                # Save audio file
                audio_file = temp_dir / f"warning_{i:06d}.wav"
                try:
                    self.tts_engine.save_to_file(audio_message, str(audio_file))
                    self.tts_engine.runAndWait()
                    
                    if audio_file.exists():
                        audio_files.append({
                            'file': str(audio_file),
                            'step_index': i,  # Use step index instead of timestamp
                            'duration': len(audio_message) * 0.1  # Rough estimate
                        })
                        last_audio_step = i
                        print(f"Generated audio for step {i}: {audio_message[:50]}...")
                except Exception as e:
                    print(f"Failed to generate audio for step {i}: {e}")
        
        print(f"Generated {len(audio_files)} audio warnings")
        return audio_files
    
    def _generate_audio_message(self, warning_frame: WarningFrame) -> str:
        """Generate audio message for TTS."""
        distance_str = f"{warning_frame.distance:.1f} meters"
        
        if warning_frame.warning_level == 'critical':
            return f"Critical warning! Obstacle {distance_str} {warning_frame.direction}!"
        elif warning_frame.warning_level == 'strong':
            return f"Warning! Obstacle {distance_str} {warning_frame.direction}"
        elif warning_frame.warning_level == 'caution':
            return f"Caution. Obstacle {distance_str} {warning_frame.direction}"
        else:
            return f"Clear. Distance {distance_str}"
    
    def _combine_video_with_audio(self, video_path: str, audio_files: List[Dict]) -> str:
        """Combine enhanced video with audio warnings."""
        if not MOVIEPY_AVAILABLE:
            print("MoviePy not available, skipping audio combination")
            return video_path
        
        print("Combining video with audio...")
        
        try:
            # Load video
            video = VideoFileClip(video_path)
            video_duration = video.duration
            
            # Create audio clips with step-based timing
            audio_clips = []
            for audio_info in audio_files:
                try:
                    audio_clip = AudioFileClip(audio_info['file'])
                    
                    # Convert step index to video timestamp
                    step_index = audio_info['step_index']
                    num_steps = len(self.warning_frames)
                    
                    # Calculate timestamp based on step position in video
                    if num_steps > 1:
                        step_timestamp = (step_index / (num_steps - 1)) * video_duration
                    else:
                        step_timestamp = 0.0
                    
                    # Ensure timestamp is within video duration
                    step_timestamp = min(step_timestamp, video_duration - audio_clip.duration)
                    step_timestamp = max(step_timestamp, 0.0)
                    
                    audio_clip = audio_clip.set_start(step_timestamp)
                    audio_clips.append(audio_clip)
                    
                    print(f"Audio for step {step_index} scheduled at {step_timestamp:.2f}s")
                    
                except Exception as e:
                    print(f"Failed to process audio file {audio_info['file']}: {e}")
            
            if audio_clips:
                # Combine all audio clips
                final_audio = CompositeAudioClip(audio_clips)
                
                # Set audio to video
                final_video = video.set_audio(final_audio)
                
                # Output path
                output_path = str(Path(video_path).with_suffix('')) + "_with_audio.mp4"
                
                # Write final video
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
                
                # Clean up
                video.close()
                final_video.close()
                for clip in audio_clips:
                    clip.close()
                final_audio.close()
                
                print(f"Final video with audio: {output_path}")
                return output_path
            else:
                video.close()
                return video_path
                
        except Exception as e:
            print(f"Error combining video with audio: {e}")
            return video_path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced warning video with visual and audio alerts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python create_warning_video.py ./enhanced_slam_outputs/slam_analysis_20240115_143045
    python create_warning_video.py ./outputs --no-audio --warning-distance-critical 0.15
    python create_warning_video.py ./outputs --speech-rate 120 --audio-interval 3.0
    
Features:
- Visual warning overlays with color-coded alerts
- Distance meter visualization
- Audio warnings for navigation assistance
- Accessibility features for visually impaired users
        """
    )
    
    # Required argument
    parser.add_argument(
        'output_dir',
        help='Path to SLAM analysis output directory'
    )
    
    # Warning system parameters
    parser.add_argument('--warning-distance-critical', type=float, default=0.2,
                       help='Distance threshold for critical warnings (meters)')
    parser.add_argument('--warning-distance-strong', type=float, default=0.4,
                       help='Distance threshold for strong warnings (meters)')
    parser.add_argument('--warning-distance-caution', type=float, default=0.6,
                       help='Distance threshold for caution warnings (meters)')
    
    # Audio parameters
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio warnings')
    parser.add_argument('--speech-rate', type=int, default=150,
                       help='Speech rate for audio warnings (words per minute)')
    parser.add_argument('--speech-volume', type=float, default=0.8,
                       help='Speech volume (0.0-1.0)')
    parser.add_argument('--audio-interval', type=float, default=2.0,
                       help='Minimum interval between audio warnings (seconds)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Build configuration
    config = {
        'warning_distance_critical': args.warning_distance_critical,
        'warning_distance_strong': args.warning_distance_strong,
        'warning_distance_caution': args.warning_distance_caution,
        'enable_audio': not args.no_audio,
        'speech_rate': args.speech_rate,
        'speech_volume': args.speech_volume,
        'audio_interval': args.audio_interval
    }
    
    try:
        # Create generator
        generator = WarningVideoGenerator(args.output_dir, config)
        
        # Load SLAM data
        generator.load_slam_data()
        
        # Generate enhanced video
        output_video = generator.generate_enhanced_video()
        
        print(f"\nEnhanced warning video created: {output_video}")
        print("\nFeatures included:")
        print("- Visual warning overlays with color-coded alerts")
        print("- Distance meter visualization")
        print("- Warning level color bar")
        if config['enable_audio']:
            print("- Audio warnings for navigation assistance")
        print("\nVideo is ready for playback!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 