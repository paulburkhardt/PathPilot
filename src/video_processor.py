import cv2
import numpy as np
from pathlib import Path

class VideoProcessor:
    def __init__(self, config):
        self.fps = config['fps']
        self.resolution = tuple(config['resolution'])
        self.cap = None

    def process_video(self, video_path: Path):
        """Process video file and yield frames."""
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize frame if needed
            if frame.shape[:2] != self.resolution[::-1]:
                frame = cv2.resize(frame, self.resolution)

            yield frame

        self.cap.release()

    def __del__(self):
        if self.cap is not None:
            self.cap.release() 