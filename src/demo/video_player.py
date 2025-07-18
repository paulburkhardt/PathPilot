import cv2
import threading
import time
import numpy as np

class VideoPlayer():

    def __init__(self,main_manager):
        self.main_manager = main_manager
        self.video_frames = None
        self.frame_idx = None
        self.video_fps = None

        self.is_playing = False

    def load_video(self,video_file:str):

        cap = cv2.VideoCapture(video_file)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        video_frames = []
        success, frame = cap.read()
        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame_rgb)
            success, frame = cap.read()
        cap.release()

        self.video_frames = video_frames
        self.frame_idx = 0
        self.video_fps = video_fps

    def start(self):

        if not self.is_playing:
            self.is_playing = True
            self.play_thread = threading.Thread(target = self.__play_loop)
            self.play_thread.daemon = True
            self.play_thread.start()

    def stop(self):
        self.is_playing = False
        if self.play_thread:
            self.play_thread.join()
        print("Joined Video thread")

    def get_current_frame(self):
        return self.current_frame

    @property
    def current_frame(self):
        if self.video_frames is not None:
            return self.video_frames[self.frame_idx]
        else:
            return np.zeros((256,256,3))

    def __play_loop(self):

        while self.is_playing and self.frame_idx < len(self.video_frames):
            if self.frame_idx == len(self.video_frames)-1:
                self.frame_idx = 0
            else:
                self.frame_idx +=1
            self.main_manager.process_frame()
            time.sleep(1/self.video_fps)
        