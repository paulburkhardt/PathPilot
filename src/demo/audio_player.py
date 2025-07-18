import pygame
import time
import threading
import tempfile
from gtts import gTTS
import os


# Constants for distance-based warning intervals (in meters : seconds)
DISTANCE_WARNING_MAP = {
    (0.0, 0.8): 0.0,    # Continuous beeping
    (0.8, 1.0): 0.1,    # 0.1s pause between beeps
    (1.0, 1.5): 0.4,    # 0.4s pause between beeps
    (1.5, 2.0): 0.8     # 0.8s pause between beeps
}


class AudioPlayer:
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
        self.audio_thread = None
        self.running = False
        
        self.left_beep = pygame.mixer.Sound(self.path_left_beep)
        self.right_beep = pygame.mixer.Sound(self.path_right_beep)

        # Current state
        self.beep = None
        self.interval = float('inf')
        self.last_beep_time = 0
        

    @property
    def path_left_beep(self):
        return "static/left_beep.wav"

    @property
    def path_right_beep(self):
        return "static/right_beep.wav"

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
        self.beep = self.left_beep if is_left else self.right_beep
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
        print("Joined audio thread.")
            
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



    