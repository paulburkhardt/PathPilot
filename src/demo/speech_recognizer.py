from typing import Any
import speech_recognition as sr

class SpeechRecognizer:
    """
    Handles all user input via microphone
    """

    def __init__(self):
        
        self.speech_recognizer = sr.Recognizer()

        self.state = "Idle"


    def get_state(self):
        return self.state

    def record_and_transcribe(self):
        return self._transcribe_audio(self._record_audio())
        
    def _record_audio(self)->Any:
        """
        Uses the microphone to record audio
        """

        with sr.Microphone() as source:
            self.state = "Adjusting for noise..."
            self.speech_recognizer.adjust_for_ambient_noise(source)
            self.state = "Done. Speak now!"
            audio = self.speech_recognizer.listen(source)
            self.state = "Done. Recorded audio."
            return audio
    
    def _transcribe_audio(self,audio)->str:
        """transcribes the audio"""

        try:
            # Recognize speech using Google Web Speech API
            self.state = "Running speech recognition..."
            text = self.speech_recognizer.recognize_google(audio)
            self.state = "Idle"
            return text
        except sr.UnknownValueError:
            self.state = "Unable to understand audio. Please try Again!"
            raise RuntimeError("Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            self.state = "Unable to reach google service. Please try Again!"
            raise RuntimeError("Google Speech Recognition service could not be reached.")
