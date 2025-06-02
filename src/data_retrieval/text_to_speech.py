from typing import Dict, Any
from gtts import gTTS
import os

class TextToSpeech:
    def __init__(self, config: Dict[str, Any]):
        pass

    def text_to_speech(self, text: str) -> Any:
        speech = gTTS(text=text, lang='en', slow=True)  # slow=True makes the speech slower and more natural
        output_dir = os.path.join('..', 'outputs', 'audio')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'speech.mp3')
        speech.save(output_path)
        #os.system(f"start {output_path}")
