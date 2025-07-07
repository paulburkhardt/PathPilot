#!/usr/bin/env python3
"""
Simple TTS test script to debug audio issues
"""

import sys

def test_tts():
    """Test text-to-speech functionality."""
    try:
        import pyttsx3
        print("✓ pyttsx3 imported successfully")
        
        # Initialize TTS engine
        engine = pyttsx3.init()
        print("✓ TTS engine initialized")
        
        # Test basic speech
        test_message = "Warning! Obstacle 0.3 meters to your right"
        print(f"Testing message: {test_message}")
        
        # Set properties
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        print("✓ TTS properties set")
        
        # Test speech output
        print("Testing audio output...")
        engine.say(test_message)
        engine.runAndWait()
        print("✓ Audio test completed")
        
        # Test file output
        print("Testing file output...")
        output_file = "test_audio.wav"
        engine.save_to_file(test_message, output_file)
        engine.runAndWait()
        
        from pathlib import Path
        if Path(output_file).exists():
            print(f"✓ Audio file created: {output_file}")
            # Clean up
            Path(output_file).unlink()
        else:
            print("✗ Audio file not created")
        
        return True
        
    except ImportError as e:
        print(f"✗ pyttsx3 not available: {e}")
        print("Install with: pip install pyttsx3")
        return False
        
    except Exception as e:
        print(f"✗ TTS error: {e}")
        return False

def test_moviepy():
    """Test moviepy functionality."""
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        print("✓ moviepy imported successfully")
        return True
    except ImportError as e:
        print(f"✗ moviepy not available: {e}")
        print("Install with: pip install moviepy")
        return False

def main():
    """Main test function."""
    print("=== Audio System Test ===")
    
    # Test TTS
    tts_ok = test_tts()
    
    # Test MoviePy
    moviepy_ok = test_moviepy()
    
    print("\n=== Test Results ===")
    print(f"TTS (pyttsx3): {'✓ OK' if tts_ok else '✗ FAILED'}")
    print(f"MoviePy: {'✓ OK' if moviepy_ok else '✗ FAILED'}")
    
    if tts_ok and moviepy_ok:
        print("\n✓ All audio systems working correctly!")
        return 0
    else:
        print("\n✗ Some audio systems failed. Check installation:")
        if not tts_ok:
            print("  - pip install pyttsx3")
            print("  - sudo apt-get install espeak espeak-data (Linux)")
        if not moviepy_ok:
            print("  - pip install moviepy")
            print("  - sudo apt-get install ffmpeg (Linux)")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 