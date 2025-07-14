"""
Generate beep sound files for the interactive video player.
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

def create_envelope(duration: float, sample_rate: int, attack: float = 0.02, decay: float = 0.05, 
                   sustain: float = 0.5, release: float = 0.1) -> np.ndarray:
    """
    Create an ADSR (Attack, Decay, Sustain, Release) envelope for the tone.
    
    Args:
        duration: Total duration in seconds
        sample_rate: Audio sample rate
        attack: Attack time in seconds
        decay: Decay time in seconds
        sustain: Sustain level (0-1)
        release: Release time in seconds
    """
    total_samples = int(duration * sample_rate)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples
    
    envelope = np.zeros(total_samples)
    
    # Attack (linear ramp up)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay (exponential decay to sustain level)
    decay_curve = np.exp(np.linspace(0, -4, decay_samples))
    decay_curve = decay_curve * (1 - sustain) + sustain
    envelope[attack_samples:attack_samples + decay_samples] = decay_curve
    
    # Sustain (constant level)
    envelope[attack_samples + decay_samples:-release_samples] = sustain
    
    # Release (exponential decay to zero)
    release_curve = np.exp(np.linspace(0, -8, release_samples))
    envelope[-release_samples:] = release_curve * sustain
    
    return envelope

def apply_bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, 
                         sample_rate: int, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to make the sound more focused."""
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def generate_beep(frequency: float, duration: float = 0.2, amplitude: float = 0.5, 
                 sample_rate: int = 44100) -> np.ndarray:
    """
    Generate a pleasing beep sound with given frequency.
    
    Args:
        frequency: Base frequency of the beep in Hz
        duration: Duration of the beep in seconds
        amplitude: Volume of the beep (0-1)
        sample_rate: Audio sample rate
    Returns:
        Numpy array of audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate harmonics
    fundamental = np.sin(2 * np.pi * frequency * t)
    second_harmonic = 0.3 * np.sin(2 * np.pi * (2 * frequency) * t)  # Octave up
    third_harmonic = 0.15 * np.sin(2 * np.pi * (3 * frequency) * t)  # Perfect fifth above octave
    
    # Combine harmonics
    tone = fundamental + second_harmonic + third_harmonic
    
    # Apply envelope
    envelope = create_envelope(duration, sample_rate)
    tone = tone * envelope
    
    # Apply bandpass filter
    tone = apply_bandpass_filter(tone, frequency * 0.7, frequency * 4, sample_rate)
    
    # Normalize and convert to 16-bit integer
    tone = tone / np.max(np.abs(tone))  # Normalize to -1 to 1
    tone = (tone * amplitude * 32767).astype(np.int16)
    
    return tone

def main():
    # Generate distinct frequencies for left and right
    # Using musical perfect fifth interval (ratio 3:2) for pleasing combination
    left_freq = 440   # A4 (440 Hz) for left warning
    right_freq = 660  # E5 (660 Hz) for right warning
    
    # Generate both beeps
    left_beep = generate_beep(left_freq, duration=0.15)  # Slightly shorter duration
    right_beep = generate_beep(right_freq, duration=0.15)
    
    # Save as wav files
    wavfile.write('static/left_beep.wav', 44100, left_beep)
    wavfile.write('static/right_beep.wav', 44100, right_beep)

if __name__ == "__main__":
    main()
