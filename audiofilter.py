import noisereduce as nr
from pydub import AudioSegment
import numpy as np
import librosa

def load_audio(file_path):
    """Load an audio file into a numpy array and return its sample rate."""
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def save_audio(audio, sample_rate, output_path):
    """Save the numpy array as an audio file."""
    audio_segment = AudioSegment(audio.tobytes(), frame_rate=sample_rate, sample_width=audio.dtype.itemsize, channels=1)
    audio_segment.export(output_path, format="wav")

def reduce_noise(audio, sample_rate):
    """Apply noise reduction to isolate vocals."""
    # Assuming the first 0.5 seconds is noise
    noise_clip = audio[:int(0.5 * sample_rate)]
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=sample_rate, y_noise=noise_clip, prop_decrease=1.0)
    return reduced_noise_audio

def main(input_file_path, output_file_path):
    audio, sample_rate = load_audio(input_file_path)
    filtered_audio = reduce_noise(audio, sample_rate)
    save_audio(filtered_audio, sample_rate, output_file_path)

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]  # Path to the input audio file
    output_path = sys.argv[2]  # Path to save the filtered audio file
    main(input_path, output_path)