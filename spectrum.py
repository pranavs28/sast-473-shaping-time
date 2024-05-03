import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


note_indices = {'C': 0, 'C#': 1, 
                'Db': 1, 'D': 2, 'D#': 3, 
                'Eb': 3, 'E': 4, 'E#': 5,
                'Fb': 4, 'F': 5, 'F#': 6, 
                'Gb': 6, 'G': 7, 'G#': 8, 
                'Ab': 8, 'A': 9, 'A#': 10, 
                'Bb': 10, 'B': 11, 'B#': 0,
                'Cb': 11}


def plot_spectrum(audio_file):
    try:
        # Load the audio file
        data, sample_rate = librosa.load(audio_file, sr=None)  # `sr=None` to preserve original sample rate
        print("Data length:", len(data), "Sample rate:", sample_rate)
        
        if len(data) == 0:
            raise ValueError("Audio data is empty. Check if the file is corrupted or the path is correct.")
        
        # Compute the Fourier transform and frequency spectrum
        spectrum = np.fft.fft(data)
        frequency = np.fft.fftfreq(len(spectrum), 1 / sample_rate)
        
        # Only take the positive frequencies
        pos_mask = np.where(frequency > 0)  # Changed from >= to > to exclude zero
        frequencies = frequency[pos_mask]
        magnitudes = np.abs(spectrum[pos_mask])
    
        # Plot the frequency spectrum
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies, magnitudes)
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 3000)  # Limit frequency range to 3000 Hz for better visibility
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("An error occurred:", str(e))

def frequency_to_note(frequency):
    # Define the notes of the scale
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Reference frequency of A4
    A4 = 440.0
    # Guard against zero frequency which causes a log2 calculation issue
    if frequency <= 0:
        return None
    # Number of semitones between the given frequency and A4
    n = int(round(12 * np.log2(frequency / A4)))
    # Calculate the index of the note
    note_index = n % 12
    return NOTES[note_index]

def get_top_notes(audio_file, n):
    # Load the audio file
    data, sample_rate = librosa.load(audio_file, sr=None)
    
    # Compute the Fourier transform and frequency spectrum
    spectrum = np.fft.fft(data)
    frequency = np.fft.fftfreq(len(spectrum), 1 / sample_rate)
    
    # Only take the positive frequencies
    pos_mask = np.where(frequency > 0)  # Exclude zero frequencies
    frequencies = frequency[pos_mask]
    magnitudes = np.abs(spectrum[pos_mask])
    
    # Map frequencies to notes and sum magnitudes
    note_sums = {}
    for freq, mag in zip(frequencies, magnitudes):
        note = frequency_to_note(freq)
        if note:
            if note in note_sums:
                note_sums[note] += mag
            else:
                note_sums[note] = mag

    # Sort notes by their total magnitudes and get the top n
    sorted_notes = sorted(note_sums.items(), key=lambda item: item[1], reverse=True)
    top_notes = sorted_notes[:n]

    return top_notes

def print_note_variants(top_notes):
    note_bases = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    note_variants = {base: {} for base in note_bases}

    # Organize notes into variants by their base note and accumulate magnitudes
    for note, magnitude in top_notes:
        base_note = note[0]  # The base note is always the first character
        # Check if the note has a sharp
        if len(note) > 1 and note[1] == '#':
            base_note += '#'  # Include the sharp in the base note

        # Accumulate magnitudes for each variant
        if base_note in note_variants[base_note[0]]:
            note_variants[base_note[0]][base_note] += magnitude
        else:
            note_variants[base_note[0]][base_note] = magnitude
    scale = []
    # Print the most prominent variant for each base note
    for base in note_bases:
        print(f"{base} Variants:")
        if note_variants[base]:
            # Find the variant with the maximum magnitude
            most_prominent = max(note_variants[base], key=note_variants[base].get)
            print(f"  Most Prominent: {most_prominent} ({note_variants[base][most_prominent]})")
            scale.append(most_prominent)
        else:
            print("  None")
    return scale


# Example usage


def standardize_scale(notes, start_note):
    # Mapping of notes to their positions in a standard C major scale
    note_indices = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    
    # Calculate the offset from C
    start_index = note_indices[start_note]
    offset = -start_index
    
    ['c', 'd#', 'e', 'f#', 'a', 'b']
    # Standardize the scale to start from C
    standardized_notes = {chr((note_indices[note] + offset) % 12 + ord('C')) for note in notes}
    
    # Map back to correct notes considering sharps
    index_to_note = {v: k for k, v in note_indices.items()}
    standardized_notes = {index_to_note[(note_indices[note] + offset) % 12] for note in notes}
    
    return standardized_notes

def identify_raga(notes, start_note):
    # Define some Carnatic ragas with scales starting from 'C' for standardization
    ragas = {
        'Mayamalavagowla': {'C', 'Db', 'E', 'F', 'G', 'Ab', 'B'},
        'Hamsadhwani': {'C', 'D', 'E', 'G', 'B'},
        'Kalyani': {'C', 'D', 'E', 'F#', 'G', 'A', 'B'},
        'Shankarabharanam': {'C', 'D', 'E', 'F', 'G', 'A', 'B'},
        'Kharaharapriya': {'C', 'D', 'Eb', 'F', 'G', 'Ab', 'B'},
        'Harikambhoji': {'C', 'D', 'E', 'F', 'G', 'A', 'A', 'Bb'},
        'Kambhoji': {'C', 'D', 'E', 'F', 'G', 'A'}
    }

    # Convert the notes based on the start note to a standard scale starting from 'C'
    standardized_notes = standardize_scale(notes, start_note)
    print(standardized_notes)

    # Identify the raga by matching the standardized scale with raga scales
    for raga, scale in ragas.items():
        if standardized_notes == scale:
            return raga

    return "Unknown raga"

def identify_tonic_and_dominant(audio_file):
    # Load the audio file with librosa
    y, sr = librosa.load(audio_file, sr=None)

    # Use a harmonic percussive source separation to isolate harmonic elements
    y_harmonic, _ = librosa.effects.hpss(y)

    # Calculate the constant-Q transform of the harmonic component
    cqt = librosa.cqt(y_harmonic, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=60, bins_per_octave=12)

    # Calculate the mean magnitude of each frequency bin across the time axis
    cqt_mag = np.mean(np.abs(cqt), axis=1)

    # Find the index of the maximum magnitude, which likely corresponds to Sa
    sa_index = np.argmax(cqt_mag)
    sa_freq = librosa.cqt_frequencies(n_bins=60, fmin=librosa.note_to_hz('C1'), bins_per_octave=12)[sa_index]

    # Calculate the frequency of Pa, which is a perfect fifth above Sa (approximately 1.5 times the frequency of Sa)
    pa_freq = sa_freq * 1.5

    # Convert frequencies to notes
    sa_note = librosa.hz_to_note(sa_freq)
    pa_note = librosa.hz_to_note(pa_freq)

    return sa_note, pa_note


def run(filename):
    plot_spectrum(filename)
    top_notes = get_top_notes(filename, 12)
    print("Top Notes:")
    for note, magnitude in top_notes:
        print(f"{note}: {magnitude}")
    scale = print_note_variants(top_notes)
    print("OG Scale: ", scale)
    # Example usage

    sa, pa = identify_tonic_and_dominant(filename)
    print("Sa (Tonic):", sa[:-1])
    numeric_scale = []
    for piece in scale: 
        numeric_scale.append((note_indices[piece] - note_indices[sa[:-1]]) % 12)
    # raga = identify_raga(scale, sa[:-1])
    print(numeric_scale)



run('mayamalavagolaivarnam.mp3')
