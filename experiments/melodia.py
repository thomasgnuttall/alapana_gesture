import numpy as np
import essentia.standard as estd
from exploration.io import create_if_not_exists
def extract_pitch_melodia(audio_path, frame_size=2048, hop_size=128, sr=44100):
    # Running melody extraction with MELODIA
    pitch_extractor = estd.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size)
    audio = estd.EqloudLoader(filename=audio_path, sampleRate=sr)()
    est_freq, _ = pitch_extractor(audio)
    est_freq = np.append(est_freq, 0.0)
    est_time = np.linspace(0.0, len(audio) / sr, len(est_freq))
    
    return est_time, est_freq


def save_pitch_track_to_dataset(filepath, est_time, est_freq):
    """
    Function to write txt annotation to file
    """
    create_if_not_exists(filepath)
    with open(filepath, 'w') as f:
        for i, j in zip(est_time, est_freq):
            f.write("{}, {}\n".format(i, j))
    print(f'saved successfully to {filepath}')

audios = [
    'audio/lara_small/spleeter/2018_11_13_am_Sec_3_P1_Anandabhairavi.mp3',
    'audio/lara_small/spleeter/2018_11_18_am_Sec_6_P6_Bilahari_B.mp3'
]

filepaths = [
    'data/pitch_tracks/lara_small/2018_11_13_am_Sec_3_P1_Anandabhairavi.csv',
    'data/pitch_tracks/lara_small/2018_11_18_am_Sec_6_P6_Bilahari_B.csv'
]

for a,f in zip(audios, filepaths):
	try:
		est_t, est_f = extract_pitch_melodia(a)
		save_pitch_track_to_dataset(f, est_t, est_f)
		print(f'SUCCESS: {a}')
	except:
		print(f'FAIL: {a}')
        