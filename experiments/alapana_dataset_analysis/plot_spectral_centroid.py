import skimage.io
from librosa.feature import spectral_centroid
import matplotlib.pyplot as plt

"""
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 256)
        gamma (float): Parameter for logarithmic compression (Default value = 100.0)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)
"""

Fs=44100
N=2048
H=512
gamma=200
M=10
norm=True

# Load
i = 736

track = all_groups[all_groups['index']==i].iloc[0]
t = track.track
start = track.start
end = track.end
audio_track = audio_tracks[t]
x = audio_track[round(start*Fs):round(end*Fs)]

#sf.write('735_origial.wav', x, samplerate=Fs)

# Fourier transform
centroid = spectral_centroid(
    y=x, sr=Fs, n_fft=N, hop_length=H, window='hanning', center=True, pad_mode='constant')[0]

sc_timestep = (len(x)/Fs)/len(centroid)

wms = 125
wl = round(wms*0.001/sc_timestep)
wl = wl if not wl%2 == 0 else wl+1
centroid125 = savgol_filter(centroid, polyorder=2, window_length=wl, mode='interp')

X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
Fs_feature = Fs / H
Y = np.log(1 + gamma * np.abs(X))

plt.figure(figsize=(15,5))
plt.imshow(Y, interpolation='nearest', aspect='auto')

plt.title(f'index={i}')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.savefig(f'plots/spectrogram_{i}.png')
plt.close('all')


plt.figure(figsize=(15,5))
plt.plot(range(len(centroid125)), centroid125)

plt.title(f'index={i}')
plt.ylabel('Spectral Centroid')
plt.xlabel('Time')
plt.savefig(f'plots/sc_no_smooth_{i}.png')
plt.close('all')
