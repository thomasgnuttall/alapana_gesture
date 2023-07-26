import skimage.io


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
i = 400

track = all_groups[all_groups['index']==i].iloc[0]
t = track.track
start = track.start
end = track.end
audio_track = audio_tracks[t]
x = audio_track[round(start*Fs):round(end*Fs)]

#sf.write('735_origial.wav', x, samplerate=Fs)

# Fourier transform
X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
Fs_feature = Fs / H
Y = np.log(1 + gamma * np.abs(X))
Y_diff = np.diff(Y)
Y_diff[Y_diff < 0] = 0


novelty_spectrum = np.sum(Y_diff, axis=0)
novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
if M > 0:
    local_average = compute_local_average(novelty_spectrum, M)
    novelty_spectrum = novelty_spectrum - local_average
    novelty_spectrum[novelty_spectrum < 0] = 0.0
if norm:
    max_value = max(novelty_spectrum)
    if max_value > 0:
        novelty_spectrum = novelty_spectrum / max_value


#skimage.io.imsave(f'plots/spectrogram.png', np.abs(X))

plt.figure(figsize=(15,5))
plt.imshow(Y, interpolation='nearest', aspect='auto')

plt.title(f'index={i}')
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.savefig(f'plots/spectrogram_{i}.png')
plt.close('all')





plt.figure(figsize=(15,5))
plt.plot(range(len(novelty_spectrum)), novelty_spectrum)

plt.title(f'index={i}')
plt.ylabel('Spectral Flux')
plt.xlabel('Time')
plt.savefig(f'plots/sf_{i}.png')
plt.close('all')
