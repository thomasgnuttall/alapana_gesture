import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, savgol_filter, sosfilt

import random

Fs=44100
N=1024
H=512
gamma=200
M=10
norm=True

# Load
i = 735

track = all_groups[all_groups['index']==i].iloc[0]
t = track.track
start = track.start
end = track.end
audio_track = audio_tracks[t]
x = audio_track[round(start*Fs):round(end*Fs)]
pitch, time, timestep = pitch_tracks[t]
pitch = pitch[round(start/timestep):round(end/timestep)]
time = time[round(start/timestep):round(end/timestep)]
pitch = interpolate_below_length(pitch, 0, (350*0.001/timestep))

def trim_zeros(pitch, time):
    m = pitch!=0
    i1,i2 = m.argmax(), m.size - m[::-1].argmax()
    return pitch[i1:i2], time[i1:i2]

sf = get_loudness(x)
loudness_timestep = len(x)/(Fs*len(sf))
pitch, sf = trim_zeros(pitch,sf)



wms = 75
wl = round(wms*0.001/loudness_timestep)
wl = wl if not wl%2 == 0 else wl+1
interp75 = savgol_filter(sf, polyorder=2, window_length=wl, mode='interp')

wms = 100
wl = round(wms*0.001/loudness_timestep)
wl = wl if not wl%2 == 0 else wl+1
interp100 = savgol_filter(sf, polyorder=2, window_length=wl, mode='interp')

wms = 125
wl = round(wms*0.001/loudness_timestep)
wl = wl if not wl%2 == 0 else wl+1
interp125 = savgol_filter(sf, polyorder=2, window_length=wl, mode='interp')

# Plot
fig, axs = plt.subplots(4, figsize=(15, 30))
plt.subplots_adjust(hspace=0.3)

axs[0].plot(range(len(sf)), sf)
axs[1].plot(range(len(interp75)), interp75)
axs[2].plot(range(len(interp100)), interp100)
axs[3].plot(range(len(interp125)), interp125)

axs[0].set_title(f'Original (index={i})', fontsize=10)
axs[1].set_title(f'75ms', fontsize=10)
axs[2].set_title(f'100ms', fontsize=10)
axs[3].set_title(f'125ms', fontsize=10)

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()

plt.xlabel('Time (s)')
axs[0].set_ylabel(f'Loudness (dB)')
axs[1].set_ylabel(f'Loudness (dB)')
axs[2].set_ylabel(f'Loudness (dB)')
axs[3].set_ylabel(f'Loudness (dB)')

plt.savefig('smooth_test.png')
