import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, savgol_filter, sosfilt

import random

feature='Pitch (cents)'



def trim_zeros(pitch, time):
	m = pitch!=0
	i1,i2 = m.argmax(), m.size - m[::-1].argmax()
	return pitch[i1:i2], time[i1:i2]

i = 325

motif = all_groups[all_groups['index']==i].iloc[0]
t = motif.track
pitch, time, timestep,pitch_d, time_d = pitch_tracks[t]

timestep = 0.010000908308633712 # for pitch
i_vec = pitch[round(motif['start']/timestep):round(motif['end']/timestep)]
i_time = time[round(motif['start']/timestep):round(motif['end']/timestep)]
i_vec,i_time = trim_zeros(i_vec, i_time)

# Gaussiaan Smooth
sigma = 2.5
gauss = gaussian_filter1d(i_vec, sigma)

# Butterworth Filter
freq = 30
sos = butter(2, freq, output='sos', fs=1/timestep)
bw = sosfilt(sos, i_vec)

# Interp
po = 2
wl = 0.125 #in seconds
wl_ = int(wl/timestep)
wl_ = wl_ if wl%2 ==0 else wl_+1
interp = savgol_filter(i_vec, polyorder=po, window_length=wl_, mode='interp')

fig, axs = plt.subplots(4, figsize=(8, 14))
plt.subplots_adjust(hspace=0.3)

axs[0].plot(i_time, i_vec)
axs[1].plot(i_time, gauss)
axs[2].plot(i_time, bw)
axs[3].plot(i_time, interp)

axs[0].set_title(f'Original (index={i})', fontsize=10)
axs[1].set_title(f'Gaussian. sigma={sigma}', fontsize=10)
axs[2].set_title(f'Butterworth. Freq={freq}Hz', fontsize=10)
axs[3].set_title(f'Savitzky-Golay. Polyorder={po}. Window length={wl}s', fontsize=10)

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()

plt.xlabel('Time (s)')
axs[0].set_ylabel(f'{feature}')
axs[1].set_ylabel(f'{feature}')
axs[2].set_ylabel(f'{feature}')
axs[3].set_ylabel(f'{feature}')

plt.savefig('smooth_test.png')