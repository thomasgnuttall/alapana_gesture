import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, savgol_filter, sosfilt

import random

feature='Pitch (cents)'

po = 2
wl = 0.125 #in seconds
wl_ = int(wl/timestep)
wl_ = wl_ if wl%2 ==0 else wl_+1

def trim_zeros(pitch, time):
	m = pitch!=0
	i1,i2 = m.argmax(), m.size - m[::-1].argmax()
	return pitch[i1:i2], time[i1:i2]

i = 736

motif = all_groups[all_groups['index']==i].iloc[0]
t = motif.track
pitch, time, timestep, pitch_d, time_d = pitch_tracks[t]

timestep = 0.010000908308633712 # for Pitch
i_vec = np.array(pitch[round(motif['start']/timestep):round(motif['end']/timestep)])
i_time = np.array(time[round(motif['start']/timestep):round(motif['end']/timestep)])

pat1, pat1_time = smooth(i_vec, i_time)

diff1, diff1_time = get_derivative(pat1, pat1_time)

p75, t75 = smooth(diff1, diff1_time, 75)
p100, t100 = smooth(diff1, diff1_time, 100)
p125, t125 = smooth(diff1, diff1_time, 125)

# Interp
interp = savgol_filter(i_vec, polyorder=po, window_length=wl_, mode='interp')




# Plot
fig, axs = plt.subplots(1, figsize=(14, 4))
plt.subplots_adjust(hspace=0.3)

axs.plot(pat1_time, pat1)
#axs[1].plot(t75, p75)
#axs[2].plot(t100, p100)
#axs[3].plot(t125, p125)

axs.set_title(f'Original (index={i})', fontsize=10)
#axs[1].set_title(f'75ms', fontsize=10)
#axs[2].set_title(f'100ms', fontsize=10)
#axs[3].set_title(f'125ms', fontsize=10)

axs.grid()
#axs[1].grid()
#axs[2].grid()
#axs[3].grid()

plt.xlabel('Time (s)')
axs.set_ylabel(f'{feature}')
#axs[1].set_ylabel(f'{feature}')
#axs[2].set_ylabel(f'{feature}')
#axs[3].set_ylabel(f'{feature}')

plt.savefig('smooth_test.png')
