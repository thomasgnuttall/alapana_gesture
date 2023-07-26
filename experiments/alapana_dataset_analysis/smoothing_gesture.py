import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, savgol_filter, sosfilt

import random

feature='1daccelerationDTWHand'

i = 736#random.choice(all_groups['index'].values)

i_vec = index_features[i][feature]
timestep = 0.016666666660457848 # for gesture
t = [i*timestep for i in range(len(i_vec))]

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
wl_ = wl_ if wl_%2 else wl_+1
interp = savgol_filter(i_vec, polyorder=po, window_length=wl_, mode='interp')

fig, axs = plt.subplots(2, figsize=(8, 14))
plt.subplots_adjust(hspace=0.3)

axs[0].plot(t, i_vec)
#axs[1].plot(t, gauss)
#axs[2].plot(t, bw)
axs[1].plot(t, interp)

axs[0].set_title(f'Original (index={i})', fontsize=10)
#axs[1].set_title(f'Gaussian. sigma={sigma}', fontsize=10)
#axs[2].set_title(f'Butterworth. Freq={freq}Hz', fontsize=10)
axs[1].set_title(f'Savitzky-Golay. Polyorder={po}. Window length={wl}s', fontsize=10)

axs[0].grid()
#axs[1].grid()
#axs[2].grid()
axs[1].grid()

plt.xlabel('Time (s)')
axs[0].set_ylabel(f'{feature}')
#axs[1].set_ylabel(f'{feature}')
#axs[2].set_ylabel(f'{feature}')
axs[1].set_ylabel(f'{feature}')

plt.savefig('smooth_test.png')