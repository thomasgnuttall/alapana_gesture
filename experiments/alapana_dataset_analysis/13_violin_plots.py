run_name = 'result_0.1'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np


from scipy.stats import spearmanr
from exploration.io import create_if_not_exists, load_pkl

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'
results_dir = os.path.join(out_dir, 'analysis', '')
sig_results_df_path = os.path.join(results_dir, 'results_with_random.csv')

results = pd.read_csv(sig_results_df_path)

plot_path = os.path.join('plots', 'violin', 'test.png')
create_if_not_exists(plot_path)

## Plots

sonic_feature =  'diff_pitch_dtw'
gests = [ 
'3dpositionDTWHand', 
'3dvelocityDTWHand', 
'3daccelerationDTWHand',
'3dpositionDTWHead', 
'3dvelocityDTWHead', 
'3daccelerationDTWHead']

df = results[results['level']=='performance']
df = df[df['x']==sonic_feature]
N = df['level_value'].nunique()

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Gesture Feature')


# create test data
np.random.seed(19680801)

data = [(df[df['y']==g]['corr_real'].values, df[df['y']==g]['corr_random'].values) for g in gests]
data = [x for y in data for x in y]

#fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)
fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(18, 8), sharey=True)

sf_dict = {
    'diff_pitch_dtw': 'Δf0', 
    'pitch_dtw': 'f0',
    'loudness_dtw': 'Loudness', 
    'spectral_centroid': 'Spectral Centroid'
}

sf2 = sf_dict[sonic_feature]
ax2.set_title(f'Correlations Across {N} performances')
ax2.set_ylabel(f'Correlation w/ {sf2}\n(p<=0.01)')
ax2.violinplot(data)

parts = ax2.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

for i,pc in enumerate(parts['bodies']):
    if not i%2:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    else:
        pc.set_facecolor('#00ddff')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

percs = np.array([np.percentile(d, [25, 50, 75]) for d in data])
quartile1 = percs[:,0]
medians = percs[:,1]
quartile3 = percs[:,2]

whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# set style for the axes   
gests_alt = [x.replace("DTW", '').replace('acceleration','Acc ').replace('velocity','Vel ').replace('position','Pos ').replace('1d','1D ').replace('3d','3D ') for x in gests]
gests_alt = [(g,g+'\n(random)') for g in gests_alt]
gests_alt = [x for y in gests_alt for x in y]
for ax in [ax2]:
    set_axis_style(ax, gests_alt)
fig.tight_layout()
plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.savefig(plot_path)
plt.close('all')