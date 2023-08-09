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
import cmasher as cmr

colors = cmr.take_cmap_colors('Accent', 5, return_fmt='hex')

from scipy.stats import spearmanr
from exploration.io import create_if_not_exists, load_pkl

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'
results_dir = os.path.join(out_dir, 'analysis', '')
sig_results_df_path = os.path.join(results_dir, 'results_with_random.csv')

results = pd.read_csv(sig_results_df_path)
results=results[(results['p_real']<0.01) & (results['n_real']>109)]

## Plots
gests = [ 
'3dpositionDTWHand', 
'3dvelocityDTWHand', 
'3daccelerationDTWHand',
'3dpositionDTWHead', 
'3dvelocityDTWHead', 
'3daccelerationDTWHead']

results = results[results['y'].isin(gests)]

nm = lambda x: x.replace("DTW", '').replace('acceleration','Acc ').replace('velocity','Vel ').replace('position','Pos ').replace('1d','1D ').replace('3d','3D ')

for sonic_feature in ['diff_pitch_dtw', 'pitch_dtw', 'pitch_dtw_mean', 'loudness_dtw', 'spectral_centroid']:
    sf = {
        'diff_pitch_dtw': 'Δf0', 
        'pitch_dtw': 'f0',
        'pitch_dtw_mean': 'f0 norm',
        'loudness_dtw': 'Loudness', 
        'spectral_centroid': 'Spectral Centroid'
    }[sonic_feature]
    for level in ['all', 'performer']:

        plot_path = os.path.join('plots', 'bar', run_name, f'{sonic_feature}_{level}.png')
        create_if_not_exists(plot_path)
        df = results[results['level']==level]
        df = df[df['x']==sonic_feature]


        df = df[['y', 'level_value', 'corr_real']]

        df['y'] = df['y'].apply(lambda y: nm(y))

        df = df.set_index(['y', 'level_value'])['corr_real']

        ax = df.unstack().plot(kind='bar', color=colors)

        fig = ax.get_figure()
        fig.set_size_inches(15,7)
        fig.subplots_adjust(bottom=0.4, right=1)
        plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})
        text = f"on a {level} level" if not level == 'all' else 'for all performers/performances'

        ax.set_title(f'Signficant correlations between feature and {sf} {text}')
        ax.set_xlabel(f'Feature')
        ax.set_ylabel(f'Spearmans Rank\nCorrelation Coefficient')
        ax.figure.savefig(plot_path, bbox_inches='tight')
        plt.close('all')

    for level in ['performance']:

        plot_path = os.path.join('plots', 'bar', run_name, f'{sonic_feature}_{level}.png')
        create_if_not_exists(plot_path)

        df = results[results['level']==level]
        df = df[df['x']==sonic_feature]

        cats = df['y'].unique()
        
        data = [df[df['y']==c]['corr_real'].values for c in cats]
        
        text = f"on a {level} level" if not level == 'all' else 'for all performers/performances'

        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_title(f'Signficant correlations between feature and {sf} {text}')
        ax.set_xlabel('Gesture feature')
        ax.set_ylabel(f'Spearmans Rank\nCorrelation Coefficient')
        ax.boxplot(data, labels=[nm(c) for c in cats])

        ax.figure.savefig(plot_path, bbox_inches='tight')
        plt.close('all')



