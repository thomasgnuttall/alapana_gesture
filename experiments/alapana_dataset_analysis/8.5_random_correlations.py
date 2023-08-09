run_name = 'result_0.1'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
sns.set_theme()

from scipy.stats import spearmanr
from exploration.io import create_if_not_exists, load_pkl

out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'
distances_path = os.path.join(out_dir, 'distances_gestures.csv')
index_features_path = os.path.join(out_dir, 'index_features.pkl')
all_groups_path = os.path.join(out_dir, 'all_groups.csv')
audio_features_path = os.path.join(out_dir, 'audio_distances.csv')
results_dir = os.path.join(out_dir, 'analysis', '')
merged_path = os.path.join(results_dir, 'results_with_random.csv')
sig_results_df_path = os.path.join(results_dir, 'results.csv')


# load data
distances = pd.read_csv(distances_path)
all_groups = pd.read_csv(all_groups_path)
audio_features = pd.read_csv(audio_features_path)
index_features = load_pkl(index_features_path)

all_groups['performer'] = all_groups['display_name'].apply(lambda y: y.split('_')[0])
counts = all_groups.groupby('performer')['index'].count().to_dict()
mc = min(counts.values())
pind = []
for p in counts.keys():
    ixs = all_groups[all_groups['performer']==p].index
    pind += list(np.random.choice(ixs, mc, replace=False))

all_groups['include_in_all'] = all_groups['index'].isin(pind)

pitch_targets = ['pitch_dtw', 'pitch_dtw_mean', 'diff_pitch_dtw']

audio_targets = ['loudness_dtw', 'spectral_centroid']

features = [
       '1dpositionDTWHand',
       '3dpositionDTWHand', '1dvelocityDTWHand',
       '3dvelocityDTWHand', '1daccelerationDTWHand',
       '3daccelerationDTWHand','1dpositionDTWHead',
       '3dpositionDTWHead', '1dvelocityDTWHead',
       '3dvelocityDTWHead', '1daccelerationDTWHead',
       '3daccelerationDTWHead']

audio_distance = distances.merge(audio_features, on=['index1', 'index2'])

targets = pitch_targets + audio_targets

for f in features:
    audio_distance[f] = audio_distance[f].sample(frac=1).values

## Remove mismatched length for analysis
########################################
def len_mismatch(l1, l2):
    l_longest = max([l1, l2])
    l_shortest = min([l1, l2])

    return l_longest/l_shortest-1 > 0.5

audio_distance['length_mismatch'] = audio_distance.apply(lambda y: len_mismatch(y.length1, y.length2), axis=1)

audio_distance_cut = audio_distance[audio_distance['length_mismatch']!=True]

audio_distance_cut = audio_distance_cut.merge(all_groups[['index','include_in_all']], left_on='index1', right_on='index')
del audio_distance_cut['index']
audio_distance_cut = audio_distance_cut.rename({'include_in_all':'include_in_all1'}, axis=1)

audio_distance_cut = audio_distance_cut.merge(all_groups[['index','include_in_all']], left_on='index2', right_on='index')
del audio_distance_cut['index']
audio_distance_cut = audio_distance_cut.rename({'include_in_all':'include_in_all2'}, axis=1)

## Get correlations
###################
def get_correlation(df, x, y, level=None):
    corr_dict = {}
    if level:
        cols = [x for x in df.columns if level in x]
        uniqs = set()
        for c in cols:
            uniqs = uniqs.union(set(df[c].unique())) 
        for u in uniqs:
            this_df = df[(df[cols[0]]==u) & (df[cols[1]]==u)]
            res = spearmanr(a=this_df[x].values, b=this_df[y].values, axis=0, nan_policy='omit')
            corr_dict[u] = {'corr': res.correlation, 'p': res.pvalue, 'n': len(this_df)} 
    else:
        res = spearmanr(a=df[x].values, b=df[y].values, axis=0, nan_policy='omit')
        corr_dict['all'] = {'corr': res.correlation, 'p': res.pvalue, 'n': len(df)}
    return corr_dict


levels = [None, 'performer', 'performance']

results = pd.DataFrame(columns=['x', 'y', 'level', 'level_value', 'corr', 'p', 'n'])
for x in targets:
    print(x)
    for y in tqdm.tqdm(features):
        for level in levels:
            if level==None:
                df = audio_distance_cut[(audio_distance_cut['include_in_all1']==True) & (audio_distance_cut['include_in_all2']==True)]
            else:
                df = audio_distance_cut
            corr_dict = get_correlation(df, x, y, level)
            for k,v in corr_dict.items():
                to_append = {
                    'x': x,
                    'y': y,
                    'level': level if level else 'all',
                    'level_value': k,
                    'corr': v['corr'],
                    'p': v['p'],
                    'n': v['n']
                }
                results = results.append(to_append, ignore_index=True)

results = results.sort_values(by='corr', ascending=False)

results = results[results['n']>109]


## Get real results
####################

real_results = pd.read_csv(sig_results_df_path)

merged = real_results.merge(results, on=['x','y','level','level_value'], suffixes=['_real', '_random'])
merged['corr_ratio'] = merged['corr_real']/merged['corr_random']


merged.to_csv(merged_path, index=False)