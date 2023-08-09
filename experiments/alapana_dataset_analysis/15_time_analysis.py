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

all_groups_path = os.path.join(out_dir, 'all_groups.csv')

results_dir = os.path.join(out_dir, 'analysis', '')
all_feature_path = os.path.join(results_dir, 'all_features.csv')
sig_results_df_path = os.path.join(results_dir, 'sig_time_results.csv')

create_if_not_exists(sig_results_df_path)

# load data
all_features = pd.read_csv(all_feature_path)
all_groups = pd.read_csv(all_groups_path)

pitch_targets = ['pitch_dtw', 'diff_pitch_dtw']

audio_targets = ['loudness_dtw', 'spectral_centroid']

features = [
       '1dpositionDTWHand',
       '3dpositionDTWHand', '3dpositionDTWHand_mean', '1dvelocityDTWHand',
       '3dvelocityDTWHand', '1daccelerationDTWHand',
       '3daccelerationDTWHand','1dpositionDTWHead',
       '3dpositionDTWHead', '1dvelocityDTWHead',
       '3dvelocityDTWHead', '1daccelerationDTWHead',
       '3daccelerationDTWHead']

targets = pitch_targets + audio_targets

## Add starts and ends
all_features = all_features[all_features['performance1']==all_features['performance2']]

all_features = all_features.merge(all_groups[['index','start']], left_on='index1', right_on='index')
del all_features['index']
all_features = all_features.rename({'start':'start1'}, axis=1)

all_features = all_features.merge(all_groups[['index','end']], left_on='index1', right_on='index')
del all_features['index']
all_features = all_features.rename({'end':'end1'}, axis=1)


all_features = all_features.merge(all_groups[['index','start']], left_on='index2', right_on='index')
del all_features['index']
all_features = all_features.rename({'start':'start2'}, axis=1)

all_features = all_features.merge(all_groups[['index','end']], left_on='index2', right_on='index')
del all_features['index']
all_features = all_features.rename({'end':'end2'}, axis=1)

## Extract time distances
def get_separation(y, starts):
	s1 = y.start1
	s2 = y.start2
	e1 = y.end1
	e2 = y.end2

	if starts:
		return abs(y.start1 - y.start2)
	else:
		if s1 < s2:
			v1 = e1
			v2 = s2
		else:
			v1 = s1
			v2 = e2
		return abs(v1-v2)

all_features['separation_starts'] = all_features.apply(lambda y: get_separation(y, starts=True), axis=1)
all_features['separation_limits'] = all_features.apply(lambda y: get_separation(y, starts=False), axis=1)


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


features = features + targets
targets = ['separation_starts', 'separation_limits']

results = pd.DataFrame(columns=['x', 'y', 'level', 'level_value', 'corr', 'p', 'n'])
for x in targets:
    print(x)
    for y in tqdm.tqdm(features):
        corr_dict = get_correlation(all_features, x, y, None)
        for k,v in corr_dict.items():
            to_append = {
                'x': x,
                'y': y,
                'level': 'all',
                'level_value': k,
                'corr': v['corr'],
                'p': v['p'],
                'n': v['n']
            }
            results = results.append(to_append, ignore_index=True)

results = results.sort_values(by='corr', ascending=False)



## Plots
########

## Best Results
###############
# For 70% of estimates to be within +/- 0.1 of the true correlation value (between -0.1 and 0.1), we need at least 109 observations
# for 90% of estimates to be within +/- 0.2 of the true correlation value (between -0.2 and 0.2), we need at least 70 observations. 
p = 0.01
significant_results = results[(results['p']<p) & (results['n']>109)]
significant_results.to_csv(sig_results_df_path, index=False)

## Plots
########
for i, row in tqdm.tqdm(list(significant_results.iterrows())):
    x = row['x']
    y = row['y']
    level = row['level']
    level_value = row['level_value']
    corr = row['corr']
    p = row['p']
    n = row['n']

    level_str = f'{level}={level_value}' if level != 'all' else level
    title = f'{x} against {y} for {level_str}'
    title += f'\n [spearmans R={round(corr,3)}, p={round(p,3)}, n={n}]'
    
    out_path = os.path.join(results_dir, 'time_plots', x, y, level, (level if level == 'all' else str(level_value)) + '.png')
    create_if_not_exists(out_path)

    if level != 'all':
        cols = [x for x in all_features.columns if level in x]
        data = all_features[(all_features[cols[0]]==level_value) & (all_features[cols[1]]==level_value)]
    else:
        data = all_features

    pl = sns.scatterplot(data, x=x, y=y, s=5)
    pl.set_title(title)
    fig = pl.get_figure()
    fig.savefig(out_path)
    plt.cla()
    plt.clf()
    plt.close()
