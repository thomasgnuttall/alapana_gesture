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
random_results_path = os.path.join(results_dir, 'results_with_random.csv')

# load data
all_groups = pd.read_csv(all_groups_path)
results = pd.read_csv(random_results_path)


for level in ['all', 'performer', 'performance']:
	print(f'On the {level} level...')
	df = results[results['level']==level]
	n = len(df)
	df = df[(df['p_real']<=0.01)]
	n_sig = len(df)
	print(f'    {n_sig}/{n} tests provided a significant correlation in the real data')
	df = df[(df['corr_real']>=0.2)]
	n_corr = len(df)
	print(f'    {n_corr}/{n_sig} significant tests provided a correlation > 0.2')

	df = results[results['level']==level]
	n = len(df)
	df = df[(df['p_random']<=0.01)]
	n_sig = len(df)
	print(f'    {n_sig}/{n} tests provided a significant correlation in the randomised data')
	df = df[(df['corr_random']>=0.2)]
	n_corr = len(df)
	print(f'    {n_corr}/{n_sig} significant tests provided a correlation > 0.2')
