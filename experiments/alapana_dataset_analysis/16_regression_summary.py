run_name = 'result_0.1'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

sns.set_theme()

from scipy.stats import spearmanr
from exploration.io import create_if_not_exists
from exploration.visualisation import flush_matplotlib

hand_path = os.path.join('experiments', 'alapana_dataset_analysis', 'results', 'final', 'just_hand_results_r2.csv')
head_path = os.path.join('experiments', 'alapana_dataset_analysis', 'results', 'final', 'just_head_results_r2.csv')
all_path = os.path.join('experiments', 'alapana_dataset_analysis', 'results', 'final', 'results_r2.csv')
random_path = os.path.join('experiments', 'alapana_dataset_analysis', 'results', 'final', 'random_results_r2.csv')

# load data
head = pd.read_csv(head_path)
hand = pd.read_csv(hand_path)
results = pd.read_csv(all_path)
random = pd.read_csv(random_path)

level = 'all'

head_score = head[(head['level']=='all') & (head['target']=='pitch_dtw')].iloc[0]['test_score']
hand_score = hand[(hand['level']=='all') & (hand['target']=='pitch_dtw')].iloc[0]['test_score']
random_score = random[(random['level']=='all') & (random['target']=='pitch_dtw')].iloc[0]['test_score']
all_score = results[(results['level']=='all') & (results['target']=='pitch_dtw')].iloc[0]['test_score']

print('Regression Results predicting f0')
print(f'Random R2: {random_score}')
print(f'Head R2: {head_score}')
print(f'Hand R2: {hand_score}')
print(f'Head and Hand R2: {all_score}')