filepaths = [
	'data/pitch_tracks/lara_small/2018_11_13_am_Sec_3_P1_Anandabhairavi.csv',
    'data/pitch_tracks/lara_small/2018_11_18_am_Sec_6_P6_Bilahari_B.csv'
]

stabs = [
    'data/stability_tracks/lara_small/2018_11_13_am_Sec_3_P1_Anandabhairavi.csv',
    'data/stability_tracks/lara_small/2018_11_18_am_Sec_6_P6_Bilahari_B.csv']

import pandas as pd
info = pd.read_csv('audio/lara_wim/info.csv', sep=';')
info = info[~info['Audio file'].isnull()]

import os
from exploration.pitch import silence_stability_from_file
for a,s in zip(filepaths, stabs):
	name = a.split('/')[-1].replace('.csv','')
	t = info[info['Audio file'].str.contains(name)]['Tonic'].values[0]
	print(a)
	print(t)
	silence_stability_from_file(a, s, tonic=t, freq_var_thresh_stab=60, gap_interp=0.350)

