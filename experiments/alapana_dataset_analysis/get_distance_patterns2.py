import pandas as pd
from exploration.pitch import get_timeseries, pitch_seq_to_cents,interpolate_below_length
from exploration.io import create_if_not_exists
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import librosa
import os
import fastdtw
import numpy as np
import tqdm
from scipy.signal import savgol_filter
from experiments.alapana_dataset_analysis.dtw import dtw_path, dtw_dtai
from scipy.ndimage import gaussian_filter1d

run_name = 'result_0.1'


out_dir = f'/Volumes/MyPassport/FOR_LARA/{run_name}/'

track_names = [
    "2018_11_13_am_Sec_1_P2_Varaali_slates",
    "2018_11_13_am_Sec_2_P1_Shankara_slates",
    "2018_11_13_am_Sec_3_P1_Anandabhairavi_slates",
    "2018_11_13_am_Sec_3_P3_Kalyani_slates",
    "2018_11_13_am_Sec_3_P5_Todi_slates",
    "2018_11_13_am_Sec_3_P8_Bilahari_B_slates",
    "2018_11_13_am_Sec_4_P1_Atana_slatesB",
    "2018_11_13_pm_Sec_1_P1_Varaali_slates",
    "2018_11_13_pm_Sec_1_P2_Anandabhairavi_slates",
    "2018_11_13_pm_Sec_2_P1_Kalyani_slates",
    "2018_11_13_pm_Sec_3_P1_Todi_A_slates",
    "2018_11_13_pm_Sec_3_P2_Todi_B_slates",
    "2018_11_13_pm_Sec_4_P1_Shankara_slates",
    "2018_11_13_pm_Sec_5_P1_Bilahari_slates",
    "2018_11_13_pm_Sec_5_P2_Atana_slates",
    "2018_11_15_Sec_10_P1_Atana_slates",
    "2018_11_15_Sec_12_P1_Kalyani_slates",
    "2018_11_15_Sec_13_P1_Bhairavi_slates",
    "2018_11_15_Sec_1_P1_Anandabhairavi_A_slates",
    "2018_11_15_Sec_3_P1_Anandabhairavi_C_slates",
    "2018_11_15_Sec_4_P1_Shankara_slates",
    "2018_11_15_Sec_6_P1_Varaali_slates",
    "2018_11_15_Sec_8_P1_Bilahari_slates",
    "2018_11_15_Sec_9_P1_Todi_slates",
    "2018_11_18_am_Sec_1_P1_Varaali_slates",
    "2018_11_18_am_Sec_2_P2_Shankara_slates",
    "2018_11_18_am_Sec_3_P1_Anandabhairavi_A_slates",
    "2018_11_18_am_Sec_3_P2_Anandabhairavi_B_slates",
    "2018_11_18_am_Sec_4_P1_Kalyani_slates",
    "2018_11_18_am_Sec_5_P1_Bhairavi_slates",
    "2018_11_18_am_Sec_5_P3_Bilahari_slates",
    "2018_11_18_am_Sec_6_P1_Atana_A_slates",
    "2018_11_18_am_Sec_6_P2_Atana_B_slates",
    "2018_11_18_am_Sec_6_P4_Todi_slates",
    "2018_11_18_am_Sec_6_P6_Bilahari_B",
    "2018_11_18_pm_Sec_1_P1_Shankara_slates",
    "2018_11_18_pm_Sec_1_P2_Varaali_slates",
    "2018_11_18_pm_Sec_1_P3_Bilahari_slates",
    "2018_11_18_pm_Sec_2_P2_Anandabhairavi_full_slates",
    "2018_11_18_pm_Sec_3_P1_Kalyani_slates",
    "2018_11_18_pm_Sec_4_P1_Bhairavi_slates",
    "2018_11_18_pm_Sec_4_P2_Atana_slates",
    "2018_11_18_pm_Sec_5_P1_Todi_full_slates",
    "2018_11_18_pm_Sec_5_P2_Sahana_slates"
]

metadata_path = 'audio/lara_wim2/info.csv'
metadata = pd.read_csv(metadata_path)
metadata = metadata[~metadata['Audio file'].isnull()]
metadata = metadata[~metadata['Tonic'].isnull()]
tonic_dict = {k:v for k,v in metadata[['Audio file', 'Tonic']].values}

def get_tonic(t, metadata):
    tonic = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Tonic'].values[0]
    return tonic

def get_raga(t, metadata):
    raga = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Raga'].values[0]
    return raga

def get_derivative(pitch, time):

    d_pitch = np.array([((pitch[i+1]-pitch[i])+((pitch[i+2]-pitch[i+1])/2))/2 for i in range(len(pitch)-2)])
    d_time = time[1:-1]

    return d_pitch, d_time

pitch_tracks = {}
for t in track_names:
    if not "2018_11_19" in t:
        p_path = f'/Volumes/MyPassport/cae-invar/data/pitch_tracks/alapana/{t}.csv'
        tonic = get_tonic(t, metadata)
        pitch, time, timestep = get_timeseries(p_path)
        pitch = pitch_seq_to_cents(pitch, tonic=tonic)
        pitch[pitch==None]=0
        pitch = interpolate_below_length(pitch, 0, (350*0.001/timestep))
        pitch_d, time_d = get_derivative(pitch, time)
        pitch_tracks[t] = (pitch, time, timestep, pitch_d, time_d)
        #pitch_tracks[t] = (gaussian_filter1d(pitch, 2.5), time, timestep, gaussian_filter1d(pitch_d, 2.5), time_d)

all_patts = pd.read_csv(os.path.join(out_dir, 'all_groups.csv'))
all_patts = all_patts[all_patts['track'].isin(track_names)]
#all_distances = pd.DataFrame(columns=['index1', 'index2', 'path1_start', 'path1_end', 'path2_start', 'path2_end', 'path_length', 'dtw_distance', 'dtw_distance_norm'])

distances_path = os.path.join(out_dir, f'distances.csv')

try:
    print('Removing previous distances file')
    os.remove(distances_path)
except OSError:
    pass
create_if_not_exists(distances_path)


def trim_zeros(pitch, time):
    m = pitch!=0
    i1,i2 = m.argmax(), m.size - m[::-1].argmax()
    return pitch[i1:i2], time[i1:i2]


def smooth(pitch, time, timestep, wms=125):
    pitch2, time2 = trim_zeros(pitch, time)
    wl = round(wms*0.001/timestep)
    wl = wl if not wl%2 == 0 else wl+1
    interp = savgol_filter(pitch2, polyorder=2, window_length=wl, mode='interp')
    return interp, time2


r=0.1

##text=List of strings to be written to file
header = 'index1,index2,pitch_dtw,diff_pitch_dtw'
with open(distances_path,'a') as file:
    file.write(header)
    file.write('\n')

    for i, row in tqdm.tqdm(list(all_patts.iterrows())):

        qstart = row.start
        qend = row.end
        qtrack = row.track
        qi = row['index']
        (qpitch, qtime, qtimestep, qpitch_d, qtime_d) = pitch_tracks[qtrack]

        sq1 = int(qstart/qtimestep)
        sq2 = int(qend/qtimestep)
        for j, rrow in all_patts.iterrows():

            rstart = rrow.start
            rend = rrow.end
            rtrack = rrow.track
            rj = rrow['index']
            if qi <= rj:
                continue
            (rpitch, rtime, rtimestep, rpitch_d, rtime_d) = pitch_tracks[rtrack]
            sr1 = int(rstart/rtimestep)
            sr2 = int(rend/rtimestep)

            pat1 = qpitch[sq1:sq2]
            pat1_time = qtime[sq1:sq2]
            pat2 = rpitch[sr1:sr2]
            pat2_time = rtime[sr1:sr2]
            
            pat1[pat1 == None] = 0
            pat2[pat2 == None] = 0
 
            pat1, pat1_time = smooth(pat1, pat1_time, rtimestep)
            pat2, pat2_time = smooth(pat2, pat2_time, qtimestep)
            
            pi = len(pat1)
            pj = len(pat2)
            l_longest = max([pi, pj])
            
            diff1,_ = get_derivative(pat1, pat1_time)
            diff2,_ = get_derivative(pat2, pat2_time)

            diff1, diff1_time = smooth(diff1, pat1_time, rtimestep)
            diff2, diff2_time = smooth(diff2, pat2_time, qtimestep)

            path, dtw_val = dtw_path(pat1, pat2, radius=int(l_longest*r))

            l = len(path)
            dtw_norm = dtw_val/l

            path, dtw_val = dtw_path(diff1, diff2, radius=int(l_longest*r))
            l = len(path)
            dtw_norm_diff = dtw_val/l

            line =f"{qi},{rj},{dtw_norm},{dtw_norm_diff}"
            #all_distances = all_distances.append({
            #   'index1':i,
            #   'index2':j,
            #   'path1_start':path1_start,
            #   'path1_end':path1_end,
            #   'path2_start':path2_start,
            #   'path2_end':path2_end,
            #   'path_length': l,
            #   'dtw_distance':dtw_val,
            #   'dtw_distance_norm':dtw_norm
            #}, ignore_index=True)
            file.write(line)
            file.write('\n')

            #plt.plot(pat1, range(len(pat1)))
            #plot_path = f'plots/testing_smoothing/{qi}.png'
            #create_if_not_exists(plot_path)
            #plt.savefig(plot_path)
            #plt.close('all')

    #all_distances.reset_index(inplace=True, drop=True)
