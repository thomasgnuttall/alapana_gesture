import os

import pandas as pd 
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import fastdtw

import dtaidistance.dtw
from experiments.alapana_dataset_analysis.dtw import dtw_path, dtw_dtai
from exploration.io import write_pkl

from scipy.ndimage import gaussian_filter1d

segment_mapping = {
    'Seg_1':  'Pelvis', # important
    'Seg_2':  'L5',
    'Seg_3':  'L3',
    'Seg_4':  'T12',
    'Seg_5':  'T8',
    'Seg_6':  'Neck',
    'Seg_7':  'Head',
    'Seg_8':  'RightShoulder', 
    'Seg_9':  'RightUpperArm',
    'Seg_10':  'RightForearm', # important
    'Seg_11':  'RightHand', # important
    'Seg_12':  'LeftShoulder', 
    'Seg_13':  'LeftUpper Arm',
    'Seg_14':  'LeftForearm', # important
    'Seg_15':  'LeftHand', # important
    'Seg_16':  'RightUpperLeg',
    'Seg_17':  'RightLowerLeg',
    'Seg_18':  'RightFoot',
    'Seg_19':  'RightToe',
    'Seg_20':  'LeftUpperLeg',
    'Seg_21':  'LeftLowerLeg',
    'Seg_22':  'LeftFoot',
    'Seg_23':  'LeftToe'
}


def pivot_mocap(df):
    df = df.pivot_table(
        values=['x','y','z'], 
        index=['time', 'time_ms', 'feature'], 
        columns=['segment']).reset_index()

    return df

def get_mocap_path(track_name, substring='Acceleration'):
    return [p for p in mocap_paths if track_name[:-3].replace('_','') in p.replace('_','') and substring in p][0]

idict = lambda y: all_groups[all_groups['index']==y].iloc[0].to_dict()

def report_handedness(all_groups):
    n = len(all_groups)
    n_right = round(len(all_groups[all_groups['handedness']=='Right'])*100/n,2)
    n_left = round(len(all_groups[all_groups['handedness']=='Left'])*100/n,2)

    ratio_mean = all_groups['handedness_ratio'].mean()
    ratio_std = all_groups['handedness_ratio'].std()

    n1 = round(sum(all_groups['handedness_ratio']>1)*100/n,2)
    n1_2 = round(sum(all_groups['handedness_ratio']>1.2)*100/n,2)
    n2 = round(sum(all_groups['handedness_ratio']>2)*100/n,2)
    n10 = round(sum(all_groups['handedness_ratio']>10)*100/n,2)
    n100 = round(sum(all_groups['handedness_ratio']>100)*100/n,2)

    print(f'For {n} motifs...')
    print('')
    print(f'{n_left}% are identified as Left handed')
    print(f'{n_right}% are identified as Right handed')
    print('')
    print('The ratio in energy between the identified dominant hand and non-dominant hand has')
    print(f'mean={ratio_mean}')
    print(f'std={ratio_std}')
    print('')
    print(f'The proportion of motifs with a ratio greater than 1 is {n1}%')
    print(f'The proportion of motifs with a ratio greater than 1.2 is {n1_2}%')
    print(f'The proportion of motifs with a ratio greater than 2 is {n2}%')
    print(f'The proportion of motifs with a ratio greater than 10 is {n10}%')
    print(f'The proportion of motifs with a ratio greater than 100 is {n100}%')

def get_motion_energy(i, hand):
    d = idict(i)
    track_name = d['track']
    end = d['end']
    start = d['start']
    mp = get_mocap_path(track_name, substring='Velocity')
    df = mocap[mp]
    min_t = df['time_ms'].iloc[0]
    
    df['time_s'] = df['time_ms'].apply(lambda y: (y-min_t)/1000)

    this_frame = df[(df['time_s']>=start) & (df['time_s']<=end)]

    this_frame = this_frame.loc[:, (slice(None), hand+'Hand')]
    vectors = this_frame.values
    return np.sum(np.apply_along_axis(lambda y: np.linalg.norm(y)**2, 1, vectors))/len(vectors)


def rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = 0,0
    px, py, pz = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy, pz


def get_motion_distance(i, j, feature):
    try:
        i_vec = index_features[i][feature]#get_motion_data(i, level, feature, handi, num_dims=num_dims, pelvis=pelvis1, angle=angle1)
        j_vec = index_features[j][feature]#get_motion_data(j, level, feature, handj, num_dims=num_dims, pelvis=pelvis2, angle=angle2)
        
        pi = len(i_vec)
        pj = len(j_vec)
        l_longest = max([pi, pj])
        #dtw_val, path = dtaidistance.dtw.distance(i_vec, j_vec, window=round(l_longest*0.20), use_c=True)
        #dtw_val, path = fastdtw.fastdtw(i_vec, j_vec, dist=None, radius=round(l_longest*0.20))
        path, dtw_val = dtw_dtai(i_vec, j_vec, r=0.1)

        return dtw_val/len(path)
    except Exception as e:
        raise e
        #return np.nan


# Palm, wrist, elbow, shoulder
desirable_segments = ['Seg_1', 'Seg_10', 'Seg_11', 'Seg_14', 'Seg_15', 'Seg_8', 'Seg_12', 'Seg_7']

mocap_dir = '/Volumes/MyPassport/gesture_network_data/mocap/position_velocity_acceleration/'

mocap_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(mocap_dir) \
                for f in filenames if os.path.splitext(f)[1] == '.txt']

columns = ['time', 'time_ms', 'feature', 'segment', 'x', 'y', 'z']
mocap = {}
for mp in tqdm.tqdm(mocap_paths):
    df = pd.read_csv(mp, sep='\t', encoding="ISO-8859-1")
    if len(df) == 0:
        print(f'missing: {mp}')
        continue

    df.columns = columns

    df = df[df['segment'].isin(desirable_segments)]

    df['segment'] = df['segment'].apply(lambda y: segment_mapping[y])

    df = pivot_mocap(df)

    mocap[mp] = df







from exploration.sequence import is_stable, reduce_stability_mask, add_center_to_mask

## Identify stability
#def get_stability_mask(sequence, min_stability_length_secs, stability_hop_secs, var_thresh, timestep):
#    stab_hop = int(stability_hop_secs/timestep)
#    reverse_sequence = np.flip(sequence)

#    # apply in both directions to array to account for hop_size errors
#    stable_mask_1 = [is_stable(sequence[s:s+stab_hop], var_thresh) for s in range(len(sequence))]
#    stable_mask_2 = [is_stable(sequence[s:s+stab_hop], var_thresh) for s in range(len(sequence))]
#    
#    zipped = zip(stable_mask_1, np.flip(stable_mask_2))
#    
#    stable_mask = np.array([int(any([s1,s2])) for s1,s2 in zipped])

#    stable_mask = reduce_stability_mask(stable_mask, min_stability_length_secs, timestep)
#    
#    return stable_mask

def is_stable(seq, max_var):
    if None in seq:
        return 0
    seq = seq.astype(float)
    mu = np.nanmean(seq)

    maximum = np.nanmax(seq)
    minimum = np.nanmin(seq)
    if (maximum < mu + max_var) and (minimum > mu - max_var):
        return 1
    else:
        return 0

## Identify stability
def get_stability_mask(sequence, min_stability_length_secs, stability_hop_secs, var_thresh, timestep):
    stab_hop = int(stability_hop_secs/timestep)
    reverse_sequence = np.flip(sequence)

    # apply in both directions to array to account for hop_size errors
    stable_mask_1 = [is_stable(sequence[s:s+stab_hop], var_thresh) for s in range(len(sequence))]
    stable_mask_2 = [is_stable(sequence[s:s+stab_hop], var_thresh) for s in range(len(sequence))]
    
    zipped = zip(stable_mask_1, np.flip(stable_mask_2))
    
    stable_mask = np.array([int(any([s1,s2])) for s1,s2 in zipped])

    stable_mask = reduce_stability_mask(stable_mask, min_stability_length_secs, timestep)
    
    return stable_mask

 
level = 'velocity'
track_name = '2018_11_13_pm_Sec_5_P1_Bilahari'
mp = get_mocap_path(track_name, substring=level)
df = mocap[mp]

# isolated gesture tracks
time = df['time_ms'].values
time = time-time[0]
time = time*0.001
right = df.loc[:, (slice(None), 'RightHand')]
left  = df.loc[:, (slice(None), 'LeftHand')]

right_x = right['x']['RightHand'].values
right_y = right['y']['RightHand'].values
right_z = right['z']['RightHand'].values

left_x = left['x']['LeftHand'].values
left_y = left['y']['LeftHand'].values
left_z = left['z']['LeftHand'].values

timestep = time[2]-time[1]
min_stability_length_secs = 0.5
stability_hop_secs = 0.5
var_thresh = 0.4

stab_left_y  = get_stability_mask(abs(left_y), min_stability_length_secs, stability_hop_secs, var_thresh, timestep)

time2 = time[200:1800]
pitch2 = left_y[200:1800]
filtered  = gaussian_filter1d(pitch2, 10)
plt.plot(time2, filtered)
plt.plot(time2, stab_left_y[200:1800])
plt.savefig('stab_test.png')
plt.close('all')

#stab_right_x = get_stability_mask(right_x, min_stability_length_secs, stability_hop_secs, var_thresh, timestep)
#stab_right_y = get_stability_mask(right_y, min_stability_length_secs, stability_hop_secs, var_thresh, timestep)
#stab_right_z = get_stability_mask(right_z, min_stability_length_secs, stability_hop_secs, var_thresh, timestep)
#stab_left_x  = get_stability_mask(left_x, min_stability_length_secs, stability_hop_secs, var_thresh, timestep)
stab_left_y  = get_stability_mask(left_y, min_stability_length_secs, stability_hop_secs, var_thresh, timestep)
#stab_left_z  = get_stability_mask(left_z, min_stability_length_secs, stability_hop_secs, var_thresh, timestep)

stab_right = np.array([int(all([x,y,z])) for x,y,z in zip(stab_right_x, stab_right_y, stab_right_z)])
stab_right = add_center_to_mask(stab_right)
stab_left  = np.array([int(all([x,y,z])) for x,y,z in zip(stab_left_x, stab_left_y, stab_left_z)])
stab_left = add_center_to_mask(stab_left_y)


splits = np.where(stab_left==2)[0]

chunks = []
for i,s in enumerate(range(len(splits))):
    if i == 0:
        chunks.append((time[0],time[splits[s]]))
    else:
        chunks.append((time[splits[s-1]],time[splits[s]]))













time2 = time[680:680+200]
sequence = right_x[680:680+200]

filtered  = gaussian_filter1d(sequence, 1.5)
plt.plot(time2, filtered)
plt.savefig('stab_test.png')
plt.close('all')


stab_hop = int(stability_hop_secs/timestep)
reverse_sequence = np.flip(sequence)

# apply in both directions to array to account for hop_size errors
stable_mask_1 = [is_stable(sequence[s:s+stab_hop], var_thresh) for s in range(len(sequence))]
stable_mask_2 = [is_stable(sequence[s:s+stab_hop], var_thresh) for s in range(len(sequence))]

zipped = zip(stable_mask_1, np.flip(stable_mask_2))

stable_mask = np.array([int(any([s1,s2])) for s1,s2 in zipped])

stable_mask = reduce_stability_mask(stable_mask, min_stability_length_secs, timestep)



