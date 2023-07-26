# 639 and 635 are the same motif
# 639 and 27; 
# 639 and 537; 
# 639 and 405.

# 259 and 115 could hardly be any more similar. 
# 259 include relatively dissimilar motifs, 
	# 53 78 43

# distances = pd.read_csv('/Volumes/MyPassport/FOR_LARA/result_0.1/distances.csv')
# 
from experiments.alapana_dataset_analysis.dtw import plot_dtw
from scipy.signal import savgol_filter

audio_distances_path = os.path.join(out_dir, 'audio_distances.csv')
distances = pd.read_csv(audio_distances_path)
distances = distances.sort_values(by='spectral_centroid').reset_index(drop=True)

i=657
j=653
wms = 125

mi = min([i,j])
mj = max([i,j])

Fs = 44100

rank = distances[(distances['index1']==mj) & (distances['index2']==mi)]
print(rank)
print(f'Rank: {rank.index[0]}/{len(distances)}')


radius = 0.1

track = all_groups[all_groups['index']==i].iloc[0]
t = track.track
start = track.start
end = track.end
audio_track = audio_tracks[t]
x = audio_track[round(start*Fs):round(end*Fs)]

centroid = spectral_centroid(
    y=x, sr=sr, n_fft=2048, hop_length=512, window='hanning', center=True, pad_mode='constant')[0]

sc_timestep = (len(x)/sr)/len(centroid)

wms = 125
wl = round(wms*0.001/sc_timestep)
wl = wl if not wl%2 == 0 else wl+1
s1 = savgol_filter(centroid, polyorder=2, window_length=wl, mode='interp')


track = all_groups[all_groups['index']==j].iloc[0]
t = track.track
start = track.start
end = track.end
audio_track = audio_tracks[t]
x = audio_track[round(start*Fs):round(end*Fs)]

centroid = spectral_centroid(
    y=x, sr=sr, n_fft=2048, hop_length=512, window='hanning', center=True, pad_mode='constant')[0]

sc_timestep = (len(x)/sr)/len(centroid)

wms = 125
wl = round(wms*0.001/sc_timestep)
wl = wl if not wl%2 == 0 else wl+1
s2 = savgol_filter(centroid, polyorder=2, window_length=wl, mode='interp')


pi = len(s1)
pj = len(s2)
l_longest = max([pi, pj])

path_dtw, dtw = dtw_path(s1, s2, radius=radius)

dtw_norm = dtw/len(path_dtw)

plot_dtw(s1, s2, path_dtw, dtw_norm, r=radius, write='plots/sf_dtw.png')

