# 639 and 635 are the same motif
# 639 and 27; 
# 639 and 537; 
# 639 and 405.

# 259 and 115 could hardly be any more similar. 
# 259 include relatively dissimilar motifs, 
	# 53 78 43

# distances = pd.read_csv('/Volumes/MyPassport/FOR_LARA/result_0.1/distances.csv')
# 

distances = pd.read_csv('/Volumes/MyPassport/FOR_LARA/result_0.1/distances.csv')

i=1092
j=1079

deriv = True

mi = min([i,j])
mj = max([i,j])

if deriv:
	PST = 'diff_pitch_dtw'
else:
	PST = 'pitch_dtw'

distances = distances.sort_values(by=PST).reset_index(drop=True)

rank = distances[(distances['index1']==mj) & (distances['index2']==mi)]
print(rank)
print(f'Rank for {PST}: {rank.index[0]}/{len(distances)}')


radius_factor = 0.1

track = all_groups[all_groups['index']==i].iloc[0]
track_name = track.track
start = track.start
end = track.end
tonic = track.tonic

(pitch, time, timestep, pitchd, timed) =  pitch_tracks[track_name]

start_seq = int(start/timestep)
end_seq = int(end/timestep)
length_seq = end_seq - start_seq


if deriv:
	s1 = pitch[start_seq:end_seq]
	s1_time = time[start_seq:end_seq]
	s1, s1_time = smooth(s1, s1_time)
	s1,s1_time = get_derivative(s1, s1_time)
	#s1, s1_time = smooth(s1, s1_time)
else:
	s1 = pitch[start_seq:end_seq]
	s1_time = time[start_seq:end_seq]
	s1, s1_time = smooth(s1, s1_time)

track = all_groups[all_groups['index']==j].iloc[0]
track_name = track.track
start = track.start
end = track.end
tonic = track.tonic

(pitch, time, timestep, pitchd, timed) =  pitch_tracks[track_name]

start_seq = int(start/timestep)
end_seq = int(end/timestep)
length_seq = end_seq - start_seq

if deriv:
	s2 = pitch[start_seq:end_seq]
	s2_time = time[start_seq:end_seq]
	s2, s2_time = smooth(s2, s2_time)
	s2,s2_time = get_derivative(s2, s2_time)
	#s2, s1_time = smooth(s2, s2_time)
else:
	s2 = pitch[start_seq:end_seq]
	s2_time = time[start_seq:end_seq]
	s2, s2_time = smooth(s2, s2_time)

#from experiments.alapana_dataset_analysis.plotdtw import plot_dtw
pi = len(s1)
pj = len(s2)
l_longest = max([pi, pj])
radius = int(l_longest*radius_factor)
r = radius
path_dtw, dtw = dtw_path(s1, s2, radius=radius)

dtw_norm = dtw/len(path_dtw)

plot_dtw(s1, s2, path_dtw, dtw_norm, r=radius, write='dtw_new_implementation2.png')

