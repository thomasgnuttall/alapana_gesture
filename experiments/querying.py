from complex_auto.util import load_pyc_bz, to_numpy

import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy.signal import convolve2d

#%load_ext autoreload
#%autoreload 2
import sys
import datetime
import os
import pickle

import skimage.io

from complex_auto.motives_extractor import *
from complex_auto.motives_extractor.extractor import *

from exploration.pitch import extract_pitch_track
from exploration.img import (
    remove_diagonal, convolve_array, binarize, diagonal_gaussian, 
    hough_transform, hough_transform_new, scharr, sobel,
    apply_bin_op, make_symmetric, edges_to_contours)
from exploration.segments import (
    extract_segments_new, get_all_segments, break_all_segments, do_patterns_overlap, reduce_duplicates, 
    remove_short, extend_segments, join_all_segments, extend_groups_to_mask, group_segments, group_overlapping,
    group_by_distance, trim_silence)
from exploration.sequence import (
    apply_exclusions, contains_silence, min_gap, too_stable, 
    convert_seqs_to_timestep, get_stability_mask, add_center_to_mask,
    remove_below_length, extend_to_mask, add_border_to_mask)
from exploration.evaluation import evaluate, load_annotations_new, get_coverage, get_grouping_accuracy
from exploration.visualisation import plot_all_sequences, plot_pitch, flush_matplotlib
from exploration.io import load_sim_matrix, write_all_sequence_audio, load_yaml, load_pkl, create_if_not_exists, write_pkl, run_or_cache
from exploration.pitch import cents_to_pitch, pitch_seq_to_cents, pitch_to_cents, get_timeseries, interpolate_below_length

from skimage.measure.entropy import shannon_entropy
import tqdm


######### PARAMETERS #########
# Sample rate of audio
sr = 44100

# size in frames of cqt window from convolution model
cqt_window = 1984 # was previously set to 1988

# Take sample of data, set to None to use all data
s1 = None # lower bound index (5000 has been used for testing)
s2 = None # higher bound index (9000 has been used for testing)

# pitch track extraction
gap_interp = 350*0.001 # Interpolate pitch tracks gaps of <gap_interp>seconds or less [set to None to skip]

# stability identification
stab_hop_secs = 0.2 # window size for stab computations in seconds
min_stability_length_secs = 0.5 # minimum legnth of region to be considered stable in seconds
freq_var_thresh_stab = 10 # max variation in pitch to be considered stable region

conv_filter_str = 'sobel'

# Binarize raw sim array 0/1 below and above this value...
# depends completely on filter passed to convolutional step
# Best...
#   scharr, 0.56
#   sobel unidrectional, 0.1
#   sobel bidirectional, 0.15
bin_thresh = 0.16
# lower bin_thresh for areas surrounding segments
bin_thresh_segment = 0.08
# percentage either size of a segment considered for lower bin thresh
perc_tail = 0.5

# Gaussian filter along diagonals with sigma...
gauss_sigma = None

# After gaussian, re-binarize with this threshold
cont_thresh = 0.15

# morphology params
etc_kernel_size = 10 # For closing
binop_dim = 3 # square dimension of binary opening structure (square matrix of zeros with 1 across the diagonal)

# Distance between consecutive diagonals to be joined in seconds
min_diff_trav = 0.5 #0.1
min_diff_trav_hyp = (2*min_diff_trav**2)**0.5 # translate min_diff_trav to corresponding diagonal distance
min_diff_trav_seq = min_diff_trav_hyp*sr/cqt_window

# extend to silent/stability mask using this proportion of pattern
ext_mask_tol = 0.5 

min_pattern_length_seconds = 1
min_length_cqt = min_pattern_length_seconds*sr/cqt_window

track2 = 'Vanajaksha Ninne Kori'
track1 = 'Koti Janmani'

ampsA, phaseA = load_pyc_bz(f"data/self_similarity/{track1}/features.pyc.bz")

ampsB, phaseB = load_pyc_bz(f"data/self_similarity/{track2}/features.pyc.bz")

pitchA, timeA, timestepA = get_timeseries(f"data/pitch_tracks/{track1}.csv")
pitchB, timeB, timestepB = get_timeseries(f"data/pitch_tracks/{track2}.csv")

pitchA[np.where(pitchA<80)[0]]=0
pitchA = interpolate_below_length(pitchA, 0, (gap_interp/timestepA))

pitchB[np.where(pitchB<80)[0]]=0
pitchB = interpolate_below_length(pitchB, 0, (gap_interp/timestepB))


def get_pattern_crosssim(s1, l1, feat1, feat2):
    feat_pat = feat1[s1:s1+l1]

    matrix = cdist(np.vstack(to_numpy(feat_pat)), np.vstack(to_numpy(feat2)), metric='cosine')

    matrix = 1 / (matrix + 1e-6)

    flength = 10
    ey = np.eye(flength) + np.eye(flength, k=1) + np.eye(flength, k=-1)

    matrix = convolve2d(matrix, ey, mode="same")
    mat_min = np.min(matrix)
    mat_max = np.max(matrix)

    matrix -= matrix.min()
    matrix /= (matrix.max() + 1e-8)

    return matrix


def entropy(X):
    h,w = X.shape
    ey = [np.eye(h,h*2,k=i)for i in np.arange(-int(h/2), int(h/2))]
    ey = sum(ey)

    entropies = []
    n1,n2 = ey.shape
    for i in range(w):
        if i+n2<X.shape[1]:
            this_matrix = X[:,i:i+n2]
            entropies.append(shannon_entropy(this_matrix*ey))
    return shannon_entropy(entropies)

#   all_shannon_entropy = []
#   all_bin_thresh = np.arange(X_conv.min(), X_conv.max(), 0.01)
#   best_bin_thresh = None
#   best_se = None

#   print('optimizing bin_thresh')
#   for bin_thresh in tqdm.tqdm(list(all_bin_thresh)):
#       X_bin = binarize(X_conv, bin_thresh, filename=None)
#       #X_bin = binarize(X_conv, 0.05, filename=bin_filename)

#       X_fill = edges_to_contours(X_bin, etc_kernel_size)

#       X_binop = apply_bin_op(X_fill, binop_dim)

#       se = entropy(X_binop)

#       if not best_se:
#           best_se = se
#           best_bin_thresh = bin_thresh
#       elif se < best_se and se != 0:
#           best_se = se
#           best_bin_thresh = bin_thresh

#       all_shannon_entropy.append(se)


#   plt.figure().clear()
#   plt.close()
#   plt.cla()
#   plt.clf()
#   plt.plot(all_bin_thresh, all_shannon_entropy)
#   plt.savefig('cross_sim_test/se_plot.png')
#   skimage.io.imsave('cross_sim_test/crossX.png', crossX)

# koti
s1 = round(324.714*sr/cqt_window)
l1 = round(2.408*sr/cqt_window)
feat1 = ampsA
feat2 = ampsB

from numpy.fft import fft, ifft

def autocorr(data):
    dataFT = fft(data, axis=1)
    dataAC = ifft(dataFT * np.conjugate(dataFT), axis=1).real
    return dataAC

corr = autocorr(X_bin)
skimage.io.imsave('cross_sim_test/crossX.png', crossX[:,1000:-1000])
skimage.io.imsave('cross_sim_test/corrX.png', corr[:,1000:-1000])

def query_pattern(s1, l1, feat1, feat2, thresh_prop, etc_kernel_size, binop_dim, min_diff_trav_seq, min_length_cqt, cqt_window, sr, timestepA):
    crossX = get_pattern_crosssim(s1, l1, feat1, feat2)

    print('Convolving similarity matrix')
    X_conv = convolve_array(crossX, sobel)

    maxconv = X_conv.max()
    bin_thresh = maxconv*thresh_prop
    print('Binarizing convolved array')
    X_bin = binarize(X_conv, bin_thresh, filename=None)
    skimage.io.imsave('cross_sim_test/crossX.png', X_bin)
    #X_bin = binarize(X_conv, 0.05, filename=bin_filename)

    entropies = get_entropies(X_bin)

    print('Identifying and isolating regions between edges')
    X_fill = edges_to_contours(X_bin, etc_kernel_size)

    print('Cleaning isolated non-directional regions using morphological opening')
    X_binop = apply_bin_op(X_fill, binop_dim)

    print('Extracting Segments')
    all_segments = extract_segments_new(X_binop)

    print('Joining segments that are sufficiently close')
    all_segments_joined = join_all_segments(all_segments, min_diff_trav_seq)

    all_segments_reduced = remove_short(all_segments_joined, min_length_cqt)

    if all_segments_reduced:
        all_groups = [[(s1+x0,s1+x1),(y0,y1)] for ((x0,y0),(x1,y1)) in all_segments_reduced]

        print('Convert sequences to pitch track timesteps')
        starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups, cqt_window, sr, timestepA)

        matches_starts  = [x[1] for x in starts_seq]
        matches_lengths = [x[1] for x in lengths_seq]

        matches_starts  = [round(s1*cqt_window/(sr*timestepA))] + matches_starts
        matches_lengths = [round(l1*cqt_window/(sr*timestepA))] + matches_lengths

        return matches_starts, matches_lengths

        #starts_sec = [[x*timestepA for x in p] for p in starts_seq]
        #lengths_sec = [[x*timestepA for x in l] for l in lengths_seq]

    else:
        return None, None


import pandas as pd
matches_dict = {}
KJ = load_annotations_new('annotations/koti_janmani.txt', min_m=2)
for i,(tier, s1, s2, text, text_full) in KJ.iterrows():
    title_str = f'\nFinding matches for {text_full}'
    print(title_str)
    print('-'*len(title_str))
    s1_cqt = round(s1*sr/cqt_window)
    s2_cqt = round(s2*sr/cqt_window)
    l = s2_cqt-s1_cqt
    matches_starts, matches_lengths = query_pattern(s1_cqt, l, feat1, feat2, 0.6, etc_kernel_size, binop_dim, min_diff_trav_seq, min_length_cqt, cqt_window, sr, timestepA)
    if not matches_starts:
        print('\nNo matches found...')
        continue
    if text not in matches_dict:
        matches_dict[text] = [(matches_starts, matches_lengths)]
    else:
        matches_dict[text] = matches_dict[text] + [(matches_starts, matches_lengths)]
    N = len(matches_starts)-1
    print(f'\n{N} matches found...')




## PLOTTING
###########
from exploration.visualisation import *

svara_cent_path = "conf/svara_cents.yaml"
svara_freq_path = "conf/svara_lookup.yaml"

svara_cent = load_yaml(svara_cent_path)
svara_freq = load_yaml(svara_freq_path)

raga = 'ritigaula'
tonic = 199

arohana = svara_freq[raga]['arohana']
avorahana = svara_freq[raga]['avorahana']
all_svaras = list(set(arohana+avorahana))
print(f'Svaras for raga, {raga}:')
print(f'   arohana: {arohana}')
print(f'   avorahana: {avorahana}')

yticks_dict = {k:cents_to_pitch(v, tonic) for k,v in svara_cent.items()}
yticks_dict = {k:v for k,v in yticks_dict.items() if any([x in k for x in all_svaras])}

plot_kwargs = {
    'yticks_dict':yticks_dict,
    'cents':True,
    'tonic':tonic,
    'emphasize':['S', 'S ', 'S  ', ' S', '  S'],
    'figsize':(15,4)
}



for text, matches in matches_dict.items():
    print(f'Writing matches for pattern, {text}')
    for i, (starts, lengths) in enumerate(matches):
        output_dir = f'cross_sim_test/KJ_v_VJ/{i}_{text}/'
        starts = [starts]
        lengths = [lengths]
        try:
            shutil.rmtree(output_dir)
        except:
            pass
        for i, seqs in enumerate(starts):
            for si, s in enumerate(seqs):
                l = lengths[i][si]
                l_sec = round(lengths_seq[i][0]*timestepA,1)
                sp = int(s)
                if si == 0:
                    this_pitch = pitchA
                    this_time = timeA
                    this_timestep = timeA[1] - timeA[0]
                    st = 'query'
                else:
                    this_pitch = pitchB
                    this_time = timeB
                    this_timestep = timeB[1] - timeB[0]
                    st = 'match'
                t_sec = s*this_timestep
                str_pos = get_timestamp(t_sec)
                plot_path = os.path.join(output_dir, f'match_{i}_len={l_sec}/{st}_time={str_pos}.png')
                create_if_not_exists(plot_path)

                plot_subsequence(
                    sp, l, this_pitch, this_time, this_timestep, path=plot_path, plot_kwargs=plot_kwargs
                )

## audio
#write_all_sequence_audio(audio_path, starts[:top_n], lengths[:top_n], timestep, results_dir)





