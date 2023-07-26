%load_ext autoreload
%autoreload 2

import skimage.io

from convert import *
from exploration.pitch import *
from exploration.io import *
from all_paths import all_paths

run_keyword= 'hpc'

cache_dir = "cache"
cuda = False
train_it = True
continue_train = False
start_epoch = 0
test_data = 'jku_input.txt'
test_cqt = False

data_type = 'cqt'

# train params
samples_epoch = 100000
batch_size = 1000
epochs = 1000
lr = 1e-3
sparsity_reg = 0e-5
weight_reg = 0e-5
norm_loss = 0
equal_norm_loss = 0
learn_norm = False
set_to_norm = -1
power_loss = 1
seed = 1
plot_interval = 500

# model params
dropout = 0.5
n_bases = 256

# CQT params
length_ngram = 32
fmin = 65.4
hop_length = 1984
block_size = 2556416
n_bins = 120
bins_per_oct = 24
sr = 44100

# MIDI params
min_pitch = 40
max_pitch = 100
beats_per_timestep = 0.25

# data loader
emph_onset = 0
rebuild = False
# shiftx (time), shifty (pitch)
shifts = 12, 24
# scalex, scaley (scaley not implemented!)
scales = 0, 0
# transform: 0 -> shifty (pitch), 1 -> shiftx (time), 2 -> scalex (time)
transform = 0, 1


torch.manual_seed(seed)
np.random.seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
out_dir = os.path.join("output", run_keyword)

assert os.path.exists(out_dir), f"The output directory {out_dir} does " \
    f"not exist. Did you forget to train using the run_keyword " \
    f"{run_keyword}?"

if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

if data_type == 'cqt':
    in_size = n_bins * length_ngram
else:
    raise AttributeError(f"Data_type {data_type} not supported. "
                         f"Possible type is 'cqt'.")
model = Complex(in_size, n_bases, dropout=dropout)
model_save_fn = os.path.join(out_dir, "model_complex_auto_"
                                     f"{data_type}.save")
model.load_state_dict(torch.load(model_save_fn, map_location='cpu'), strict=False)


def find_nearest(array, value, ix=True):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if ix:
        return idx
    else:
        return array[idx]


# params from config
length_ngram=32
cuda=False
data_type = 'cqt'
n_bins=120
bins_per_oct=24
fmin=80
hop_length=1984
step_size=1
mode='cosine'

all_paths = [
    (('2018_11_13_am_Sec_3_P1_Anandabhairavi', 'anandabhairavi', 187.0),
    (
        './audio/lara_small/spleeter/2018_11_13_am_Sec_3_P1_Anandabhairavi.mp3',
        './data/stability_tracks/lara_small/2018_11_13_am_Sec_3_P1_Anandabhairavi.csv' ,
        './data/pitch_tracks/lara_small/2018_11_13_am_Sec_3_P1_Anandabhairavi.csv'
    )),
    (('2018_11_18_am_Sec_6_P6_Bilahari_B', 'bilahari', 136.0),
    (
        './audio/lara_small/spleeter/2018_11_18_am_Sec_6_P6_Bilahari_B.mp3',
        './data/stability_tracks/lara_small/2018_11_18_am_Sec_6_P6_Bilahari_B.csv' ,
        './data/pitch_tracks/lara_small/2018_11_18_am_Sec_6_P6_Bilahari_B.csv'
    ))
]

for i_path in range(len(all_paths)):
    try:
        (title, raga, tonic), (file, mask_file, pitch_file) = all_paths[i_path]
        track_name = pitch_file.replace('.csv','').split('/')[-1]
        out_dir = f'/Volumes/Shruti/asplab2/cae-invar/data/self_similarity/{track_name}/'
        if False:#os.path.isfile(os.path.join(out_dir,'self_sim.npy')):
            print(f'Skipping {track_name}')
        else:
            print('\n---------')
            print(title)
            print('---------')

            #def create_matrices(file, mask_file, length_ngram, cuda, data_type, n_bins, bins_per_oct, fmin, hop_length, step_size=1, mode='cosine'):
            print('Computing CAE features')
            data = get_input_repr(file, data_type, n_bins, bins_per_oct, fmin, hop_length)
            mask, time, timestep  = get_timeseries(mask_file)
            pitch, fr_time, fr_timestep  = get_timeseries(pitch_file)

            ampls, phases = to_amp_phase(model, data, step_size=step_size, length_ngram=length_ngram, cuda=cuda)
            results = np.array([ampls, phases])

            print('Computing self similarity matrix')

            matrix = create_ss_matrix(ampls, mode='cosine')
            matrix = np.pad(matrix, ((0, 9), (0, 9)), mode='constant',
                            constant_values=matrix.max())
            matrix = 1 / (matrix + 1e-6)

            for k in range(-8, 9):
                eye = 1 - np.eye(*matrix.shape, k=k)
                matrix = matrix * eye

            flength = 10
            ey = np.eye(flength) + np.eye(flength, k=1) + np.eye(flength, k=-1)
            matrix = convolve2d(matrix, ey, mode="same")
            matrix -= matrix.min()
            matrix /= (matrix.max() + 1e-8)
            #plt.imsave('random/4final.png', matrix, cmap="hot")

            ## Output
            metadata = {
                'orig_size': (len(data), len(data)),
                'sparse_size': (matrix.shape[0], matrix.shape[0]),
                'audio_path': file,
                'pitch_path': pitch_file,
                'raga': raga,
                'tonic': tonic,
                'title': title
            }

            out_path_mat = os.path.join(out_dir, 'self_sim.npy')
            out_path_meta = os.path.join(out_dir, 'metadata.pkl')
            out_path_feat = os.path.join(out_dir, "features.pyc.bz")

            create_if_not_exists(out_dir)

            print(f"Saving features to {out_path_feat}..")
            save_pyc_bz(results, out_path_feat)

            print(f"Saving self sim matrix to {out_path_mat}..")
            np.save(out_path_mat, matrix)

            print(f'Saving metadata to {out_path_meta}')
            write_pkl(metadata, out_path_meta)
    except Exception as e:
        print(f'{title} failed')
        print(f'{e}')


