from src.io import load_pkl

all_lengths = []
track_names = [
    'Koti Janmani',
    'Sharanu Janakana',
    'Vanajaksha Ninne Kori'
]

paths = [
    'output/for_paper/for_visualisation/Koti Janmani/lengths.pkl',
    'output/for_paper/for_visualisation/Sharanu Janakana/lengths.pkl',
    'output/for_paper/for_visualisation/Vanajaksha Ninne Kori/lengths.pkl'
]
for p in paths:
    all_lengths.append(load_pkl(p))

# per recording, overall and of annotations
#    - pattern lengths histogram
#    - group lengths histogram
#    - within group deviation

import matplotlib.pyplot as plt
import numpy as np

flat_tracks = [np.array([x*0.01 for y in v for x in y]) for v in all_lengths]
mean_std = [[(round(np.mean(y)*0.01,2), round(np.std(y)*0.01,2)) for y in v] for v in all_lengths]

bins = np.arange(2, 10.5, 0.1)
plt.figure(figsize=(10,4))
plt.hist(flat_tracks[0], bins=bins, alpha=1/3, label=track_names[0], density=True)
#plt.hist(flat_tracks[1], bins=bins, alpha=1/3, label=track_names[1], density=True)
#plt.hist(flat_tracks[2], bins=bins, alpha=1/3, label=track_names[2], density=True)
plt.legend(loc='upper right')
plt.title('Pattern Length')
plt.xlabel('Binned Pattern Length')
plt.ylabel('Normalised count')
plt.savefig('length_hist.png')
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()
