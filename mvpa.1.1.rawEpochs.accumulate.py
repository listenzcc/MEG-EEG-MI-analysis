"""
File: mvpa.1.raw.epoch.py
Author: Chuncheng Zhang
Date: 2025-07-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    MPVA with raw epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-21 ------------------------
# Requirements and constants
from FBCSP.FBCSP_class import filter_bank, FBCSP_info
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import matplotlib

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mne.decoding import (
    CSP,
    Scaler,
    Vectorizer,
    LinearModel,
    SlidingEstimator,
    GeneralizingEstimator,
    get_coef,
    cross_val_multiscore,
)

from util.bands import Bands
from util.io.file import save
from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# --------------------------------------------------------------------------------
mode = 'meg'  # 'meg', 'eeg'
band_name = 'all'  # 'delta', 'theta', 'alpha', 'beta', 'gamma', 'all'
subject_directory = Path('./rawdata/S01_20220119')

# Use the arguments
parse = argparse.ArgumentParser('Decode on epochs, with accumulating manner.')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

# --------------------------------------------------------------------------------
# Prepare the paths
subject_name = subject_directory.name
OUTPUT_DIR = Path(f'./data/MVPA-accumulate/{subject_name}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% ---- 2025-07-21 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    The epochs are cut with large scale: (-2, 5) seconds.
    The filter is applied to the scale.
    After the filter, the epochs are cropped with (-1, 4) seconds.
    The method is to prevent cropping effect.
    '''
    # Setup options
    bands = Bands()
    l_freq, h_freq = bands.get_band(band_name)
    epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6}
    filter_kwargs = {'l_freq': l_freq, 'h_freq': h_freq, 'n_jobs': n_jobs}
    use_latest_ds_directories = 8

    # Read from file
    found = find_ds_directories(subject_directory)
    mds = [read_ds_directory(p) for p in found[-use_latest_ds_directories:]]

    # The concat requires the same dev_head_t
    dev_head_t = mds[0].raw.info['dev_head_t']

    # Read data and convert into epochs
    event_id = mds[0].event_id
    for md in tqdm(mds, 'Convert to epochs'):
        md.raw.info['dev_head_t'] = dev_head_t
        md.add_proj()
        md.generate_epochs(**epochs_kwargs)

        md.eeg_epochs.load_data()
        md.meg_epochs.load_data()

        md.eeg_epochs.filter(**filter_kwargs)
        md.meg_epochs.filter(**filter_kwargs)

        md.eeg_epochs.crop(tmin=-1, tmax=4)
        md.meg_epochs.crop(tmin=-1, tmax=4)

        md.eeg_epochs.apply_baseline((-1, 0))
        md.meg_epochs.apply_baseline((-1, 0))

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    groups = []
    for i, e in enumerate(mds):
        groups.extend([i for _ in range(len(e.meg_epochs))])

    eeg_epochs = mne.concatenate_epochs(
        [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    meg_epochs = mne.concatenate_epochs(
        [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])

    return eeg_epochs, meg_epochs, groups


evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs, groups = concat_epochs(mds)

# %%
print(eeg_epochs)
print(meg_epochs)
# print(groups)

# %% ---- 2025-07-21 ------------------------
# Play ground

if mode == 'meg':
    epochs = meg_epochs.copy().pick_types(meg=True, ref_meg=False)
elif mode == 'eeg':
    epochs = eeg_epochs.copy()
else:
    raise ValueError(f'Unknown mode: {mode}')

epochs = epochs.resample(100, npad="auto")  # resample to 100 Hz

cv = np.max(groups)+1
# MEG signals: n_epochs, n_meg_channels, n_times
X = epochs.get_data(copy=False)
y = epochs.events[:, 2]  # target
print(f'{X.shape=}, {y.shape=}')

# %%
k_select = 10
n_components = 4
freq_bands = [[4+i*4, 8+i*4] for i in range(9)]+[[8, 32]]
filter_type = 'iir'
filt_order = 5

FB = filter_bank(freq_bands, epochs.info['sfreq'], filt_order, filter_type)
filtered_X = FB.filt(X)
# filtered_X shape is (n_bands, n_samples, n_channels, n_times)
print(f'{filtered_X.shape=}')

# %%
times = epochs.times
results = {
    'subject': subject_name,
    'y_true': y,
    'tmin': np.min(times),
    'decoded': []
}
for tmax in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    tmax_idx = len(times[times < tmax])
    y_pred = y*0
    all_index = np.array([e for e in range(len(y))])
    for test_group in tqdm(set(groups)):
        test_index = all_index[np.where([e == test_group for e in groups])[0]]
        train_index = all_index[np.where([e != test_group for e in groups])[0]]
        fbcsp = FBCSP_info(FB, n_components, (0, tmax_idx), k_select)
        fbcsp.fit(filtered_X[:, train_index], y[train_index])
        y_pred[test_index] = fbcsp.predict(filtered_X[:, test_index])

    print(y, y_pred)
    results['decoded'].append({
        'y_pred': y_pred,
        'tmax': tmax
    })

save(results, OUTPUT_DIR.joinpath(f'{mode}-results.dump'))

print(f'Finished: {subject_name=}')

# %% ---- 2025-07-21 ------------------------
# Pending

# %%
