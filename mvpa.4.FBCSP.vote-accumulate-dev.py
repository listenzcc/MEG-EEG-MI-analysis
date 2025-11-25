"""
File: mvpa.4.FBCSP.vote.py
Author: Chuncheng Zhang
Date: 2025-07-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    MPVA using FBCSP method, the proba of each bands are recorded to summary.
    So the summary can be achieved by joint probability.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-21 ------------------------
# Requirements and constants
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut, ShuffleSplit, cross_val_predict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

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
from util.easy_import import *
from util.io.file import save
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# --------------------------------------------------------------------------------
mode = 'eeg'  # 'meg', 'eeg'
band_name = 'all'  # 'delta', 'theta', 'alpha', 'beta', 'gamma', 'all'
subject_directory = Path('./rawdata/S01_20220119')

# subject_directory = Path("./rawdata/S07_20231220")

# Use the arguments
parse = argparse.ArgumentParser('Compute FBCSP decoding with vote')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

# --------------------------------------------------------------------------------
# Prepare the paths
subject_name = subject_directory.name
data_directory = Path(f'./data/MVPA.FBCSP.vote-accumulate-dev/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# %%
# min_freq = 3.0
# max_freq = 25.0
# n_freqs = 12  # how many frequency bins to use

# # Assemble list of frequency range tuples
# freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
# freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples
# freqs, freq_ranges

bands = Bands()
freq_ranges = [v for v in bands.bands.values()]

freq_ranges = [(e, e+4) for e in range(1, 45, 2)]

freq_ranges


# %% ---- 2025-07-21 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    # Raw freq is 1200 Hz
    # epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6}
    epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 12}

    l_freq, h_freq = bands.get_band(band_name)
    filter_kwargs = {'l_freq': l_freq, 'h_freq': h_freq, 'n_jobs': n_jobs}
    use_latest_ds_directories = 8  # 8

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

        if mode in ['eeg', 'all']:
            md.eeg_epochs.load_data()
            # md.eeg_epochs.filter(**filter_kwargs)
            # md.eeg_epochs.crop(tmin=-1, tmax=4)
            # md.eeg_epochs.apply_baseline((-1, 0))

        if mode in ['meg', 'all']:
            md.meg_epochs.load_data()
            # md.meg_epochs.filter(**filter_kwargs)
            # md.meg_epochs.crop(tmin=-1, tmax=4)
            # md.meg_epochs.apply_baseline((-1, 0))

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    groups = []
    for i, e in enumerate(mds):
        if mode in ['eeg', 'all']:
            groups.extend([i for _ in range(len(e.eeg_epochs))])
        else:
            groups.extend([i for _ in range(len(e.meg_epochs))])

    if mode in ['eeg', 'all']:
        eeg_epochs = mne.concatenate_epochs(
            [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    else:
        eeg_epochs = None

    if mode in ['meg', 'all']:
        meg_epochs = mne.concatenate_epochs(
            [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])
    else:
        meg_epochs = None

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


# MEG signals: n_epochs, n_meg_channels, n_times
# X = epochs.get_data(copy=False)
y = epochs.events[:, 2]  # target

# print(f'{X.shape=}, {y.shape=}, {np.array(groups).shape=}')

# %%
# init scores
freq_CSP_results = {
    'subject_name': subject_name,
    'y_true': y,
    'mode': mode,
    'freqs': freq_ranges,
    'results': []
}

# Loop through each frequency range of interest
for freqIdx, (fmin, fmax) in enumerate(freq_ranges):
    filter_kwargs = {'l_freq': fmin, 'h_freq': fmax, 'n_jobs': n_jobs}
    epochs_filter = epochs.copy()
    epochs_filter.filter(**filter_kwargs)
    epochs_filter.apply_baseline((-1, 0))

    # for tmax in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]):
    for tmax in tqdm([0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0]):
        _epochs = epochs_filter.copy()
        _epochs.crop(tmin=0, tmax=tmax)

        X = _epochs.get_data(copy=False)

        cv = LeaveOneGroupOut()

        clf = make_pipeline(
            Scaler(epochs_filter.info),
            CSP(),
            SelectKBest(mutual_info_classif, k=10),
            LogisticRegression(),
        )
        y_proba = cross_val_predict(
            estimator=clf, X=X, y=y, groups=groups, cv=cv, method='predict_proba')
        y_pred = np.argmax(y_proba, axis=1) + 1

        # y_pred = cross_val_predict(estimator=clf, X=X, y=y, groups=groups, cv=cv)

        print(classification_report(y_true=y, y_pred=y_pred))
        freq_CSP_results['results'].append({
            'fmin': fmin,
            'fmax': fmax,
            'tmax': tmax,
            'y_proba': y_proba,
            'y_pred': y_pred,
        })

# %%

print(freq_CSP_results)
save(freq_CSP_results, data_directory.joinpath('freq-CSP-results.dump'))


# %% ---- 2025-07-21 ------------------------
# Pending

# %%
