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
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut, ShuffleSplit, cross_val_predict
import joblib
import matplotlib

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# --------------------------------------------------------------------------------
mode = 'meg'  # 'meg', 'eeg'
band_name = 'all'  # 'delta', 'theta', 'alpha', 'beta', 'gamma', 'all'
subject_directory = Path('./rawdata/S01_20220119')

# subject_directory = Path("./rawdata/S07_20231220")

# Use the arguments
parse = argparse.ArgumentParser('Compute freq CSP')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

# --------------------------------------------------------------------------------
# Prepare the paths
subject_name = subject_directory.name
data_directory = Path(f'./data/MVPA.CSP/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# pdf_path = data_directory / f'decoding-{mode}-band-{band_name}.pdf'
# dump_path = Path(pdf_path).with_suffix('.dump')

# %%
min_freq = 3.0
max_freq = 25.0
n_freqs = 12  # how many frequency bins to use
n_cycles = 10.0  # how many complete cycles: used to define window size

# Assemble list of frequency range tuples
freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples
freqs, freq_ranges

# %%

# %% ---- 2025-07-21 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    bands = Bands()
    l_freq, h_freq = bands.get_band(band_name)
    epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6}
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
            md.eeg_epochs.filter(**filter_kwargs)
            md.eeg_epochs.crop(tmin=-1, tmax=4)
            md.eeg_epochs.apply_baseline((-1, 0))

        if mode in ['meg', 'all']:
            md.meg_epochs.load_data()
            md.meg_epochs.filter(**filter_kwargs)
            md.meg_epochs.crop(tmin=-1, tmax=4)
            md.meg_epochs.apply_baseline((-1, 0))

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


# epochs = epochs.resample(100, npad="auto")  # resample to 100 Hz

# cv = np.max(groups)+1
# MEG signals: n_epochs, n_meg_channels, n_times
X = epochs.get_data(copy=False)
y = epochs.events[:, 2]  # target

print(f'{X.shape=}, {y.shape=}, {np.array(groups).shape=}')

# %%
# scoring = make_scorer(accuracy_score, greater_is_better=True)

# clf = make_pipeline(
#     CSP(n_components=4),
#     LinearDiscriminantAnalysis(),
# )

# %%
# cv = LeaveOneGroupOut()
# res = cross_val_score(estimator=clf, X=X, y=y,
#                       groups=groups, cv=cv, scoring=scoring)
# print(res)

# %%
# cv = LeaveOneGroupOut()
# y_pred = cross_val_predict(estimator=clf, X=X, y=y, groups=groups, cv=cv)
# print(y_pred)
# print(classification_report(y_true=y, y_pred=y_pred))

# %%

# %%
# init scores
freq_CSP_results = {
    'subject_name': subject_name,
    'y_true': y,
    'mode': mode,
}

# Loop through each frequency range of interest
for freq, (fmin, fmax) in enumerate(freq_ranges):
    # Infer window size based on the frequency being used
    w_size = n_cycles / ((fmax + fmin) / 2.0)  # in seconds

    filter_kwargs = {'l_freq': fmin, 'h_freq': fmax, 'n_jobs': n_jobs}
    epochs_filter = epochs.copy()
    epochs_filter.filter(**filter_kwargs)

    X = epochs_filter.get_data(copy=False)

    cv = LeaveOneGroupOut()
    clf = make_pipeline(
        CSP(n_components=8),
        # StandardScaler(),  # Against outlier
        # SelectKBest(score_func=mutual_info_classif, k=50),  # MI特征选择，k为保留特征数
        LinearDiscriminantAnalysis(),
        # LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    )
    y_pred = cross_val_predict(
        estimator=clf, X=X, y=y, groups=groups, cv=cv)

    print(classification_report(y_true=y, y_pred=y_pred))
    freq_CSP_results[freq] = {
        'fmin': fmin,
        'fmax': fmax,
        'y_pred': y_pred}

print(freq_CSP_results)
joblib.dump(freq_CSP_results, data_directory.joinpath('freq-CSP-results.dump'))


# %% ---- 2025-07-21 ------------------------
# Pending

# %%
