"""
File: mvpa.6.py
Author: Chuncheng Zhang
Date: 2025-08-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    MVPA for MEG & EEG epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-11 ------------------------
# Requirements and constants
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from mne.decoding import CSP, LinearModel

from util.bands import Bands
from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
from util.io.file import save

# --------------------------------------------------------------------------------
mode = 'all'
band_name = 'all'

# --------------------------------------------------------------------------------
subject_directory = Path('./rawdata/S01_20220119')

subject_name = subject_directory.name
data_directory = Path(f'./data/MVPA.6/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-08-11 ------------------------
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


# %% ---- 2025-08-11 ------------------------
# Play ground
evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs, groups = concat_epochs(mds)

print(eeg_epochs)
print(meg_epochs)

# %% ---- 2025-08-11 ------------------------
# Pending


def decoding(epochs):
    epochs = epochs.copy()
    X = epochs.get_data()
    y = epochs.events[:, -1]
    cv = LeaveOneGroupOut()
    clf = make_pipeline(
        CSP(),
        LinearModel(LogisticRegression(solver='liblinear'))
    )
    y_pred = cross_val_predict(clf, X, y, groups=groups, cv=cv)

    print(subject_name)
    print(metrics.classification_report(y_true=y, y_pred=y_pred))

    return {
        'y_true': y,
        'y_pred': y_pred,
        'subject': subject_name
    }


eeg_res = decoding(eeg_epochs)
meg_res = decoding(meg_epochs)

save(eeg_res, data_directory.joinpath('res-eeg.dump'))
save(meg_res, data_directory.joinpath('res-meg.dump'))

# %% ---- 2025-08-11 ------------------------
# Pending
