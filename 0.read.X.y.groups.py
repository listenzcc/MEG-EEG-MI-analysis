"""
File: 0.read.X.y.groups.py
Author: Chuncheng Zhang
Date: 2025-09-28
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read X, y, groups, times ... for FBCNet.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-28 ------------------------
# Requirements and constants
from util.io.file import save
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

from util.easy_import import *

# %%
subject_directory = Path('./rawdata/S01_20220119')

# Use the arguments
parse = argparse.ArgumentParser(
    'Read X y group and others for FBCNet decoding.')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

# %%
DATA_DIR = Path(f'./data/{subject_directory.name}')
DATA_DIR.mkdir(exist_ok=True, parents=True)

MODE = 'all'  # 'meg' | 'eeg' | 'all'
USE_LATEST_DS_DIRS = 8  # 8


# %% ---- 2025-09-28 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    # Raw freq is 1200 Hz
    # epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6}
    epochs_kwargs = {'tmin': -1, 'tmax': 4, 'decim': 12}

    # Read from file
    found = find_ds_directories(subject_directory)
    mds = [read_ds_directory(p) for p in found[-USE_LATEST_DS_DIRS:]]

    # The concat requires the same dev_head_t
    dev_head_t = mds[0].raw.info['dev_head_t']

    # Read data and convert into epochs
    event_id = mds[0].event_id
    for md in tqdm(mds, 'Convert to epochs'):
        md.raw.info['dev_head_t'] = dev_head_t
        md.add_proj()
        md.generate_epochs(**epochs_kwargs)

        if MODE in ['eeg', 'all']:
            md.eeg_epochs.load_data()

        if MODE in ['meg', 'all']:
            md.meg_epochs.load_data()

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    groups = []
    for i, e in enumerate(mds):
        if MODE in ['eeg', 'all']:
            groups.extend([i for _ in range(len(e.eeg_epochs))])
        else:
            groups.extend([i for _ in range(len(e.meg_epochs))])

    if MODE in ['eeg', 'all']:
        eeg_epochs = mne.concatenate_epochs(
            [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    else:
        eeg_epochs = None

    if MODE in ['meg', 'all']:
        meg_epochs = mne.concatenate_epochs(
            [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])
    else:
        meg_epochs = None

    return eeg_epochs, meg_epochs, groups


# %% ---- 2025-09-28 ------------------------
# Play ground
evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs, groups = concat_epochs(mds)

for target_mode, epochs in zip(['eeg', 'meg'], [eeg_epochs, meg_epochs]):
    if epochs is None:
        continue

    X = epochs.get_data()
    y = epochs.events[:, 2]
    info = epochs.info

    saving = {
        'X': X,
        'y': y,
        'info': info,
        'groups': groups
    }

    save(saving, DATA_DIR.joinpath(
        f'{target_mode}-X-y-info-groups.dump'))

# %% ---- 2025-09-28 ------------------------
# Pending


# %% ---- 2025-09-28 ------------------------
# Pending
