"""
File: 0.read.evoked.py
Author: Chuncheng Zhang
Date: 2025-07-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-03 ------------------------
# Requirements and constants
from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

subject_directory = Path('./rawdata/S01_20220119')

parse = argparse.ArgumentParser('Compute TFR')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

subject_name = subject_directory.name

data_directory = Path(f'./data/evoked/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-07-03 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    epochs_kwargs = {'tmin': -3, 'tmax': 5, 'decim': 6}
    use_latest_ds_directories = 8  # 8

    # Read from file
    mds = []
    found = find_ds_directories(subject_directory)
    mds.extend([read_ds_directory(p)
                for p in found[-use_latest_ds_directories:]])

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
        md.eeg_epochs.apply_baseline((-2, 0))
        md.meg_epochs.apply_baseline((-2, 0))

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    eeg_epochs = mne.concatenate_epochs(
        [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    meg_epochs = mne.concatenate_epochs(
        [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])
    return eeg_epochs, meg_epochs


# %% ---- 2025-07-03 ------------------------
# Play ground
# Read data
evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs = concat_epochs(mds)

# %%
for evt in evts:
    evoked = eeg_epochs[evt].average()
    evoked.save(data_directory.joinpath(
        f'eeg-evt{evt}-n{evoked.nave}-ave.fif'), overwrite=True)
    evoked = meg_epochs[evt].average()
    evoked.save(data_directory.joinpath(
        f'meg-evt{evt}-n{evoked.nave}-ave.fif'), overwrite=True)

# %%
# for evt in evts:
#     evoked = eeg_epochs[evt].average()
#     evoked.pick(['C3', 'Cz', 'C4'])
#     evoked.filter(l_freq=0.1, h_freq=40.0, n_jobs=32)
#     evoked.crop(tmin=-0.5, tmax=0.5)
#     evoked.apply_baseline((None, 0))
#     evoked.plot_joint(title=f'EEG MRCP @evt:{evt}', times=[0, 0.1, 0.2])

#     evoked = meg_epochs[evt].average()
#     evoked.pick(['MLC42', 'MZC03', 'MRC42'])
#     evoked.filter(l_freq=0.1, h_freq=40.0, n_jobs=32)
#     evoked.crop(tmin=-0.5, tmax=0.5)
#     evoked.apply_baseline((None, 0))
#     evoked.plot_joint(title=f'MEG MRCP @evt:{evt}', times=[0, 0.1, 0.2])

# %% ---- 2025-07-03 ------------------------
# Pending


# %% ---- 2025-07-03 ------------------------
# Pending
