"""
File: 1.compute.tfr.py
Author: Chuncheng Zhang
Date: 2025-06-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compute TFR objects for MEG & EEG

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-24 ------------------------
# Requirements and constants
from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

subject_directory = Path('./rawdata/S01_20220119')

parse = argparse.ArgumentParser('Compute TFR')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()

subject_directory = Path(args.subject_dir)
subject_name = subject_directory.name
data_directory = Path(f'./data/TFR/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-06-24 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    epochs_kwargs = {'tmin': -3, 'tmax': 6, 'decim': 6}
    use_latest_ds_directories = 2  # 8

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

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    eeg_epochs = mne.concatenate_epochs(
        [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    meg_epochs = mne.concatenate_epochs(
        [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])
    return eeg_epochs, meg_epochs


def compute_tfr(epochs: mne.Epochs):
    # Setup options
    freqs = np.arange(2, 41)  # frequencies from 2-40Hz
    tfr_options = dict(
        method='morlet',
        average=False,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        n_jobs=20,
        decim=2
    )
    crop_options = dict(tmin=-2, tmax=epochs.tmax)
    baseline_options = dict(mode='ratio')

    # Load data and compute
    epochs.load_data()
    tfr: mne.time_frequency.EpochsTFR = epochs.compute_tfr(**tfr_options)

    # Crop and apply baseline
    tfr.crop(**crop_options).apply_baseline((None, 0), **baseline_options)

    return tfr


def tfr_apply_baseline(tfr: mne.time_frequency.EpochsTFR, mode: str = 'ratio'):
    '''
    Copy the tfr.
    Apply baseline to the tfr with mode.

    mode‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
    Perform baseline correction by
    - subtracting the mean of baseline values (‘mean’)
    - dividing by the mean of baseline values (‘ratio’)
    - dividing by the mean of baseline values and taking the log (‘logratio’)
    - subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
    - subtracting the mean of baseline values and dividing by the standard deviation of baseline values (‘zscore’)
    - dividing by the mean of baseline values, taking the log, and dividing by the standard deviation of log baseline values (‘zlogratio’)
    '''
    tfr = tfr.copy()
    tfr.apply_baseline((None, 0), mode=mode)
    return tfr


# %% ---- 2025-06-24 ------------------------
# Play ground
evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs = concat_epochs(mds)

logger.info(f'Start with {subject_name}')
for me, epochs in zip(['meg', 'eeg'], [meg_epochs, eeg_epochs]):
    logger.info(f'Compute tfr: {subject_name}, {me}, {epochs}')
    tfr = compute_tfr(epochs)
    for evt, mode in itertools.product(evts, ['logratio', 'ratio', 'mean']):
        tfr_apply_baseline(tfr[evt], mode=mode).average().save(
            data_directory.joinpath(f'{me}-{mode}-{evt}-average-tfr.h5'), overwrite=True)
logger.info(f'Done with {subject_name}')

# %% ---- 2025-06-24 ------------------------
# Pending


# %% ---- 2025-06-24 ------------------------
# Pending
