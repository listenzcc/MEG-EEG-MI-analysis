"""
File: 5.read.fsaverage.py
Author: Chuncheng Zhang
Date: 2025-06-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read fsaverage files for fwd.

    By looking at Display sensitivity maps for EEG and MEG sensors plot the sensitivity maps for EEG and compare it with the MEG, can you justify the claims that:
    <https://mne.tools/stable/auto_examples/forward/forward_sensitivity_maps.html#ex-sensitivity-maps>
    <https://mne.tools/stable/generated/mne.sensitivity_map.html#mne.sensitivity_map>

    - MEG is not sensitive to radial sources
    - EEG is more sensitive to deep sources

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-27 ------------------------
# Requirements and constants
import joblib
import sys
import io
from contextlib import redirect_stdout

from util.easy_import import *
from util.subject_fsaverage import SubjectFsaverage
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
# from util.read_example_raw import md

subject_directory = Path('./rawdata/S01_20220119')

parse = argparse.ArgumentParser('Compute Inverse (in Fsaverage space)')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

subject_name = subject_directory.name

data_directory = Path(f'./data/fsaverage/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# md.generate_epochs(**dict(tmin=-2, tmax=5, decim=6))
# raw = md.raw
# eeg_epochs = md.eeg_epochs
# meg_epochs = md.meg_epochs
# print(raw)
# print(eeg_epochs)
# print(meg_epochs)


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

        # ! Necessary to inverse operation
        md.eeg_epochs.set_eeg_reference(projection=True)
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


evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs = concat_epochs(mds)

# %% ---- 2025-06-27 ------------------------
# Function and class

# Prepare subject
subject = SubjectFsaverage()
subject.pipeline()
fwd_eeg = subject.read_forward_solution(eeg_epochs.info, 'eeg')
fwd_meg = subject.read_forward_solution(meg_epochs.info, 'meg')
print('src', subject.src)
print('bem', subject.bem)

# Compute cov
print('Computing noise_cov')
with redirect_stdout(io.StringIO()):
    method = ['empirical']
    noise_cov = dict(
        eeg=mne.compute_covariance(eeg_epochs, tmax=0, method=method),
        meg=mne.compute_covariance(meg_epochs, tmax=0, method=method),
    )
print(noise_cov)

# Compute inverse operator
print('Computing inverse_operator')
with redirect_stdout(io.StringIO()):
    inverse_operator = dict(
        eeg=subject.make_inverse_operator(
            eeg_epochs.info, fwd_eeg, noise_cov['eeg']),
        meg=subject.make_inverse_operator(
            meg_epochs.info, fwd_meg, noise_cov['meg']),
    )
print(inverse_operator)

stuff_estimate_snr = dict(
    cov_eeg=noise_cov['eeg'],
    cov_meg=noise_cov['meg'],
    fwd_eeg=fwd_eeg,
    fwd_meg=fwd_meg,
    info_eeg=eeg_epochs.info,
    info_meg=meg_epochs.info
)

joblib.dump(stuff_estimate_snr, data_directory.joinpath(
    'stuff-estimate-snr.dump'))

# Compute inverse
snr = 3.0  # Standard assumption for average data but using it for single trial
kwargs = dict(
    lambda2=1.0 / snr**2,
    method="dSPM"  # use dSPM method (could also be MNE or sLORETA)
)

# Compute EEG inverse
print('Computing EEG inverse')
with redirect_stdout(io.StringIO()):
    eeg_epochs.load_data()
    for evt in evts:
        evoked = eeg_epochs[evt].average()
        eeg_stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator['eeg'], **kwargs)
        print(eeg_stc)
        eeg_stc.save(data_directory.joinpath(
            f'eeg-evt{evt}.stc'), overwrite=True)

# Compute MEG inverse
print('Computing MEG inverse')
with redirect_stdout(io.StringIO()):
    for evt in evts:
        evoked = meg_epochs[evt].average()
        meg_stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator['meg'], **kwargs)
        print(meg_stc)
        meg_stc.save(data_directory.joinpath(
            f'meg-evt{evt}.stc'), overwrite=True)

sys.exit(0)

# %% ---- 2025-06-27 ------------------------
# Pending


# %% ---- 2025-06-27 ------------------------
# Pending
