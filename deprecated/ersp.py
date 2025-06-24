"""
File: ersp.py
Author: Chuncheng Zhang
Date: 2025-05-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    ERSP analysis for the MEG and EEG data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-14 ------------------------
# Requirements and constants
import mne
import argparse

from rich import print
from pathlib import Path
from tqdm.auto import tqdm
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
from util.analysis.ERSP_index_computer import ERSP_analysis_with_saving


# %% ---- 2025-05-14 ------------------------
# Function and class
parse = argparse.ArgumentParser('ERSP Analysis')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
print(args.subject_dir)

# subject_directory = Path('./rawdata/S01_20220119')
subject_directory = Path(args.subject_dir)
subject_name = subject_directory.name

result_directories = dict(
    h5=Path('./data/erd/h5', subject_name),
    pdf=Path('./data/erd/pdf', subject_name),
)

assert subject_directory.is_dir(
), f'Subject directory not exists: {subject_directory}'

for k, v in result_directories.items():
    v.mkdir(parents=True, exist_ok=True)

# %%
found = find_ds_directories(subject_directory)
print(found)

# %%
# Read raw data and convert raw to epochs
# mds = [read_ds_directory(f) for f in found[-8:]]
mds = [read_ds_directory(f) for f in found[-8:]]
dev_head_t = mds[0].raw.info['dev_head_t']
event_id = []
for md in mds:
    md.raw.info['dev_head_t'] = dev_head_t
    md.add_proj()
    md.generate_epochs(tmin=-3, tmax=6, decim=6)
    event_id = md.event_id

print('*' * 80)
print(f'Working with {len(mds)} data.')
print(mds)

# %%
# Concatenate epochs
selected_channels = ['C3', 'Cz', 'C4']
for md in mds:
    md.eeg_epochs.load_data()
    md.eeg_epochs.pick_channels(selected_channels)

eeg_epochs = mne.concatenate_epochs(
    [md.eeg_epochs for md in tqdm(mds, 'Concatenate EEG epochs')])
print(eeg_epochs)

ERSP_analysis_with_saving(
    epochs=eeg_epochs,
    event_ids=event_id,
    selected_channels=selected_channels,
    pdf_path=result_directories['pdf'].joinpath('ERSP_eeg.pdf'),
    df_path=result_directories['h5'].joinpath('ERSP_eeg-df.h5'),
    tfr_path=result_directories['h5'].joinpath('ERSP_eeg-tfr.h5'),
)

# %%
selected_channels = ['MLC42', 'MZC03', 'MRC42']  # ['MLP34', 'MZC01', 'MRP34']
for md in mds:
    md.meg_epochs.load_data()
    md.meg_epochs.pick_channels(selected_channels)

meg_epochs = mne.concatenate_epochs(
    [md.meg_epochs for md in tqdm(mds, 'Concatenate MEG epochs')])
print(meg_epochs)

ERSP_analysis_with_saving(
    epochs=meg_epochs,
    event_ids=event_id,
    selected_channels=selected_channels,
    pdf_path=result_directories['pdf'].joinpath('ERSP_meg.pdf'),
    df_path=result_directories['h5'].joinpath('ERSP_meg-df.h5'),
    tfr_path=result_directories['h5'].joinpath('ERSP_meg-tfr.h5'),
)

print(f'Done with {subject_directory}')

# %%

# %% ---- 2025-05-14 ------------------------
# Play ground


# %% ---- 2025-05-14 ------------------------
# Pending

# %% ---- 2025-05-14 ------------------------
# Pending
