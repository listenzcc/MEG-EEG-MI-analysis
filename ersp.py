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

from rich import print
from pathlib import Path
from tqdm.auto import tqdm
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
from util.analysis.ERSP_index import ERSP_analysis_with_pdf


# %% ---- 2025-05-14 ------------------------
# Function and class

# %%
found = find_ds_directories('./rawdata/S01_20220119')
print(found)

# %%
# Read raw data and convert raw to epochs
mds = [read_ds_directory(f) for f in found[-8:]]
dev_head_t = mds[0].raw.info['dev_head_t']
for md in mds:
    md.raw.info['dev_head_t'] = dev_head_t
    md.add_proj()
    md.convert_raw_to_epochs(tmin=-1, tmax=5, decim=6)
print(mds)

# %%
# Concatenate epochs
selected_channels = ['C3', 'Cz', 'C4']
for md in mds:
    md.eeg_epochs.pick_channels(selected_channels)

eeg_epochs = mne.concatenate_epochs(
    [md.eeg_epochs for md in tqdm(mds, 'Concatenate EEG epochs')])
print(eeg_epochs)

ERSP_analysis_with_pdf(
    epochs=eeg_epochs,
    event_ids=md.event_id,
    selected_channels=selected_channels,
    pdf_path=Path('./ERSP_eeg.pdf'))

# %%
selected_channels = ['MLP34-4504', 'MZC01-4504', 'MRP34-4504']
for md in mds:
    md.meg_epochs.pick_channels(selected_channels)

meg_epochs = mne.concatenate_epochs(
    [md.meg_epochs for md in tqdm(mds, 'Concatenate MEG epochs')])
print(meg_epochs)

ERSP_analysis_with_pdf(
    epochs=meg_epochs,
    event_ids=md.event_id,
    selected_channels=selected_channels,
    pdf_path=Path('./ERSP_meg.pdf'))

# %%

# %% ---- 2025-05-14 ------------------------
# Play ground


# %% ---- 2025-05-14 ------------------------
# Pending

# %% ---- 2025-05-14 ------------------------
# Pending
