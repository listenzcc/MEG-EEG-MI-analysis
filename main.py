"""
File: main.py
Author: Chuncheng Zhang
Date: 2025-05-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Main enter.

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


# %% ---- 2025-05-14 ------------------------
# Function and class

# %% ---- 2025-05-14 ------------------------
# Play ground
if __name__ == '__main__':
    found = find_ds_directories('./rawdata')
    print(found)
    md = read_ds_directory(found[2])
    md.add_proj()
    md.convert_raw_to_epochs(tmin=-1, tmax=5)
    print(md)
    print(md.eeg_epochs)
    print(md.meg_epochs)
    fig = md.eeg_epochs.plot_sensors(show_names=True)
    fig = md.meg_epochs.plot_sensors(show_names=True)


# %% ---- 2025-05-14 ------------------------
# Pending
spectrum = md.raw.compute_psd()
fig = spectrum.plot()
for key in ['1', '2', '3', '4', '5']:
    evoked = md.eeg_epochs[key].average()
    fig = evoked.plot_joint()

# %%
evoked = md.meg_epochs['1'].average()
fig = evoked.plot_joint()

# %% ---- 2025-05-14 ------------------------
# Pending
empty_room_raw = md.noise_raw
empty_room_projs = mne.compute_proj_raw(empty_room_raw)
fig = mne.viz.plot_projs_topomap(
    empty_room_projs, colorbar=True, vlim="joint", info=empty_room_raw.info
)

# %%
md.raw


# %%
mne.compute_proj_raw(md.noise_raw.pick_types(eeg=True))
# %%
md.noise_raw

# %%
