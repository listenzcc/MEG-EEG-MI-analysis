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
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory


# %% ---- 2025-05-14 ------------------------
# Function and class


# %% ---- 2025-05-14 ------------------------
# Play ground
if __name__ == '__main__':
    found = find_ds_directories('./rawdata')
    print(found)
    md = read_ds_directory(found[0])
    md.convert_raw_to_epochs(tmin=-1, tmax=5)
    print(md)
    print(md.eeg_epochs)
    print(md.meg_epochs)
    fig = md.eeg_epochs.plot_sensors()
    fig = md.meg_epochs.plot_sensors()

# %%

# %%

# %% ---- 2025-05-14 ------------------------
# Pending

# %% ---- 2025-05-14 ------------------------
# Pending

# %%

# %%
