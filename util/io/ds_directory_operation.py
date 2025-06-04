"""
File: ds_directory_operation.py
Author: Chuncheng Zhang
Date: 2025-05-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Find the task directories.
    Work with the ds directory.

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
import mne.io.ctf
from typing import Union
from pathlib import Path
from ..data import MyData
from ..logging import logger

experiment_name_pattern = 'S*_G33IA_*.ds'
noise_name_pattern = 'Noise-default*.ds'

# %% ---- 2025-05-14 ------------------------
# Function and class


def find_ds_directories(root: Union[Path, str], pattern: str = experiment_name_pattern) -> list:
    root = Path(root)
    directories = list(root.rglob(pattern))
    directories = sorted(directories)
    logger.info(f'Found ds directories: {directories}')
    return directories


def read_ds_directory(directory: Path) -> MyData:
    directory = Path(directory)
    logger.info(f'Read ds: {directory}')
    raw = mne.io.read_raw_ctf(directory)

    noise_directory = find_ds_directories(
        directory.parent, noise_name_pattern)[0]
    noise_raw = mne.io.read_raw_ctf(noise_directory)

    events, event_id = mne.events_from_annotations(raw)
    md = MyData()
    md.setattr(**dict(raw=raw, noise_raw=noise_raw,
               events=events, event_id=event_id))
    return md


# %% ---- 2025-05-14 ------------------------
# Play ground


# %% ---- 2025-05-14 ------------------------
# Pending


# %% ---- 2025-05-14 ------------------------
# Pending
