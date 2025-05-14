"""
File: find_task_files.py
Author: Chuncheng Zhang
Date: 2025-05-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Find the task files(folders).

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-14 ------------------------
# Requirements and constants
from pathlib import Path
from ..logging import logger

ds_folder_pattern = 'S*_G33IA_*.ds'

# %% ---- 2025-05-14 ------------------------
# Function and class
def find_ds_folders(root:Path) -> list:
    root = Path(root)
    ds_folders = list(root.rglob(ds_folder_pattern))
    logger.info(f'Found ds folders: {len(ds_folders)}')
    return ds_folders



# %% ---- 2025-05-14 ------------------------
# Play ground



# %% ---- 2025-05-14 ------------------------
# Pending



# %% ---- 2025-05-14 ------------------------
# Pending
