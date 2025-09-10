"""
File: check_epochs.py
Author: Chuncheng Zhang
Date: 2025-09-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the epoch for details.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-10 ------------------------
# Requirements and constants
from util.easy_import import *
from util.read_example_raw import md
from collections import defaultdict

data_directory = Path('./data')

# %% ---- 2025-09-10 ------------------------
# Function and class


# %% ---- 2025-09-10 ------------------------
# Play ground
md.generate_epochs()
print(md.meg_epochs)
print(md.eeg_epochs)


# %% ---- 2025-09-10 ------------------------
# Pending
md.meg_epochs.load_data()
md.meg_epochs.pick('mag')
md.meg_epochs.ch_names

ch_names = [e for e in md.meg_epochs.ch_names if len(e) == 5]

area_marks_1 = set([e[2:3] for e in ch_names])
area_marks_2 = set([e[1:3] for e in ch_names if not e[1] == 'Z'])
print(f'{area_marks_1=}, {area_marks_2=}')

ch_name_dct = defaultdict(list)
for m in area_marks_1:
    for name in ch_names:
        if name[2] == m:
            ch_name_dct[m].append(name)
for m in area_marks_2:
    for name in ch_names:
        if name[1:3] == m or name[1:3] == 'Z'+m[1]:
            ch_name_dct[m].append(name)
print(ch_name_dct)

json.dump(ch_name_dct, open(
    data_directory.joinpath('meg_ch_name_dct.json'), 'w'))

# %% ---- 2025-09-10 ------------------------
# Pending

# %%
