"""
File: mvpa.1.1.rawEpochs.megAreas.summary.py
Author: Chuncheng Zhang
Date: 2025-09-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the megAreas MVPA results.

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
from util.io.file import load

plt.style.use('ggplot')

meg_ch_name_dct = json.load(open('./data/meg_ch_name_dct.json'))

data_directory = Path('./data/MVPA.megAreas')


# %% ---- 2025-09-10 ------------------------
# Function and class


# %% ---- 2025-09-10 ------------------------
# Play ground
print(meg_ch_name_dct.keys())

#
mode = 'meg'
evts = [1, 2, 3, 4, 5]

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ch_mark in sorted(list(meg_ch_name_dct.keys())):
    files = list(data_directory.rglob(f'decoding-meg-chmark-{ch_mark}.dump'))
    objs = [load(file) for file in files]
    times = objs[0]['times']
    scores_mean = np.mean([obj['scores'] for obj in objs], axis=0)
    scores_std = np.std([obj['scores'] for obj in objs], axis=0)

    if len(ch_mark) == 1:
        ax = axes[0]
    elif ch_mark[0] == 'L':
        ax = axes[1]
    else:
        ax = axes[2]

    ax.plot(times, np.diag(scores_mean), label=f"score({ch_mark})")

for ax in axes:
    ax.axhline(1/len(evts), color="gray", linestyle="--", label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AccScore")
    ax.legend(loc='lower right')  # , bbox_to_anchor=(1, 1))
    ax.axvline(0.0, color="gray", linestyle="-")
    ax.set_title(f"Decoding over time (mode: {mode})")
    ax.set_ylim([0.15, 0.35])

fig.tight_layout()
fig.savefig(data_directory.joinpath('mvpa_megAreas_summary.png'))

plt.show()

# %%

# %% ---- 2025-09-10 ------------------------
# Pending


# %% ---- 2025-09-10 ------------------------
# Pending
