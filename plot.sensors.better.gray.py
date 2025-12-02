"""
File: plot.sensors.better.gray.py
Author: Chuncheng Zhang
Date: 2025-11-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot MEG & EEG sensors for better look.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-03 ------------------------
# Requirements and constants
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from util.easy_import import *

from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
from util.read_example_raw import md


# %% ---- 2025-11-03 ------------------------
# Function and class


# %% ---- 2025-11-03 ------------------------
# Play ground
md.generate_epochs()
print(md.eeg_epochs.info)
print(md.meg_epochs.info)

# %%
# plt.style.use('ggplot')
# plt.style.use('seaborn-v0_8')

kwargs = {
    'linewidth': 0,
    'to_sphere': False,
    # 'sphere': [0, 0, 0, 0.095],
    'show': False
}

# fig, axes = plt.subplots(1, 2, figsize=(12, 8))


fig = plt.figure(figsize=(12, 8))
# 调整宽度比例，让瘦高的图窄一些，矮胖的图宽一些
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])  # 第二张图更宽

ax1 = plt.subplot(gs[0])  # 瘦高图
ax2 = plt.subplot(gs[1])  # 矮胖图

axes = [ax1, ax2]


# EEG
kwargs.update({
    'ch_groups': [
        [i for i, e in enumerate(md.eeg_epochs.ch_names) if e in 'C3 C4 Cz'],
        [i for i, e in enumerate(md.eeg_epochs.ch_names)
         if e not in 'C3 C4 Cz'],
    ],
    'cmap': ListedColormap(['gray', 'gray']),
    'title': 'EEG sensors layout (top view)',
    'to_sphere': True,
    'linewidth': 0
})

ax = axes[1]
mne.viz.plot_sensors(md.eeg_epochs.info, axes=ax, **kwargs)
ax.set_title(kwargs['title'])

# MEG
unique_area_marks = sorted(
    set([e[2] for e in md.meg_epochs.ch_names if len(e) == 5]))
print(unique_area_marks)

ch_groups = [[] for _ in range(len(unique_area_marks)+1)]
for i, name in enumerate(md.meg_epochs.ch_names):
    if not len(name) == 5:
        continue
    if False and name in ['MLC42', 'MZC03', 'MRC42']:
        j = 0
    else:
        j = unique_area_marks.index(name[2]) + 1
    ch_groups[j].append(i)

colors = ['red', 'gray'] + list(colormaps['Set1'].colors[1:len(ch_groups)-1])
# colors = ['red', 'gray'] + list(colormaps['tab10'].colors[:len(ch_groups)-1])
cm = ListedColormap(colors)
kwargs.update({
    'ch_groups': ch_groups,
    'cmap': cm,
    'title': 'MEG sensors layout (top view)',
    'to_sphere': True,
    'linewidth': 0
})
ax = axes[0]
mne.viz.plot_sensors(md.meg_epochs.info, axes=ax, **kwargs)
ax.set_title(kwargs['title'])

# fig.tight_layout()

plt.show()

# %%

# %% ---- 2025-11-03 ------------------------
# Pending
list(colormaps)
cm_names = {k: v for k, v in colormaps.items() if v.N <
            100 and not k.endswith('_r')}

i = 0
for k, v in cm_names.items():
    i += 1
    print(i, k, v.N)
    display(v)


# %% ---- 2025-11-03 ------------------------
# Pending


# %%

# %%

# %%

# %%
