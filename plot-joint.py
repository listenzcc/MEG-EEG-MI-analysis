"""
File: plot-joint.py
Author: Chuncheng Zhang
Date: 2025-06-13
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot joint for the MEG & EEG epochs/evoked.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-13 ------------------------
# Requirements and constants
import mne
import numpy as np
import matplotlib.pyplot as plt

from rich import print
from matplotlib.colors import TwoSlopeNorm
from mne.stats import permutation_cluster_1samp_test as pcluster_test

from util.io.ds_directory_operation import find_ds_directories, read_ds_directory


# %% ---- 2025-06-13 ------------------------
# Function and class

class Param:
    subject_dir = './rawdata/S01_20220119'
    tmin = -3
    tmax = 7
    eeg_channels = ['C3', 'Cz', 'C4']
    meg_channels = ['MLC42', 'MZC03', 'MRC42']


# %% ---- 2025-06-13 ------------------------
# Play ground
found = find_ds_directories(Param.subject_dir)
print(found)
md = read_ds_directory(found[3])
md.add_proj()
md.convert_raw_to_epochs(tmin=Param.tmin, tmax=Param.tmax)
md.eeg_epochs.load_data()
md.meg_epochs.load_data()
print(md)
print(md.eeg_epochs)
print(md.meg_epochs)
print(md.noise_raw)

# %%
md.eeg_epochs.pick_channels(Param.eeg_channels)
fig = md.eeg_epochs['1'].average().plot_joint()

md.meg_epochs.pick_channels(Param.meg_channels)
fig = md.meg_epochs['1'].average().plot_joint()

# %%
epochs = md.eeg_epochs
freqs = np.arange(2, 36)  # frequencies from 2-35Hz
tmin = -2
tmax = 4
baseline = (None, 0)

cmap = "RdBu"
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

cluster_test_kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)

tfr = epochs.compute_tfr(
    # method="multitaper",
    method="morlet",
    average=False,
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    n_jobs=20,
    decim=2,
)
# tfr = tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
tfr = tfr.crop(tmin, tmax).apply_baseline(baseline, mode="logratio")
print('tfr', tfr)

num_channels = len(epochs.ch_names)
event_ids = epochs.event_id

# Plot the ERSP for every events.
for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    # The latest column is colorbar
    fig, axes = plt.subplots(
        1, num_channels+1, figsize=(num_channels*4, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
    )
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        title = epochs.ch_names[ch]
        # positive clusters
        # _, c1, p1, _ = pcluster_test(
        #     tfr_ev.data[:, ch], tail=1, **cluster_test_kwargs)
        # negative clusters
        # _, c2, p2, _ = pcluster_test(
        #     tfr_ev.data[:, ch], tail=-1, **cluster_test_kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        # c = np.stack(c1 + c2, axis=2)  # combined clusters
        # p = np.concatenate((p1, p2))  # combined p-values
        # mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot(
            [ch],
            cmap=cmap,
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            # mask=mask,
            # mask_style="mask",
        )

        ax.set_title(title, fontsize=10)
        ax.axvline(0, linewidth=1, color="black",
                   linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]
                 ).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    plt.show()

# %%

# %%

# %% ---- 2025-06-13 ------------------------
# Pending
# Plot sensors
# to_sphere: When False, the sensor array appears similar as to looking downwards straight above the subject’s head.
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# md.meg_epochs.plot_sensors(show_names=True, axes=ax, to_sphere=False)
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# md.eeg_epochs.plot_sensors(show_names=True, axes=ax, to_sphere=False)
# plt.show()


# %% ---- 2025-06-13 ------------------------
# Pending

# %%
