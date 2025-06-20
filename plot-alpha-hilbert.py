"""
File: plot-alpha.py
Author: Chuncheng Zhang
Date: 2025-06-19
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the alpha band activity for the MEG & EEG epochs/evoked.

    Alpha band: (8-12 Hz)

    - Evoked waveform.
    - Topographic map.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-19 ------------------------
# Requirements and constants
import mne

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rich import print
from pathlib import Path
from tqdm.auto import tqdm
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# %% ---- 2025-06-19 ------------------------
# Function and class
subject_directory = Path('./rawdata/S01_20220119')
found = find_ds_directories(subject_directory)
print(found)


mds = [read_ds_directory(f) for f in found[-8:]]
dev_head_t = mds[0].raw.info['dev_head_t']
event_id = []
for md in mds:
    md.raw.info['dev_head_t'] = dev_head_t
    md.add_proj()
    md.convert_raw_to_epochs(tmin=-3, tmax=6, decim=6)

md = mds[0]
# %% ---- 2025-06-19 ------------------------
# Play ground
for md in tqdm(mds, 'Load MEG&EEG epochs'):
    md.eeg_epochs.load_data()
    md.eeg_epochs.filter(8, 12, fir_design='firwin', phase='zero-double')
    md.meg_epochs.load_data()
    md.meg_epochs.filter(8, 12, fir_design='firwin', phase='zero-double')
eeg_epochs = mne.concatenate_epochs([md.eeg_epochs for md in mds])
meg_epochs = mne.concatenate_epochs([md.meg_epochs for md in mds])
print(eeg_epochs)
print(meg_epochs)

# %%
# print(md.event_id)
# for _event in md.event_id:
#     evoked = eeg_epochs[_event].average()
#     title = f'EEG Evoked Topomap @event: {_event}'
#     evoked.plot_joint(title=title, show=True)

#     evoked = meg_epochs[_event].average()
#     title = f'MEG Evoked Topomap @event: {_event}'
#     evoked.plot_joint(title=title, show=True)

# %% ---- 2025-06-19 ------------------------
# Pending
crop_args = (-2, 5)  # seconds
evoked_plot_times = [0, 0.5, 1, 2]  # seconds
combine = 'gfp'  # mean | gfp

pdf_path = 'Plot-Alpha-Hilbert.pdf'

# Save to pdf
matplotlib.use('pdf')
with PdfPages(pdf_path) as pdf:
    for _event in tqdm(md.event_id, 'Plotting Alpha Band Hilbert Envelope'):
        # ----------------------------------------
        # ---- EEG plot ----
        epochs_array = eeg_epochs[_event].copy().crop(
            *crop_args).apply_hilbert(envelope=True)
        print(epochs_array)
        figs = epochs_array.plot_image(
            combine=combine,
            title=f'EEG Envelope @event: {_event}',
        )
        pdf.savefig(figs[0].figure)

        evoked = epochs_array.average()
        evoked.apply_baseline(baseline=(None, 0))
        fig = evoked.plot_joint(
            title=f'EEG Evoked @event: {_event}',
            times=evoked_plot_times,
        )
        pdf.savefig(fig.figure)

        epochs_array = meg_epochs[_event].copy().crop(
            *crop_args).apply_hilbert(envelope=True)
        print(epochs_array)
        figs = epochs_array.plot_image(
            picks='mag',
            combine=combine,
            title=f'MEG Envelope @event: {_event}',
        )
        pdf.savefig(figs[0].figure)

        evoked = epochs_array.average()
        evoked.apply_baseline(baseline=(None, 0))
        fig = evoked.plot_joint(
            title=f'MEG Evoked @event: {_event}',
            times=evoked_plot_times,
        )
        pdf.savefig(fig.figure)


# %% ---- 2025-06-19 ------------------------
# Pending

# %%

# %%

# %%
