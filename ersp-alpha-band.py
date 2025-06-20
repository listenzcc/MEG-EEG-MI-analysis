"""
File: ersp-alpha-band.py
Author: Chuncheng Zhang
Date: 2025-06-20
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the ERSP field map for the MEG & EEG epochs/evoked.

    - Alpha band: (8-12 Hz)
    - Time range: (0, 4) seconds
    - Unit: dB

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-20 ------------------------
# Requirements and constants
import mne
import argparse

from rich import print
from pathlib import Path
from tqdm.auto import tqdm
from threading import Thread

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

parse = argparse.ArgumentParser('ERSP Analysis')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()

subject_directory = Path(args.subject_dir)
print(f'Subject directory: {subject_directory}')
# subject_directory = Path('./rawdata/S01_20220119')
# %% ---- 2025-06-20 ------------------------
# Function and class


def read_epochs_from_directory(directory: Path):
    found = find_ds_directories(directory)
    print(found)

    use_latest_k_files = 8  # 8
    mds = [read_ds_directory(f) for f in found[-use_latest_k_files:]]
    dev_head_t = mds[0].raw.info['dev_head_t']
    for md in mds:
        md.raw.info['dev_head_t'] = dev_head_t
        md.add_proj()
        md.convert_raw_to_epochs(tmin=-3, tmax=6, decim=6)

    ts = []
    for md in tqdm(mds, 'Load MEG&EEG epochs'):
        # md.eeg_epochs.load_data()
        t = Thread(target=md.eeg_epochs.load_data, daemon=True)
        t.start()
        ts.append(t)
        # md.meg_epochs.load_data()
        t = Thread(target=md.meg_epochs.load_data, daemon=True)
        t.start()
        ts.append(t)

    for t in ts:
        t.join()

    eeg_epochs = mne.concatenate_epochs([md.eeg_epochs for md in mds])
    meg_epochs = mne.concatenate_epochs([md.meg_epochs for md in mds])

    return eeg_epochs, meg_epochs


def compute_tfr(epochs):
    freqs = [8, 9, 10, 11, 12]  # Alpha band frequencies
    time_range = (-2, 4)  # Time range in seconds
    compute_tfr_method = 'morlet'
    baseline_mode = 'logratio'
    tfr = epochs.compute_tfr(
        method=compute_tfr_method,
        average=False,
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        n_jobs=20,
        decim=2
    )
    tfr.crop(*time_range).apply_baseline(baseline=(None, 0), mode=baseline_mode)
    return tfr


# %% ---- 2025-06-20 ------------------------
# Play ground
eeg_epochs, meg_epochs = read_epochs_from_directory(subject_directory)
print(eeg_epochs)
print(meg_epochs)
print(eeg_epochs.event_id)

# %%

matplotlib.use('pdf')
for epochs, suptitle in zip([eeg_epochs, meg_epochs],
                            ['ERSP Alpha Band (EEG)', 'ERSP Alpha Band (MEG)']):

    # Compute TFR for epochs
    tfr = compute_tfr(epochs)
    tfr_df = tfr.to_data_frame(long_format=True)
    print(tfr)
    print(tfr_df)

    # Average across frequencies
    df = tfr_df.copy()
    df = df.groupby(['condition', 'time', 'channel', 'ch_type'], observed=True)[
        'value'].mean().reset_index()
    print(df)
    save_path = Path('data/erd/h5', subject_directory.name, f'{suptitle}.h5')
    df.to_hdf(save_path, key='df', mode='w', format='table')
    print(f'Saved ERSP data frame to: {save_path}')

    # Plotting
    conditions = df['condition'].unique()
    n = len(conditions)
    fig, axes = plt.subplots(
        1, n+1, figsize=(5*n, 5),
        gridspec_kw={"width_ratios": [10] * n + [1]}
    )

    for i, cond in enumerate(conditions):
        _df = df.query(f'condition=="{cond}"').copy()
        _df['value'] *= 10  # Convert to dB

        vmax = 5
        vmin = -10

        ax = axes[i]
        ax.scatter(_df['time'], _df['channel'], c=_df['value'],
                   cmap='viridis', marker='s',
                   vmin=vmin, vmax=vmax)
        ax.set_title(f'ERSP @event: {cond}')
        ax.set_xlabel('Time (s)')
        if i == 0:
            ax.set_ylabel('Channel')
        else:
            ax.set_yticks([])

    fig.colorbar(axes[0].collections[0], cax=axes[-1],
                 orientation='vertical').ax.set_yscale('linear')
    fig.suptitle(suptitle, fontsize=16)

    save_path = Path('data/erd/pdf', subject_directory.name, f'{suptitle}.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f'Saved ERSP field map to: {save_path}')

# %% ---- 2025-06-20 ------------------------
# Pending


# %% ---- 2025-06-20 ------------------------
# Pending
