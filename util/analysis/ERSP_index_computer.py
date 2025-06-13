"""
File: ERSP_index_computer.py
Author: Chuncheng Zhang
Date: 2025-05-22
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    ERDS plot for the MEG and EEG data.
    ERDS maps are also known as ERSP (event-related spectral perturbation)

    - <https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html>
    - <https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html#footcite-makeig1993>

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-22 ------------------------
# Requirements and constants
import numpy as np
import pandas as pd
from pathlib import Path

import mne
from mne.stats import permutation_cluster_1samp_test as pcluster_test

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

from ..logging import logger

freqs = np.arange(2, 36)  # frequencies from 2-35Hz

cmap = "RdBu"
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

cluster_test_kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)

# %% ---- 2025-05-22 ------------------------
# Function and class


def ERSP_analysis_with_saving(epochs, event_ids, selected_channels, pdf_path, df_path, tfr_path):
    # Make pdf_path
    pdf_path = Path(pdf_path)
    if not pdf_path.suffix == ".pdf":
        raise ValueError(f"ERSP pdf file must be a .pdf file: {pdf_path}")
    if pdf_path.exists():
        logger.warning(f"ERSP pdf file already exists: {pdf_path}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute tfr
    tfr = ERSP_analysis(epochs, selected_channels)
    # Convert into dataframe
    df = tfr.to_data_frame(time_format=None, long_format=True)

    # Save into files
    # How to read: df = pd.read_hdf(df_path)
    # Average across the epochs
    _df = df.groupby(
        ["condition", "freq", "time", "channel", "ch_type"], observed=True)["value"].mean().reset_index()
    _df.to_hdf(df_path, key='df', mode='w', format='table')
    # How to read: tfr = mne.time_frequency.read_tfrs(tfr_path)
    tfr.save(tfr_path, overwrite=True)

    with PdfPages(pdf_path) as pdf:
        num_channels = len(selected_channels)
        # Plot the ERSP for every events.
        for event in event_ids:
            # select desired epochs for visualization
            tfr_ev = tfr[event]
            # The latest column is colorbar
            fig, axes = plt.subplots(
                1, num_channels+1, figsize=(num_channels*4, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
            )
            for ch, ax in enumerate(axes[:-1]):  # for each channel
                try:
                    title = epochs.ch_names[ch]
                    # positive clusters
                    _, c1, p1, _ = pcluster_test(
                        tfr_ev.data[:, ch], tail=1, **cluster_test_kwargs)
                    # negative clusters
                    _, c2, p2, _ = pcluster_test(
                        tfr_ev.data[:, ch], tail=-1, **cluster_test_kwargs)

                    # note that we keep clusters with p <= 0.05 from the combined clusters
                    # of two independent tests; in this example, we do not correct for
                    # these two comparisons
                    c = np.stack(c1 + c2, axis=2)  # combined clusters
                    p = np.concatenate((p1, p2))  # combined p-values
                    mask = c[..., p <= 0.05].any(axis=-1)

                    # plot TFR (ERDS map with masking)
                    tfr_ev.average().plot(
                        [ch],
                        cmap=cmap,
                        cnorm=cnorm,
                        axes=ax,
                        colorbar=False,
                        show=False,
                        mask=mask,
                        mask_style="mask",
                    )

                    ax.set_title(title, fontsize=10)
                    ax.axvline(0, linewidth=1, color="black",
                               linestyle=":")  # event
                    if ch != 0:
                        ax.set_ylabel("")
                        ax.set_yticklabels("")
                except Exception:
                    continue
            fig.colorbar(axes[0].images[-1], cax=axes[-1]
                         ).ax.set_yscale("linear")
            fig.suptitle(f"ERDS ({event})")
            if pdf:
                pdf.savefig(fig)
            else:
                plt.show()

        # Plot the ERSP time series
        # Map to frequency bands:
        freq_bounds = {"_": 0, "Delta": 3, "Theta": 7,
                       "Alpha": 13, "Beta": 35, "Gamma": 140}
        df["band"] = pd.cut(
            df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
        )

        # Filter to retain only relevant frequency bands:
        freq_bands_of_interest = ["Delta", "Theta", "Alpha", "Beta"]
        df = df[df.band.isin(freq_bands_of_interest)]
        df["band"] = df["band"].cat.remove_unused_categories()

        # Order channels for plotting:
        df["channel"] = df["channel"].cat.reorder_categories(
            selected_channels, ordered=True)

        g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
        g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
        axline_kw = dict(color="black", linestyle="dashed",
                         linewidth=0.5, alpha=0.5)
        g.map(plt.axhline, y=0, **axline_kw)
        g.map(plt.axvline, x=0, **axline_kw)
        g.set(ylim=(-1, 2))
        g.set_axis_labels("Time (s)", "ERDS")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
        g.add_legend(ncol=len(event_ids), loc="lower center")
        g.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
        if pdf:
            pdf.savefig(g.figure)
        else:
            plt.show()


def ERSP_analysis(epochs: mne.Epochs, selected_channels: list):
    epochs = epochs.copy()
    epochs.load_data()
    epochs.pick_channels(selected_channels)
    logger.debug(f"Selected channels: {selected_channels}, {epochs.ch_names}")
    # tmin = epochs.tmin
    # tmax = epochs.tmax
    tmin = -2
    tmax = epochs.tmax
    baseline = (None, 0)  # baseline interval (in s)

    assert tmin >= epochs.tmin, f'Invalid tmin: {tmin} < {epochs.tmin}'
    assert tmax <= epochs.tmax, f'Invalid tmax: {tmax} > {epochs.tmax}'

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
    '''
mode‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
Perform baseline correction by
subtracting the mean of baseline values (‘mean’)
dividing by the mean of baseline values (‘ratio’)
dividing by the mean of baseline values and taking the log (‘logratio’)
subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
subtracting the mean of baseline values and dividing by the standard deviation of baseline values (‘zscore’)
dividing by the mean of baseline values, taking the log, and dividing by the standard deviation of log baseline values (‘zlogratio’)
    '''
    mode = 'percent'
    mode = 'mean'
    mode = 'logratio'
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode=mode)

    return tfr


# %% ---- 2025-05-22 ------------------------
# Play ground


# %% ---- 2025-05-22 ------------------------
# Pending


# %% ---- 2025-05-22 ------------------------
# Pending
