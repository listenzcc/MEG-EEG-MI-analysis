"""
File: ERSP_index.py
Author: Chuncheng Zhang
Date: 2025-05-22
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    ERDS plot for the MEG and EEG data.
    ERDS maps are also known as ERSP (event-related spectral perturbation)

    - [https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html]
    - [https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html#footcite-makeig1993]

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
from pathlib import Path

import mne
from mne.stats import permutation_cluster_1samp_test as pcluster_test

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

from ..logging import logger

freqs = np.arange(2, 36)  # frequencies from 2-35Hz
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = (-1, 0)  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)  # for cluster test

# %% ---- 2025-05-22 ------------------------
# Function and class


def ERSP_analysis_with_pdf(pdf_path, epochs, event_ids, selected_channels):
    pdf_path = Path(pdf_path)
    if not pdf_path.suffix == ".pdf":
        raise ValueError(f"ERSP pdf file must be a .pdf file: {pdf_path}")
    if pdf_path.exists():
        logger.warning(f"ERSP pdf file already exists: {pdf_path}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        logger.debug(f"ERSP analysis with pdf: {pdf_path}")
        ERSP_analysis(epochs, event_ids, selected_channels, pdf)


def ERSP_analysis(epochs: mne.Epochs, event_ids: dict, selected_channels: list, pdf: PdfPages):
    epochs = epochs.copy()
    epochs.load_data()
    epochs.pick_channels(selected_channels)
    logger.debug(f"Selected channels: {selected_channels}, {epochs.ch_names}")
    tmin = epochs.tmin
    tmax = epochs.tmax

    tfr = epochs.compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=2,
    )
    tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")

    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(
            1, len(selected_channels)+1, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
        )
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            title = epochs.ch_names[ch]
            # positive clusters
            _, c1, p1, _ = pcluster_test(
                tfr_ev.data[:, ch], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = pcluster_test(
                tfr_ev.data[:, ch], tail=-1, **kwargs)

            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot(
                [ch],
                cmap="RdBu",
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
        fig.colorbar(axes[0].images[-1], cax=axes[-1]
                     ).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({event})")
        if pdf:
            pdf.savefig(fig)
        else:
            plt.show()

# %% ---- 2025-05-22 ------------------------
# Play ground


# %% ---- 2025-05-22 ------------------------
# Pending


# %% ---- 2025-05-22 ------------------------
# Pending
