"""
File: erd.2nd.groupanalysis.py
Author: Chuncheng Zhang
Date: 2025-10-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Perform 2nd groupanalysis for the erd results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-10-09 ------------------------
# Requirements and constants
import scipy
from util.io.file import load
from util.easy_import import *

from mne.stats import permutation_cluster_1samp_test
from mne.stats import permutation_t_test

DATA_DIR = Path('./data.v2/erd.permutation1000')
OUTPUT_DIR = Path('./data.v2/erd.permutation1000.groupanalysis')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODE = 'meg'

# %%
n_observations = 10  # 10 subjects
pval = 0.001  # arbitrary
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1-pval, df)  # one-tailed, t distribution
thresh

# %% ---- 2025-10-09 ------------------------
# Function and class


def group_level_cluster_test(t_stats_all, tail, n_permutations=5000):
    """
    t_stats_all: 所有被试的统计量数组
    tail: 0-双尾, 1-正尾, -1-负尾
    """
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        t_stats_all,
        # threshold=thresh if tail > 0 else -thresh,
        threshold=None,
        n_permutations=n_permutations,
        tail=tail,
        # adjacency=False,
        out_type='mask'
    )

    return t_obs, clusters, cluster_pv, H0


# %% ---- 2025-10-09 ------------------------
# Play ground
with PdfPages(OUTPUT_DIR.joinpath(f'ERDS-{MODE}.pdf')) as pdf:
    for event in ['1', '2', '3', '4', '5']:
        t_stats_files = sorted(DATA_DIR.rglob(f'{MODE}-{event}-t1-t2.dump'))
        tfr_freqs_files = sorted(DATA_DIR.rglob(
            f'{MODE}-{event}-tfr_ev_avg.dump'))
        print(f'{t_stats_files=}')
        print(f'{tfr_freqs_files=}')

        #
        t1_all = []
        t2_all = []
        t_all = []
        p_all = []
        for p in t_stats_files:
            obj = load(p)
            t1_all.append(obj['t1'])
            t2_all.append(obj['t2'])
            t_all.append(obj['t'])
            p_all.append(obj['p'])
            shape = obj['shape']
        t1_all = np.array(t1_all)
        t2_all = np.array(t2_all)
        t_all = np.array(t_all)

        print(f'{t1_all.shape=}, {t2_all.shape=}, {t_all.shape=}, {shape=}')

        # stophere

        t1, c1, p1, h1 = group_level_cluster_test(t1_all, 1)
        t2, c2, p2, h2 = group_level_cluster_test(t2_all, -1)
        if len(c1) == 0:
            c = np.stack(c2, axis=2)  # combined clusters
        elif len(c2) == 0:
            c = np.stack(c1, axis=2)  # combined clusters
        else:
            c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))
        mask = c[..., p <= 0.05].any(axis=-1)
        print(f'{np.max(t1_all)=}, {np.min(t1_all)=}')
        print(f'{np.max(t2_all)=}, {np.min(t2_all)=}')
        print(f'{len(c1)=}, {len(c2)=}')

        # shape is (n_samples, n_freqs, n_times)
        # total = np.prod(shape[1:])
        # n_permutations = 1000
        # X = t_all
        # print(X.shape)
        # t, p, h = permutation_t_test(X, n_permutations)
        # p_threshold = 0.05
        # print(
        #     f'Positive tail test: {np.sum(p <= p_threshold)=} | Total: {total}')
        # print(f'{t.shape=}, {p.shape=}, {h.shape=}, {shape}')
        # t = t.reshape(shape[1:])
        # prob = p.reshape(shape[1:])
        # mask = prob <= p_threshold

        #
        tfrs = []
        for p in tfr_freqs_files:
            obj = load(p)
            tfrs.append(obj['tfr_ev_avg'])
        tfr = tfrs[0].copy()
        # Convert into dB
        tfr.data = 10 * np.mean(np.stack([t.data for t in tfrs]), axis=0)

        # min, center & max ERDS
        vmin, vmax = -5, 5
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        n_channels = len(tfr.ch_names)
        n_rows = 1
        fig, axes = plt.subplots(
            n_rows, n_channels+1, figsize=((n_channels+1)*4, 4*n_rows),
            gridspec_kw={"width_ratios": [10]*n_channels + [1]}
        )
        # for each channel
        for ch in range(n_channels):
            ax = axes[ch]
            # plot TFR (ERDS map with masking)
            tfr.plot(
                [ch],
                cmap="RdBu_r",
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask,
                mask_style="mask",
            )

            z = np.abs(tfr.data[ch])
            m, n = np.meshgrid(tfr.freqs, tfr.times)
            ax.contour(n, m, z.transpose(), levels=[3])

            ax.set_title(tfr.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        axes[-1].set_title('dB')
        fig.colorbar(axes[0].images[-1], cax=axes[-1]
                     ).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({MODE}@{event})")
        pdf.savefig(fig)
        # plt.show()


# %% ---- 2025-10-09 ------------------------
# Pending

# %% ---- 2025-10-09 ------------------------
# Pending

# %%
