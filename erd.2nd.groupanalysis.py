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

DATA_DIR = Path('./data.v2/erd')
OUTPUT_DIR = Path('./data.v2/erd')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODE = 'meg'

# %%
n_observations = 10  # 10 subjects
pval = 0.001  # arbitrary
df = n_observations - 1  # degrees of freedom for the test
thresh = scipy.stats.t.ppf(1-pval, df)  # one-tailed, t distribution

# %% ---- 2025-10-09 ------------------------
# Function and class


def group_level_cluster_test(t_stats_all, tail, n_permutations=5000):
    """
    t_stats_all: 所有被试的统计量数组
    tail: 0-双尾, 1-正尾, -1-负尾
    """
    # 组水平单样本t检验 (检验均值是否显著不等于0)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        t_stats_all,
        threshold=thresh if tail > 0 else -thresh,
        n_permutations=n_permutations,
        tail=tail,
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
        for p in t_stats_files:
            obj = load(p)
            t1_all.append(obj['t1'])
            t2_all.append(obj['t2'])
        t1_all = np.array(t1_all)
        t2_all = np.array(t2_all)

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
        fig, axes = plt.subplots(
            1, n_channels+1, figsize=((n_channels+1)*4, 4),
            gridspec_kw={"width_ratios": [10]*n_channels + [1]}
        )
        for ch, ax in enumerate(axes[:-1]):  # for each channel
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
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({MODE}@{event})")
        pdf.savefig(fig)
        # plt.show()


# %% ---- 2025-10-09 ------------------------
# Pending


# %% ---- 2025-10-09 ------------------------
# Pending
