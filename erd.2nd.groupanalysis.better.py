"""
File: erd.2nd.groupanalysis.better.py
Author: Chuncheng Zhang
Date: 2025-10-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Perform 2nd groupanalysis for the erd results.
    The script draws better, I hope.

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

def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=12, va='bottom')
    return


n_channels = 3
n_rows = 5
fig, axes = plt.subplots(
    n_rows+1, n_channels, figsize=(12, 12),  # (n_channels*2, 3*n_rows),
    gridspec_kw={"height_ratios": [10]*n_rows + [1]}
)

event_names = [e.title()
               for e in ['hand', 'wrist', 'elbow', 'shoulder',  'rest']]

for i_event, event in enumerate(['1', '2', '3', '4', '5']):
    t_stats_files = sorted(DATA_DIR.rglob(f'{MODE}-{event}-t1-t2.dump'))
    tfr_freqs_files = sorted(DATA_DIR.rglob(
        f'{MODE}-{event}-tfr_ev_avg.dump'))
    print(f'{t_stats_files=}')
    print(f'{tfr_freqs_files=}')

    if False:
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

    #
    tfrs = []
    for p in tfr_freqs_files:
        obj = load(p)
        tfrs.append(obj['tfr_ev_avg'])

    tfr = tfrs[0].copy()

    tmap = [t < 0 and t > -1 for t in tfr.times]
    print(tfr.times)
    for i, t in enumerate(tfrs):
        d = t.data[:, :, tmap]
        print(i, np.max(d), np.min(d), np.mean(d))

    # Convert into dB
    # tfr.data shape is (n_channels, n_freqs, n_times)
    # tfr.data = 10 * tfr.data
    tfr.data = 10 * np.mean(np.stack([t.data for t in tfrs]), axis=0)
    for i in range(tfr.data.shape[0]):
        for j in range(tfr.data.shape[1]):
            tfr.data[i][j] -= np.mean(tfr.data[i, j, tfr.times < 0])
    # tfr.data -= np.mean(tfr.data[:, :, tfr.times < 0])

    # min, center & max ERDS
    vmin, vmax, vcenter, cmap = -5, 1, 0, 'RdBu_r'
    vmin, vmax, vcenter, cmap = -5, 1, -3, 'RdBu'
    vmin, vmax, vcenter, cmap = -3, 3, 0, 'RdBu'
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # draw for each channel
    for i_ch in range(n_channels):
        ax = axes[i_event, i_ch]
        # plot TFR (ERDS map with masking)
        tfr.plot(
            [i_ch],
            cmap=cmap,
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            # mask=mask,
            # mask_style="mask",
        )

        # 3dB contour
        z = np.abs(tfr.data[i_ch])
        m, n = np.meshgrid(tfr.freqs, tfr.times)
        # cc = ax.contour(n, m, z.transpose(), levels=[3])
        # ax.clabel(cc, inline=True, fontsize=10, fmt='-%1.0f dB')

        ax.set_title(tfr.ch_names[i_ch], fontsize=10, fontweight='bold')
        ax.axvline(0, linewidth=1, color="black", linestyle=":")

        if i_ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
        else:
            add_top_left_notion(ax, 'abcdefghi'[i_event])

        if i_event != 4:
            ax.set_xlabel("")
            # ax.set_xticklabels("")

        if i_ch == 2:
            ax2 = ax.twinx()
            ax2.set_yticklabels("")
            ax2.set_yticks([])
            ax2.set_ylabel(event_names[i_event],
                           rotation=-90,
                           ha='center',
                           va='center',
                           fontweight='bold',
                           labelpad=20,  # 增加標籤與軸的距離
                           fontsize=14,  # 增大字體
                           )


cbar = fig.colorbar(axes[0, 0].images[-1], cax=axes[5, 1],
                    orientation='horizontal')
cbar.ax.set_xscale("linear")
cbar.outline.set_visible(False)
axes[5, 1].set_title('dB')

# Delete the unwanted axes
fig.delaxes(axes[5, 0])
fig.delaxes(axes[5, 2])

fig.suptitle(f"TFR ({MODE.upper()})", fontweight='bold')
fig.tight_layout()

file_name = OUTPUT_DIR.joinpath(f'TFR(ERD)-{MODE}.png')

fig.savefig(file_name)
print(f'Saved into {file_name=}')

# plt.show()


# %% ---- 2025-10-09 ------------------------
# Pending

# %% ---- 2025-10-09 ------------------------
# Pending

# %%
