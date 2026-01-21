"""
File: summary.analysis.erd.py
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
# OUTPUT_DIR = Path('./data.v2/erd.permutation1000.groupanalysis')
OUTPUT_DIR = Path('./data/erd.exampleChannels.detail')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODE = 'eeg'

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
    n_rows+1, n_channels, figsize=(8, 16),  # (n_channels*2, 3*n_rows),
    gridspec_kw={"height_ratios": [10]*n_rows + [1]},
    dpi=300
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
    # print(tfr.times)
    for i, t in enumerate(tfrs):
        d = t.data[:, :, tmap]
        # print(i, np.max(d), np.min(d), np.mean(d))

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
    vmin, vmax, vcenter, cmap = -3, 3, 0, 'RdBu_r'
    vmin, vmax, vcenter, cmap = -3.5, 3.5, 0, 'RdBu_r'

    if MODE == 'meg':
        vmin, vmax, vcenter, cmap = -2, 2, 0, 'RdBu_r'

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

        # The time-freq matrix of the given channel
        # ! Save data here
        mat = tfr.data[i_ch]
        ch_name = tfr.ch_names[i_ch]
        print(f'{i_event=}, {i_ch=}, {mat.shape=}, {tfr.times.shape=}, {tfr.freqs.shape=}, {ch_name=}')
        obj = {
            'i_event': i_event,
            'i_ch': i_ch,
            'mat': mat,
            'times': tfr.times,
            'freqs': tfr.freqs,
            'ch_name': ch_name
        }
        file_name = f'{MODE}-{i_event}-{i_ch}-{ch_name}.dump'
        file_path = OUTPUT_DIR / file_name
        import joblib
        joblib.dump(obj, file_path)
        print(f'Saved to {file_path}')

        # The loop ends here
        continue

    # The loop ends here
    continue


exit(0)


# %% ---- 2025-10-09 ------------------------
# Pending

# %% ---- 2025-10-09 ------------------------
# Pending

# %%
