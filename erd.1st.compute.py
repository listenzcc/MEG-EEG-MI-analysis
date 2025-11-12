"""
File: erd.1st.compute.py
Author: Chuncheng Zhang
Date: 2025-10-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read data and select channels.
    Compute TRF.
    Perform cluster tests.
    Visualization.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-10-09 ------------------------
# Requirements and constants
from util.io.file import save
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
from util.easy_import import *

from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.stats import permutation_t_test

# %%
# Use the arguments
try:
    parse = argparse.ArgumentParser(f'{__file__}', exit_on_error=False)
    parse.add_argument('-s', '--subject-name', required=True)
    parse.add_argument('-m', '--mode', default='meg')
    args = parse.parse_args()

    profile = omegaconf.DictConfig(dict(
        subject_name=args.subject_name,
        mode=args.mode,
    ))

except Exception as err:
    # ! Not allow any errors.
    logger.exception(err)
    raise RuntimeError('Can not bare error in profile.')

    # Or, use default profile.
    profile = omegaconf.DictConfig(dict(
        subject_name='S01_20220119',
        mode='meg',
    ))

profile.merge_with(dict(
    subject_dir=Path('rawdata', profile.subject_name),
    output_dir=Path('data.v2', 'erd-2.permutation1000', profile.subject_name),
    use_latest_ds_dirs=8,  # 2 or 8, lower value means loading fewer data
    epochs_kwargs={'tmin': -2, 'tmax': 5, 'decim': 12},  # sfreq is 1200 Hz
))

logger.info(f'{profile}')

profile.output_dir.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-10-09 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Read from file
    found = find_ds_directories(profile.subject_dir)
    mds = [read_ds_directory(p) for p in found[-profile.use_latest_ds_dirs:]]

    # The concat requires the same dev_head_t
    dev_head_t = mds[0].raw.info['dev_head_t']

    # Read data and convert into epochs
    event_id = mds[0].event_id
    for md in tqdm(mds, 'Convert to epochs'):
        md.raw.info['dev_head_t'] = dev_head_t
        md.add_proj()
        md.generate_epochs(**profile.epochs_kwargs)

        if profile.mode in ['eeg', 'all']:
            md.eeg_epochs.load_data()

        if profile.mode in ['meg', 'all']:
            md.meg_epochs.load_data()

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    groups = []
    for i, e in enumerate(mds):
        if profile.mode in ['eeg', 'all']:
            groups.extend([i for _ in range(len(e.eeg_epochs))])
        else:
            groups.extend([i for _ in range(len(e.meg_epochs))])

    if profile.mode in ['eeg', 'all']:
        eeg_epochs = mne.concatenate_epochs(
            [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    else:
        eeg_epochs = None

    if profile.mode in ['meg', 'all']:
        meg_epochs = mne.concatenate_epochs(
            [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])
    else:
        meg_epochs = None

    return eeg_epochs, meg_epochs, groups


# %% ---- 2025-10-09 ------------------------
# Load epochs
mds, event_id = read_data()
event_ids = sorted(list(event_id.keys()))
eeg_epochs, meg_epochs, groups = concat_epochs(mds)

# Select channels
if profile.mode == 'meg':
    epochs = meg_epochs
    epochs.pick(['MLC42', 'MZC03', 'MRC42'])
    n_channels = 3

if profile.mode == 'eeg':
    epochs = eeg_epochs
    epochs.pick(['C3', 'Cz', 'C4'])
    n_channels = 3

# %%
# Setup for erd analysis
freqs = np.arange(2, 36)  # frequencies from 2-35Hz
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
tmin, tmax = -1, 4  # time crops AFTER tfr
baseline = (-1, 0)  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

n_permutations = 1000
# kwargs for cluster test
kwargs = dict(
    n_permutations=n_permutations, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)

# %%
tfr = epochs.compute_tfr(
    method='morlet',  # "multitaper",
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
)
# tfr.crop(tmin, tmax).apply_baseline(baseline, "logratio")  # "percent")

for event in tqdm(event_ids, 'TFR, test and visualization'):
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    tfr_ev_avg = tfr_ev.average()
    tfr_ev_avg.crop(tmin, tmax).apply_baseline(baseline, "logratio")  # "percent")

    saving = dict(
        tfr_ev_avg=tfr_ev_avg,
    )
    save(saving, profile.output_dir.joinpath(
        f'{profile.mode}-{event}-tfr_ev_avg.dump'))

    fig, axes = plt.subplots(
        1, n_channels+1, figsize=((n_channels+1)*4, 4),
        gridspec_kw={"width_ratios": [10]*n_channels + [1]}
    )
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        t1, c1, p1, h1 = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
        # negative clusters
        t2, c2, p2, h2 = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)
        d = tfr_ev.data[:, ch].copy()
        shape = d.shape  # (n_freqs, n_times)
        d = d.reshape(d.shape[0], -1)
        t, p, h = permutation_t_test(d, n_permutations=n_permutations)

        saving = dict(
            t1=t1,
            t2=t2,
            t=t,
            p=p,
            shape=shape
        )
        save(saving, profile.output_dir.joinpath(
            f'{profile.mode}-{event}-t1-t2.dump'))
        continue

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        if len(c1) == 0:
            c = np.stack(c2, axis=2)  # combined clusters
        elif len(c2) == 0:
            c = np.stack(c1, axis=2)  # combined clusters
        else:
            c = np.stack(c1 + c2, axis=2)  # combined clusters

        p = np.concatenate((p1, p2))  # combined p-values

        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev_avg.plot(
            [ch],
            cmap="RdBu",
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            mask=mask,
            mask_style="mask",
        )

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    fig.savefig(profile.output_dir.joinpath(f'{profile.mode}-{event}.png'))
    # plt.show()

# %% ---- 2025-10-09 ------------------------
# Pending

# %% ---- 2025-10-09 ------------------------
# Pending
