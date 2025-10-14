"""
File: mvpa.source.1.py
Author: Chuncheng Zhang
Date: 2025-09-16
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read X, y, times, groups.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-16 ------------------------
# Requirements and constants
from mne.beamformer import apply_dics_tfr_epochs, make_dics
from mne.time_frequency import csd_tfr
from util.io.file import load, save
from util.easy_import import *
from util.subject_fsaverage import SubjectFsaverage
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# %%
subject_directory = Path("./rawdata/S07_20231220")

parse = argparse.ArgumentParser('Compute TFR')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

subject_name = subject_directory.name

data_directory = Path(f'./data/fsaverage/{subject_name}')

# %% ---- 2025-09-16 ------------------------
# Function and class


def read_data():
    """
    Read data (.ds directories) and convert raw to epochs.
    """
    # Setup options
    epochs_kwargs = {"tmin": -2, "tmax": 5, "decim": 6}
    use_latest_ds_directories = 8  # 8

    # Read from file
    mds = []
    found = find_ds_directories(subject_directory)
    mds.extend([read_ds_directory(p)
               for p in found[-use_latest_ds_directories:]])

    # The concat requires the same dev_head_t
    dev_head_t = mds[0].raw.info["dev_head_t"]

    # Read data and convert into epochs
    event_id = mds[0].event_id
    for md in tqdm(mds, "Convert to epochs"):
        md.raw.info["dev_head_t"] = dev_head_t
        md.add_proj()
        md.generate_epochs(**epochs_kwargs)
        md.eeg_epochs.load_data()
        md.meg_epochs.load_data()
        md.eeg_epochs.apply_baseline((-2, 0))
        md.meg_epochs.apply_baseline((-2, 0))

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    groups = []
    for i, e in enumerate(mds):
        groups.extend([i for _ in range(len(e.meg_epochs))])
    groups = np.array(groups)

    eeg_epochs = mne.concatenate_epochs(
        [md.eeg_epochs for md in tqdm(mds, "Concat EEG Epochs")]
    )
    meg_epochs = mne.concatenate_epochs(
        [md.meg_epochs for md in tqdm(mds, "Concat MEG Epochs")]
    )

    return eeg_epochs, meg_epochs, groups


# %%
# Load the labels
subject = SubjectFsaverage()
parc = "aparc_sub"
labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir
)
labels_parc_df = pd.DataFrame(
    [(e.name, e) for e in labels_parc], columns=["name", "label"]
)
labels_parc_df = labels_parc_df[labels_parc_df['name'].map(
    lambda e: not e.startswith('unknown'))]
labels_parc_df

# %%
# Load epochs (MEG & EEG)
evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs, groups = concat_epochs(mds)

# %%
# Prepare inverse computation
stuff_estimate_snr = load(data_directory.joinpath('stuff-estimate-snr.dump'))

fwd_eeg = stuff_estimate_snr['fwd_eeg']
fwd_meg = stuff_estimate_snr['fwd_meg']
noise_cov = {
    'eeg': stuff_estimate_snr['cov_eeg'],
    'meg': stuff_estimate_snr['cov_meg'],
}

# Compute inverse operator
print('Computing inverse_operator')
with redirect_stdout(io.StringIO()):
    inverse_operator = dict(
        eeg=subject.make_inverse_operator(
            eeg_epochs.info, fwd_eeg, noise_cov['eeg']),
        meg=subject.make_inverse_operator(
            meg_epochs.info, fwd_meg, noise_cov['meg']),
    )
print(inverse_operator)

# Compute inverse
snr = 3.0  # Standard assumption for average data but using it for single trial
kwargs = dict(
    lambda2=1.0 / snr**2,
    method="dSPM"  # use dSPM method (could also be MNE or sLORETA)
)

# # Compute EEG inverse
# print('Computing EEG inverse')
# eeg_epochs.set_eeg_reference(projection=True)
# eeg_epochs_stc = mne.minimum_norm.apply_inverse_epochs(
#     eeg_epochs, inverse_operator['eeg'], **kwargs)
# print(len(eeg_epochs_stc), eeg_epochs_stc[0])

# # Compute MEG inverse
# print('Computing MEG inverse')
# meg_epochs_stc = mne.minimum_norm.apply_inverse_epochs(
#     meg_epochs, inverse_operator['meg'], **kwargs)
# print(len(meg_epochs_stc), meg_epochs_stc[0])


# %% ---- 2025-09-16 ------------------------
# Load X, y and save them.
tmin = -1
tmax = 4
freqs = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
freqs = [e for e in range(8, 32, 4)]
print(freqs)

# %%


for mode, epochs in [('meg', meg_epochs),
                     ('eeg', eeg_epochs)]:
    # Compute tfr
    epochs_tfr = epochs.compute_tfr(
        'morlet', freqs, n_cycles=freqs, return_itc=False, output="complex", average=False, n_jobs=n_jobs)
    print(epochs_tfr)

    # Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
    # We are interested in increases in power relative to the baseline period, so
    # we will make a separate CSD for just that period as well.
    csd = csd_tfr(epochs_tfr, tmin=tmin, tmax=tmax)
    baseline_csd = csd_tfr(epochs_tfr, tmin=tmin, tmax=0)

    # compute scalar DICS beamfomer
    fwd = stuff_estimate_snr[f'fwd_{mode}']
    filters = make_dics(
        epochs.info,
        fwd,
        csd,
        noise_csd=baseline_csd,
        pick_ori="max-power",
        reduce_rank=True,
        real_filter=True,
    )

    # project the TFR for each epoch to source space
    epochs_stcs_generator = apply_dics_tfr_epochs(
        epochs_tfr, filters, return_generator=True)

    # average across frequencies and epochs
    data = []
    # Walk through samples
    for stcs in tqdm(epochs_stcs_generator, 'Lvl1', total=len(epochs)):
        d1 = []
        # Walk through freqs
        for stc in tqdm(stcs, 'Lvl2'):
            stc.crop(tmin, tmax)
            d2 = []
            for label in labels_parc_df['label']:
                # d shape is (n_vertex, n_times)
                d = stc.in_label(label).data
                # Average across vertex
                d2.append(np.mean((d * np.conj(d)).real, axis=0))
            d1.append(d2)
        data.append(d1)

    # data shape is (n_samples, n_freqs, n_vertex, n_times)
    data = np.array(data)
    print(data.shape)
    saving = {
        'X': data,
        'freqs': freqs
    }
    save(saving, data_directory.joinpath(f'{mode}-tfr-source-X-freqs.dump'))
    continue

    # Compute stc
    stc = mne.minimum_norm.apply_inverse_epochs(
        epochs, inverse_operator[mode], **kwargs)

    # Save stc
    epochs.crop(tmin, tmax)

    [e.crop(tmin, tmax) for e in stc]

    times = epochs.times
    y = epochs.events[:, -1]

    data = []
    for label in tqdm(labels_parc_df['label'], 'Read labels'):
        # d shape is (n_epochs, n_vertices, n_times)
        d = np.array([e.in_label(label).data for e in stc])
        data.append(np.mean(d, axis=1))
    X = np.array(data).transpose((1, 0, 2))

    print(f'{X.shape=}, {times.shape=}, {y.shape=}, {groups.shape=}')

    # Save the data
    saving = {
        'X': X,
        'y': y,
        'times': times,
        'groups': groups,
    }
    save(saving, data_directory.joinpath(
        f'{mode}-source-X-y-times-groups.dump'))


# %% ---- 2025-09-16 ------------------------
# Train & test

# %% ---- 2025-09-16 ------------------------
# Pending
