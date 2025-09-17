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
from util.io.file import load, save
from util.bands import Bands
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
    epochs_kwargs = {"tmin": -3, "tmax": 5, "decim": 6}
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

# Compute EEG inverse
print('Computing EEG inverse')
eeg_epochs.set_eeg_reference(projection=True)
eeg_epochs_stc = mne.minimum_norm.apply_inverse_epochs(
    eeg_epochs, inverse_operator['eeg'], **kwargs)
print(len(eeg_epochs_stc), eeg_epochs_stc[0])

# Compute MEG inverse
print('Computing MEG inverse')
meg_epochs_stc = mne.minimum_norm.apply_inverse_epochs(
    meg_epochs, inverse_operator['meg'], **kwargs)
print(len(meg_epochs_stc), meg_epochs_stc[0])


# %% ---- 2025-09-16 ------------------------
# Load X, y
tmin = 0
tmax = 4
times = eeg_epochs.copy().crop(tmin, tmax).times
y = eeg_epochs.events[:, -1]
groups = np.array(groups)

data = []
for label in tqdm(labels_parc_df['label'], 'Read labels'):
    # d.shape = (n_epochs, n_vertices, n_times)
    d = np.array([e.in_label(label).crop(
        tmin, tmax).data for e in eeg_epochs_stc])
    data.append(np.mean(d, axis=1))
X = np.array(data).transpose((1, 0, 2))

print(f'{X.shape=}, {times.shape=}, {y.shape=}, {groups.shape=}')

# %%
all_data = {
    'X': X,
    'y': y,
    'times': times,
    'groups': groups,
}
save(all_data, data_directory.joinpath('X-y-times-groups.dump'))


# %% ---- 2025-09-16 ------------------------
# Train & test


# %% ---- 2025-09-16 ------------------------
# Pending
