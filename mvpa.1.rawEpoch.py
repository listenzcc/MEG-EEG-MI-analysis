"""
File: mvpa.1.raw.epoch.py
Author: Chuncheng Zhang
Date: 2025-07-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    MPVA with raw epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-21 ------------------------
# Requirements and constants
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)

from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

subject_directory = Path('./rawdata/S01_20220119')

# parse = argparse.ArgumentParser('Compute TFR')
# parse.add_argument('-s', '--subject-dir', required=True)
# args = parse.parse_args()
# subject_directory = Path(args.subject_dir)

subject_name = subject_directory.name
data_directory = Path(f'./data/MVPA.rawEpochs/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-07-21 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    '''
    # Setup options
    epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6}
    filter_kwargs = {'l_freq': 0.1, 'h_freq': 30, 'n_jobs': n_jobs}
    use_latest_ds_directories = 2  # 8

    # Read from file
    found = find_ds_directories(subject_directory)
    mds = [read_ds_directory(p) for p in found[-use_latest_ds_directories:]]

    # The concat requires the same dev_head_t
    dev_head_t = mds[0].raw.info['dev_head_t']

    # Read data and convert into epochs
    event_id = mds[0].event_id
    for md in tqdm(mds, 'Convert to epochs'):
        md.raw.info['dev_head_t'] = dev_head_t
        md.add_proj()
        md.generate_epochs(**epochs_kwargs)

        md.eeg_epochs.load_data()
        md.meg_epochs.load_data()

        md.eeg_epochs.filter(**filter_kwargs)
        md.meg_epochs.filter(**filter_kwargs)

        md.eeg_epochs.crop(tmin=-1, tmax=4)
        md.meg_epochs.crop(tmin=-1, tmax=4)

        md.eeg_epochs.apply_baseline((-1, 0))
        md.meg_epochs.apply_baseline((-1, 0))

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    groups = []
    for i, e in enumerate(mds):
        groups.extend([i for _ in range(len(e.meg_epochs))])

    eeg_epochs = mne.concatenate_epochs(
        [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    meg_epochs = mne.concatenate_epochs(
        [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])

    return eeg_epochs, meg_epochs, groups


evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs, groups = concat_epochs(mds)

# %%
print(eeg_epochs)
print(meg_epochs)
print(groups)

# %% ---- 2025-07-21 ------------------------
# Play ground
epochs = meg_epochs.copy().pick_types(meg=True, ref_meg=False)

cv = np.max(groups)+1
# MEG signals: n_epochs, n_meg_channels, n_times
X = epochs.get_data(copy=False)
y = epochs.events[:, 2]  # target: auditory left vs visual left

# Uses all MEG sensors and time points as separate classification
# features, so the resulting filters used are spatio-temporal
clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear"),  # liblinear is faster than lbfgs
)

scores = cross_val_multiscore(clf, X, y, groups=groups, cv=cv, n_jobs=n_jobs)

# Mean scores across cross-validation splits
score = np.mean(scores, axis=0)
print(f"Spatio-temporal: {100 * score:0.1f}% (Detail: {scores})")

# %%
# CSP: Common Spatial Patterns
if False:
    csp = CSP(n_components=3, norm_trace=False)
    clf_csp = make_pipeline(csp, LinearModel(
        LogisticRegression(solver="liblinear")))
    scores = cross_val_multiscore(
        clf_csp, X, y, groups=groups, cv=cv, n_jobs=None)

    # Mean scores across cross-validation splits
    score = np.mean(scores, axis=0)
    print(f"CSP: {100 * score:0.1f}% (Detail: {scores})")

# %%
# Over time decoding
# We will train the classifier on all left visual vs auditory trials on MEG

clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))

scoring = make_scorer(accuracy_score, greater_is_better=True)
time_decod = SlidingEstimator(
    clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

scores = cross_val_multiscore(
    time_decod, X, y, groups=groups, cv=cv, n_jobs=n_jobs)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label="score")
ax.axhline(1/len(evts), color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("AccScore")
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Sensor space decoding")

# %%
# Temporal generalization decoding
# define the Temporal generalization object
time_gen = GeneralizingEstimator(
    clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=n_jobs)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs.times, np.diag(scores), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("AccScore")
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Decoding MEG sensors over time")

# %% ---- 2025-07-21 ------------------------
# Pending

# %%

# %%
