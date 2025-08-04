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
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import matplotlib

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mne.decoding import (
    CSP,
    Scaler,
    Vectorizer,
    LinearModel,
    SlidingEstimator,
    GeneralizingEstimator,
    get_coef,
    cross_val_multiscore,
)

from util.bands import Bands
from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# --------------------------------------------------------------------------------
mode = 'meg'  # 'meg', 'eeg'
band_name = 'all'  # 'delta', 'theta', 'alpha', 'beta', 'gamma', 'all'
subject_directory = Path('./rawdata/S01_20220119')

# Use the arguments
# parse = argparse.ArgumentParser('Compute TFR')
# parse.add_argument('-s', '--subject-dir', required=True)
# args = parse.parse_args()
# subject_directory = Path(args.subject_dir)

# --------------------------------------------------------------------------------
# Prepare the paths
subject_name = subject_directory.name
data_directory = Path(f'./data/MVPA/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

pdf_path = data_directory / f'decoding-{mode}-band-{band_name}.pdf'
dump_path = Path(pdf_path).with_suffix('.dump')

# %% ---- 2025-07-21 ------------------------
# Function and class


def read_data():
    '''
    Read data (.ds directories) and convert raw to epochs.
    The epochs are cut with large scale: (-2, 5) seconds.
    The filter is applied to the scale.
    After the filter, the epochs are cropped with (-1, 4) seconds.
    The method is to prevent cropping effect.
    '''
    # Setup options
    bands = Bands()
    l_freq, h_freq = bands.get_band(band_name)
    epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 6}
    filter_kwargs = {'l_freq': l_freq, 'h_freq': h_freq, 'n_jobs': n_jobs}
    use_latest_ds_directories = 8  # 8

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
# print(groups)

# %% ---- 2025-07-21 ------------------------
# Play ground

if mode == 'meg':
    epochs = meg_epochs.copy().pick_types(meg=True, ref_meg=False)
elif mode == 'eeg':
    epochs = eeg_epochs.copy()
else:
    raise ValueError(f'Unknown mode: {mode}')


epochs = epochs.resample(40, npad="auto")  # resample to 40 Hz

cv = np.max(groups)+1
# MEG signals: n_epochs, n_meg_channels, n_times
X = epochs.get_data(copy=False)
y = epochs.events[:, 2]  # target

# %%
scoring = make_scorer(accuracy_score, greater_is_better=True)


# %%

matplotlib.use('pdf')
with PdfPages(pdf_path) as pdf:
    # Uses all MEG sensors and time points as separate classification
    # features, so the resulting filters used are spatio-temporal
    clf = make_pipeline(
        Scaler(epochs.info),
        Vectorizer(),
        StandardScaler(),  # In question
        LogisticRegression(solver="liblinear"),
    )

    scores = cross_val_multiscore(
        clf, X, y, groups=groups, cv=cv, n_jobs=n_jobs)

    # Mean scores across cross-validation splits
    score = np.mean(scores, axis=0)
    print(f"Spatio-temporal: {100 * score:0.1f}% (Detail: {scores})")

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

    # Over time decoding
    clf = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver="liblinear"))
    )

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
    pdf.savefig(fig)

    # Patterns in sensor space
    clf = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver="liblinear"))
    )

    time_decod = SlidingEstimator(
        clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

    time_decod.fit(X, y)

    coef = get_coef(time_decod, "patterns_", inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(
        coef[:, 0, :].squeeze(), epochs.info, tmin=epochs.times[0])
    joint_kwargs = dict(ts_args=dict(time_unit="s"),
                        topomap_args=dict(time_unit="s"))

    fig = evoked_time_gen.plot_joint(
        times='peaks', title="patterns", show=False, **joint_kwargs
    )
    pdf.savefig(fig)

    # Temporal generalization decoding
    # define the Temporal generalization object
    time_gen = GeneralizingEstimator(
        clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=n_jobs)

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    # Plot the diagonal (it's exactly the same as the time-by-time decoding above)
    fig, ax = plt.subplots()
    ax.plot(epochs.times, np.diag(scores), label="score")
    ax.axhline(1/len(evts), color="k", linestyle="--", label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AccScore")
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Decoding over time")
    pdf.savefig(fig)

    # Plot the temporal generalization matrix
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        scores,
        interpolation="lanczos",
        origin="lower",
        cmap="RdBu_r",
        extent=epochs.times[[0, -1, 0, -1]],
        vmin=0.0,
        vmax=0.5,
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Temporal generalization")
    ax.axvline(0, color="k")
    ax.axhline(0, color="k")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy Score")
    pdf.savefig(fig)

    joblib.dump({
        'times': epochs.times,
        'scores': scores,
        'mode': mode,
        'band_name': band_name,
        'subject_name': subject_name,
    }, dump_path)

print(f'Finished decoding {pdf_path}')

# %% ---- 2025-07-21 ------------------------
# Pending

# %%
