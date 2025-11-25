"""
File: mvpa.1.rawEpochs.accumulate.noFB.py
Author: Chuncheng Zhang
Date: 2025-07-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    MPVA with raw epochs without FB.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-21 ------------------------
# Requirements and constants
import seaborn as sns
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

# %%

# --------------------------------------------------------------------------------
mode = 'meg'  # 'meg', 'eeg'
band_name = 'all'  # 'delta', 'theta', 'alpha', 'beta', 'gamma', 'all'
subject_directory = Path('./rawdata/S07_20231220')

# Use the arguments
parse = argparse.ArgumentParser('Compute TFR')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

# --------------------------------------------------------------------------------
# Prepare the paths
subject_name = subject_directory.name
OUTPUT_DIR = Path(f'./data/MVPA.accumulate/{subject_name}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
SUMMARY_MODE = True
if SUMMARY_MODE:
    dump_files = list(OUTPUT_DIR.parent.rglob('*.joblib'))
    print(dump_files)
    dfs = [joblib.load(f) for f in dump_files]
    for df, file in zip(dfs, dump_files):
        mode = file.stem.split('.')[-1]
        df['mode'] = mode
    df = pd.concat(dfs)
    print(df)
    import seaborn as sns
    sns.lineplot(df, x='tmax', y='accuracy', hue='mode')
    plt.show()

# %%

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
    epochs_kwargs = {'tmin': -2, 'tmax': 5, 'decim': 12}  # decim=6, 12 ...
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
results = []

for tmax in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0]):
    tmax_idx = len(epochs.times[epochs.times < tmax])
    X_t = X[:, :, :tmax_idx]

    # Spatio-temporal decoding

    # Uses all MEG sensors and time points as separate classification
    # features, so the resulting filters used are spatio-temporal
    clf = make_pipeline(
        Scaler(epochs.info),
        Vectorizer(),
        StandardScaler(),  # In question
        LinearModel(LogisticRegression(solver="lbfgs", max_iter=200)),
    )

    scores = cross_val_multiscore(
        clf, X_t, y, groups=groups, cv=cv, n_jobs=2)

    # Mean scores across cross-validation splits
    score = np.mean(scores, axis=0)
    results.append((tmax, score, scores))
    print(f"Spatio-temporal {tmax=}: {100 * score:0.1f}% (Detail: {scores})")

results = pd.DataFrame(results, columns=['tmax', 'accuracy', 'accuracies'])
results['subject'] = subject_name
results['mode'] = mode
display(results)

joblib.dump(results, OUTPUT_DIR /
            f'mvpa.rawEpochs.accumulate.noFB.{mode}.joblib')
exit(0)


# %%
sns.lineplot(results, x='tmax', y='accuracy')
plt.show()

# %%
# Over time decoding
clf = make_pipeline(
    StandardScaler(),
    LinearModel(LogisticRegression(solver="lbfgs"))
)

time_decod = SlidingEstimator(
    clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

scores = cross_val_multiscore(
    time_decod, X, y, groups=groups, cv=cv, n_jobs=n_jobs)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

print(f'Time decoding: {scores=}')

print(f'Finished decoding {subject_name=}, {mode=}')

# %% ---- 2025-07-21 ------------------------
# Pending

# %%
