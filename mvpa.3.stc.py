"""
File: mvpa.3.stc.py
Author: Chuncheng Zhang
Date: 2025-07-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    1. Read data.
    2. Compute STC.
    3. Select label (ROI).
    4. Get stc's data and perform MVPA.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending

Reference:
- <https://mne.tools/stable/auto_examples/inverse/dics_epochs.html#sphx-glr-auto-examples-inverse-dics-epochs-py>
"""

# %%
import sys

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, make_scorer
from mne.time_frequency import csd_tfr
from mne.beamformer import apply_dics_tfr_epochs, make_dics

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

from util.subject_fsaverage import SubjectFsaverage
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory
from util.easy_import import *
from util.bands import Bands


# subject_directory = Path("./rawdata/S01_20220119")
subject_directory = Path("./rawdata/S07_20231220")

# parse = argparse.ArgumentParser('Compute TFR')
# parse.add_argument('-s', '--subject-dir', required=True)
# args = parse.parse_args()
# subject_directory = Path(args.subject_dir)

subject_name = subject_directory.name

data_directory = Path(f"./data/tfr-stc-beta/{subject_name}")
data_directory.mkdir(parents=True, exist_ok=True)

ROI_labels = [
    # meg-alpha
    ('alpha', 'precentral_5-lh'),
    ('alpha', 'precentral_8-lh'),
    ('alpha', 'postcentral_7-lh'),
    ('alpha', 'postcentral_9-lh'),
    ('alpha', 'postcentral_4-lh'),
    ('alpha', 'supramarginal_6-lh'),
    ('alpha', 'supramarginal_9-lh'),
    ('alpha', 'superiorparietal_5-lh'),
    ('alpha', 'superiorparietal_3-lh'),
    # meg-beta
    ('beta', 'postcentral_5-lh'),
    ('beta', 'postcentral_4-lh'),
    ('beta', 'superiorparietal_5-lh'),
    ('beta', 'precentral_4-lh'),
    ('beta', 'precentral_6-lh'),
    ('beta', 'precentral_8-lh'),
    ('beta', 'precentral_5-lh'),
    ('beta', 'precentral_3-lh'),
    ('beta', 'superiorfrontal_18-lh'),
    ('beta', 'superiorfrontal_17-lh'),
    ('beta', 'superiorfrontal_12-lh'),
    ('beta', 'caudalmiddlefrontal_6-lh'),
    ('beta', 'postcentral_7-lh'),
    ('beta', 'postcentral_3-lh'),
]

ROI_labels_df = pd.DataFrame(ROI_labels, columns=['band', 'label'])
ROI_labels_df

# %%


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
subject = SubjectFsaverage()
parc = "aparc_sub"
labels_parc = mne.read_labels_from_annot(
    subject.subject, parc=parc, subjects_dir=subject.subjects_dir
)
labels_parc_df = pd.DataFrame(
    [(e.name, e) for e in labels_parc], columns=["name", "label"]
)
labels_parc_df
label_obj = labels_parc_df.query('name=="precentral_5-lh"').iloc[0]['label']
label_obj

# %%


def compute_stc(epochs, fwd, freqs, tmin, tmax):
    epochs_tfr = epochs.compute_tfr(
        "morlet",
        freqs,
        n_cycles=freqs,
        return_itc=False,
        output="complex",
        average=False,
        n_jobs=n_jobs,
    )
    # epochs_tfr.crop(tmin=tmin, tmax=tmax)
    print(epochs_tfr)

    # Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
    # We are interested in increases in power relative to the baseline period, so
    # we will make a separate CSD for just that period as well.
    csd = csd_tfr(epochs_tfr, tmin=tmin, tmax=tmax)
    baseline_csd = csd_tfr(epochs_tfr, tmin=tmin, tmax=0)

    # compute scalar DICS beamfomer
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
        epochs_tfr, filters, return_generator=True
    )

    # epochs_stcs = list(epochs_stcs_generator)
    output_stcs = []
    for stcs in tqdm(epochs_stcs_generator, 'Processing stcs'):
        stc = stcs[0]
        stc = stc.in_label(label_obj)
        # Average across the freqs
        data = np.stack([s.data for s in stcs], axis=0)

        # Convert from complex to real
        # data = np.abs(data)
        # stc.data = data.mean(axis=0)

        stc.crop(tmin=-1, tmax=4)
        stc.apply_baseline((-1, 0))
        output_stcs.append(stc)

    return output_stcs


# %%
evts = ["1", "2", "3", "4", "5"]
mds, event_id = read_data()
eeg_epochs, meg_epochs, groups = concat_epochs(mds)
print(eeg_epochs)
print(meg_epochs)


# %%
subject = SubjectFsaverage()
subject.pipeline()
fwd_eeg = subject.read_forward_solution(eeg_epochs.info, "eeg")
fwd_meg = subject.read_forward_solution(meg_epochs.info, "meg")
print("src", subject.src)
print("bem", subject.bem)

# %%
bands = Bands()
freqs = [e for e in bands.mk_band_range("alpha")]
print("freqs", freqs)

tmin = -0.5
tmax = 4

epochs = meg_epochs.copy()
fwd = fwd_meg

stcs = compute_stc(epochs, fwd, freqs, tmin, tmax)
print(stcs)


# %%
cv = np.max(groups) + 1
# MEG signals: n_epochs, n_meg_channels, n_times
X = np.stack([stc.data for stc in stcs], axis=0)
X = np.abs(X)
y = epochs.events[:, 2]  # target
print(X.shape)
print(y.shape)

# %%

# Over time decoding
clf = make_pipeline(
    StandardScaler(), LinearModel(LogisticRegression(solver="liblinear"))
)

scoring = make_scorer(accuracy_score, greater_is_better=True)
time_decod = SlidingEstimator(
    clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

scores = cross_val_multiscore(
    time_decod, X, y, groups=groups, cv=cv, n_jobs=n_jobs)

print(scores)

# %%
plt.plot(stcs[0].times, np.mean(scores, axis=0))
plt.show()

# %%
# Assemble the classifier using scikit-learn pipeline
clf = make_pipeline(
    CSP(n_components=4, reg=None, log=True, norm_trace=False),
    StandardScaler(),  # In question
    LogisticRegression(solver="liblinear"),
)
res = cross_val_score(estimator=clf, X=X, y=y,
                      groups=groups, cv=cv, scoring=scoring)
print(res)

# %%
