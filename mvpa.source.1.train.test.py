"""
File: mvpa.source.1.train.test.py
Author: Chuncheng Zhang
Date: 2025-09-16
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Train and test on the source level.
    The data is prepared by the mvpa.source.1.py script.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-16 ------------------------
# Requirements and constants
from mne.decoding import get_coef
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

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

from util.easy_import import *
from util.io.file import load, save

# %%
subject_directory = Path("./rawdata/S07_20231220")

# parse = argparse.ArgumentParser('Compute TFR')
# parse.add_argument('-s', '--subject-dir', required=True)
# args = parse.parse_args()
# subject_directory = Path(args.subject_dir)

subject_name = subject_directory.name

data_directory = Path(f'./data/fsaverage/{subject_name}')

# %% ---- 2025-09-16 ------------------------
# Function and class


# %% ---- 2025-09-16 ------------------------
# Play ground
all_data = load(data_directory.joinpath('X-y-times-groups.dump'))
X = all_data['X']
y = all_data['y']
times = all_data['times']
groups = all_data['groups']
print(f'{X.shape=}, {y.shape=}, {times.shape=}, {groups.shape=}')


# %% ---- 2025-09-16 ------------------------
# Pending
clf = make_pipeline(
    StandardScaler(),
    LinearModel(LogisticRegression())
)

scoring = make_scorer(accuracy_score, greater_is_better=True)

time_decod = SlidingEstimator(
    clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

cv = LeaveOneGroupOut()

# Scores for each group, shape is (n_groups, n_times)
scores = cross_val_multiscore(
    time_decod, X, y, groups=groups, cv=cv, n_jobs=n_jobs)

# Mean scores across cross-validation splits
avg_scores = np.mean(scores, axis=0)

# Train again on the whole data
time_decod.fit(X, y)
try:
    # Coef of time_decod, shape is (n_features, n_classes, n_times)
    coef = get_coef(time_decod)
except Exception:
    coef = None
trained = {
    'scores': scores,
    'coef': coef
}
save(trained, data_directory.joinpath('trained.dump'))

# %% ---- 2025-09-16 ------------------------
# Pending

# %%


# %%
