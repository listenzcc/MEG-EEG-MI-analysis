"""
File: mvpa.5.FBCSP.py
Author: Chuncheng Zhang
Date: 2025-08-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    FBCSP decoding.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-08-04 ------------------------
# Requirements and constants
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from util.bands import Bands
from util.easy_import import *
from util.io.ds_directory_operation import find_ds_directories, read_ds_directory

# --------------------------------------------------------------------------------
mode = 'eeg'  # 'meg', 'eeg'
band_name = 'all'  # 'delta', 'theta', 'alpha', 'beta', 'gamma', 'all'
subject_directory = Path('./rawdata/S01_20220119')

# Use the arguments
parse = argparse.ArgumentParser('Compute TFR')
parse.add_argument('-s', '--subject-dir', required=True)
args = parse.parse_args()
subject_directory = Path(args.subject_dir)

# --------------------------------------------------------------------------------
# Prepare the paths
subject_name = subject_directory.name
data_directory = Path(f'./data/MVPA.FBCSP/{subject_name}')
data_directory.mkdir(parents=True, exist_ok=True)

# %% ---- 2025-08-04 ------------------------
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

        if mode in ['eeg', 'all']:
            md.eeg_epochs.load_data()
            md.eeg_epochs.filter(**filter_kwargs)
            md.eeg_epochs.crop(tmin=-1, tmax=4)
            md.eeg_epochs.apply_baseline((-1, 0))

        if mode in ['meg', 'all']:
            md.meg_epochs.load_data()
            md.meg_epochs.filter(**filter_kwargs)
            md.meg_epochs.crop(tmin=-1, tmax=4)
            md.meg_epochs.apply_baseline((-1, 0))

    return mds, event_id


def concat_epochs(mds: list[MyData]):
    groups = []
    for i, e in enumerate(mds):
        if mode in ['eeg', 'all']:
            groups.extend([i for _ in range(len(e.eeg_epochs))])
        else:
            groups.extend([i for _ in range(len(e.meg_epochs))])

    if mode in ['eeg', 'all']:
        eeg_epochs = mne.concatenate_epochs(
            [md.eeg_epochs for md in tqdm(mds, 'Concat EEG Epochs')])
    else:
        eeg_epochs = None

    if mode in ['meg', 'all']:
        meg_epochs = mne.concatenate_epochs(
            [md.meg_epochs for md in tqdm(mds, 'Concat MEG Epochs')])
    else:
        meg_epochs = None

    return eeg_epochs, meg_epochs, groups


# %% --------------------------------------------------------------------------------
# Load epochs
evts = ['1', '2', '3', '4', '5']
mds, event_id = read_data()
eeg_epochs, meg_epochs, groups = concat_epochs(mds)
print(f'{eeg_epochs=}')
print(f'{meg_epochs=}')

if mode == 'meg':
    epochs = meg_epochs.copy().pick_types(meg=True, ref_meg=False)
elif mode == 'eeg':
    epochs = eeg_epochs.copy()
else:
    raise ValueError(f'Unknown mode: {mode}')

print(f'{epochs=}')

# cv = np.max(groups)+1
# # MEG signal shape is (n_epochs, n_meg_channels, n_times)
# X = epochs.get_data(copy=False)
# y = epochs.events[:, 2]  # target
# print(f'{X=}, {y=}, {groups=}')

# %% ---- 2025-08-04 ------------------------
# Play ground
# Setup freq ranges


def filter_bank_csp(epochs, freq_ranges, groups):

    y = epochs.events[:, 2]
    cv = np.max(groups) + 1

    band_features = []
    band_scores = {}

    for l_freq, h_freq in tqdm(freq_ranges, 'Working with freq_ranges'):
        # 复制epochs并滤波
        with redirect_stdout(io.StringIO()):
            epochs_filt = epochs.copy().filter(l_freq=l_freq, h_freq=h_freq)

        X_filt = epochs_filt.get_data(copy=False)
        # X_filt = np.asarray(X_filt, dtype=np.float64)
        print(X_filt.shape, type(X_filt))
        print(y.shape, type(y))

        # 创建管道
        pipeline = Pipeline([
            ('CSP', CSP(n_components=6, reg=None, log=False, norm_trace=False)),
            # ('Scaler', StandardScaler()),  # 添加标准化
            ('LDA', LDA(solver='lsqr', shrinkage='auto'))
        ])

        scoring = make_scorer(accuracy_score, greater_is_better=True)

        # 交叉验证
        print('---- Cross validation ----')
        with redirect_stdout(io.StringIO()):
            cv_scores = cross_val_score(
                pipeline, X_filt, y,
                groups=groups,
                # cv=cv,
                cv=LeaveOneGroupOut(),
                # n_jobs=n_jobs,
                scoring=scoring
            )
        mean_score = np.mean(cv_scores)
        band_scores[(l_freq, h_freq)] = mean_score

        # 训练完整模型获取特征
        print('---- Get CSP features ----')
        with redirect_stdout(io.StringIO()):
            pipeline.fit(X_filt, y)
        features = pipeline.named_steps['CSP'].transform(X_filt)
        band_features.append(features)

    # 组合所有频带特征
    X_combined = np.concatenate(band_features, axis=1)

    # 评估组合特征
    combined_score = np.mean(cross_val_score(
        LDA(), X_combined, y,
        cv=LeaveOneGroupOut(),
        # cv=StratifiedKFold(n_splits=cv, shuffle=True),
        n_jobs=n_jobs
    ))

    return {
        'band_scores': band_scores,
        'combined_score': combined_score,
        'band_features': band_features,
        'combined_features': X_combined
    }


# freq_ranges = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
# res = filter_bank_csp(epochs, freq_ranges, groups)
# print(f'{res=}')
# res['band_scores']
# res['combined_score']


# %% ---- 2025-08-04 ------------------------
# Pending


class SafeFBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, freq_ranges, sfreq, n_components=4):
        self.freq_ranges = freq_ranges
        self.sfreq = sfreq
        self.n_components = n_components
        self.csp_filters = {}  # 存储各频带CSP

    def fit(self, X, y):
        sfreq = self.sfreq
        # X shape: (n_epochs, n_channels, n_times)
        for l_freq, h_freq in tqdm(self.freq_ranges, 'Fit'):
            with redirect_stdout(io.StringIO()):
                # 频带滤波（每个epoch独立滤波）
                X_filt = np.array([mne.filter.filter_data(
                    x, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq)
                    for x in X])

                # 训练CSP
                # ? Does CSP require its own scaler?
                csp = CSP(n_components=self.n_components)
                csp.fit(X_filt, y)
            self.csp_filters[(l_freq, h_freq)] = csp
        return self

    def transform(self, X):
        features = []
        sfreq = self.sfreq
        for (l_freq, h_freq), csp in tqdm(self.csp_filters.items(), 'Transform'):
            with redirect_stdout(io.StringIO()):
                X_filt = np.array([mne.filter.filter_data(
                    x, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq)
                    for x in X])
            features.append(csp.transform(X_filt))
        return np.concatenate(features, axis=1)


sfreq = epochs.info['sfreq']
freq_ranges = [(4, 8), (8, 12), (12, 16), (16, 20),
               (20, 24), (24, 28), (28, 32)]

# pipeline
pipeline = Pipeline([
    ('fb_csp', SafeFBCSP(sfreq=sfreq, freq_ranges=freq_ranges)),
    # ('sacle', StandardScaler()),
    ('sacle', RobustScaler()),  # Against outlier
    # ('svc', SVC(kernel='linear', C=0.01)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=50)),  # MI特征选择，k为保留特征数
    ('lda', LDA(solver='lsqr', shrinkage='auto'))
])

# MEG signal shape is (n_epochs, n_meg_channels, n_times)
X = epochs.get_data(copy=False)
y = epochs.events[:, 2]  # target
print(f'{X=}, {y=}, {groups=}')

cv_scores = cross_val_score(
    pipeline, X, y, groups=groups, cv=LeaveOneGroupOut())

print(f'{cv_scores=}')
np.save(data_directory.joinpath('cv_scores.npy'), cv_scores)

# %%
print(subject_name, cv_scores)


# %% ---- 2025-08-04 ------------------------
# Pending

# %%
# %%
