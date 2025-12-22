"""
File: mvpa.4.FBCSP.dense.vote.summary.py
Author: Chuncheng Zhang
Date: 2025-12-18
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the FBCSP dense voting results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-12-18 ------------------------
# Requirements and constants
from util.easy_import import *
from sklearn import metrics

# %%
eeg_data_dir = Path('./data/MVPA.FBCSP.vote.eeg.dense')
meg_data_dir = Path('./data/MVPA.FBCSP.vote.meg.dense')

# %% ---- 2025-12-18 ------------------------
# Function and class


# %% ---- 2025-12-18 ------------------------
# Play ground
raw_dump_files = sorted(list(eeg_data_dir.rglob('*.dump')))
raw_dump_files.extend(sorted(list(meg_data_dir.rglob('*.dump'))))
print(raw_dump_files)


# %% ---- 2025-12-18 ------------------------
# Pending
accs = []
for p in tqdm(raw_dump_files, 'Read files'):
    obj = joblib.load(p)
    y_true = obj['y_true']
    subject = obj['subject_name']
    mode = obj['mode']
    freqs = obj['freqs']

    proba_array = []

    for idx in tqdm(range(len(freqs)), 'Loop freqs'):
        decoded = obj[idx]
        freq = np.mean((decoded['fmin'], decoded['fmax']))
        y_pred = decoded['y_pred']
        acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

        proba_array.append(decoded['y_proba'])
        proba = np.prod(proba_array, axis=0)
        y_pred = np.argmax(proba, axis=1) + 1
        acc2 = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

        accs.append((freq, acc, acc2, subject, mode))

accs = pd.DataFrame(accs, columns=['freq', 'acc', 'acc2', 'subject', 'mode'])
print(accs)

# %%
accs.to_csv('./data/mvpa.FBCSP.dense.vote.summary.csv', index=False)
sns.lineplot(accs, x='freq', y='acc', hue='mode')
plt.show()

# %% ---- 2025-12-18 ------------------------
# Pending

# %%
