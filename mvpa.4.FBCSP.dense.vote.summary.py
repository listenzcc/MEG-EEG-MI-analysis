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
MODE = 'eeg'
DATA_DIR = Path(f'./data/MVPA.FBCSP.vote.{MODE}.dense')

# %% ---- 2025-12-18 ------------------------
# Function and class


# %% ---- 2025-12-18 ------------------------
# Play ground
raw_dump_files = sorted(list(DATA_DIR.rglob('*.dump')))
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

        accs.append((freq, acc, acc2, subject))

accs = pd.DataFrame(accs, columns=['freq', 'acc', 'acc2', 'subject'])
print(accs)

# %%
sns.lineplot(accs, x='freq', y='acc')
plt.show()

# %% ---- 2025-12-18 ------------------------
# Pending

# %%
