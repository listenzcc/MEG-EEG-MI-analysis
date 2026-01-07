"""
File: mvpa.4.FBCSP.dense.vote.summary.analysis.py
Author: Chuncheng Zhang
Date: 2025-12-18
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the FBCSP dense voting results.
    Analysis why the eeg decodes better than meg in 0 - 15 Hz in freq bins.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-12-18 ------------------------
# Requirements and constants
from itertools import product
from sklearn.decomposition import PCA
from util.easy_import import *
from sklearn import metrics

# %%
eeg_data_dir = Path('./data/MVPA.FBCSP.vote.eeg.dense')
meg_data_dir = Path('./data/MVPA.FBCSP.vote.meg.dense')

output_dir = Path('./data/')

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
details = {
    'meg': {},
    'eeg': {}
}
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

        if freq > 15:
            break

        y_pred = decoded['y_pred']
        acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

        proba_array.append(decoded['y_proba'])
        proba = np.prod(proba_array, axis=0)
        y_pred = np.argmax(proba, axis=1) + 1
        acc2 = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

        accs.append((freq, acc, acc2, subject, mode))

    details[mode][subject[:3]] = {
        'proba_array': np.array(proba_array),
    }

accs = pd.DataFrame(accs, columns=['freq', 'acc', 'acc2', 'subject', 'mode'])
print(accs)

# %%
freqs

# %%
# accs.to_csv('./data/mvpa.FBCSP.dense.vote.summary.csv', index=False)
sns.lineplot(accs, x='freq', y='acc', hue='mode')
plt.show()

# %% ---- 2025-12-18 ------------------------
# Pending
accs

# %%
print(details)
df = pd.DataFrame(details)
print(df)
df.to_json(output_dir / 'decoding-results-in-bands.json')

# %%
df1 = pd.read_json(output_dir / 'decoding-results-in-bands.json')
display(df1)

# %%
subjects = [f'S{sub+1:02d}' for sub in range(10)]
x_positions = [np.mean(e) for e in freqs if np.mean(e) <= 15]
x_positions_int = [e for e in x_positions if e % 5 == 0]

event_id = 0

fig, axes = plt.subplots(len(subjects), 2, figsize=(6, 30))
for i, sub in enumerate(subjects):
    # meg/eeg shape is (n_samples, n_freqs)
    meg = details['meg'][sub]['proba_array'][:, :, event_id].T
    eeg = details['eeg'][sub]['proba_array'][:, :, event_id].T
    axes[i][0].imshow(meg, extent=[min(x_positions),
                      max(x_positions), 0, meg.shape[0]],)
    axes[i][1].imshow(eeg, extent=[min(x_positions),
                      max(x_positions), 0, meg.shape[0]],)
    axes[i][0].set_title(f'MEG - ({sub})')
    axes[i][1].set_title(f'EEG - ({sub})')

    for ax in axes[i]:
        ax.set_aspect('auto')  # 保持自动纵横比
        ax.set_xticks(x_positions_int)
        ax.set_xlabel('Hz')
        ax.set_yticks([])

fig.tight_layout()
plt.show()

# %%
X = meg.copy()
pca = PCA(n_components=.95)  # 降维到2维
X_pca = pca.fit_transform(X)
print("解释方差比例:", pca.explained_variance_ratio_)
print("累计解释方差:", np.cumsum(pca.explained_variance_ratio_))

X = eeg.copy()
pca = PCA(n_components=.95)  # 降维到2维
X_pca = pca.fit_transform(X)
print("解释方差比例:", pca.explained_variance_ratio_)
print("累计解释方差:", np.cumsum(pca.explained_variance_ratio_))


# %%
event_ids = [0, 1, 2, 3, 4]
TASK_TABLE = {
    '1': 'Hand',
    '2': 'Wrist',
    '3': 'Elbow',
    '4': 'Shoulder',
    '5': 'Rest'
}
modes = ['meg', 'eeg']
pca_results = []
for event_id, mode, sub in product(event_ids, modes, subjects):
    # mat shape is (n_samples, n_freqs)
    mat = details[mode][sub]['proba_array'][:, :, event_id].T
    X = mat.copy()
    pca = PCA(n_components=.9)  # 降维到2维
    X_pca = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    evrc = np.cumsum(evr)
    pca_results.append({
        'evt': list(TASK_TABLE.values())[event_id],
        'mode': mode,
        'sub': sub,
        'n': len(evr) / ((273/35) if mode == 'meg' else 1),
        'evr': evr,
        'evrc': evrc,
    })
pca_results = pd.DataFrame(pca_results)
display(pca_results)
pca_results.to_csv(output_dir / 'decoding-compare-in-pca.csv')

# %%
sns.boxplot(pca_results, x='evt', y='n', hue='mode')
plt.show()
# %%
