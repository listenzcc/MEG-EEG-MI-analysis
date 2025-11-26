"""
File: summary.all.decoding.results.py
Author: Chuncheng Zhang
Date: 2025-11-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary all the decoding results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-26 ------------------------
# Requirements and constants
from util.easy_import import *


# %% ---- 2025-11-26 ------------------------
# Function and class
def read_csv(fname):
    df = pd.read_csv(fname, index_col=0)
    return fix_column_contents(df)


def fix_column_contents(df):
    df['mode'] = df['mode'].map(lambda e: e.upper())
    df['mode'] = df['mode'].map(lambda e: 'COMBINE' if e == 'ALL' else e)
    df['subject'] = df['subject'].map(lambda e: e[:3].upper())

    e = 'confusionMatrix'
    if e in df.columns:
        df[e] = df[e].map(string_to_numpy)

    return df


def string_to_numpy(array_str):
    import re
    # 匹配所有行
    rows = re.findall(r'\[([^\]]+)\]', array_str)
    array_data = []

    for row in rows:
        # 提取每行中的数字
        numbers = re.findall(r'[\d.]+', row)
        if numbers:
            array_data.append([float(num) for num in numbers])

    return np.array(array_data)


# %% ---- 2025-11-26 ------------------------
# Play ground

# %%
acc_fbcsp = read_csv('./data/decoding-fbcsp.csv')
acc_fbcnet = read_csv('../MEG-EEG-MI-fbcnet/data/decoding-fbcnet.csv')
display(acc_fbcsp)
display(acc_fbcnet)

# %%
acc_freq = read_csv('./data/decoding-on-freq.csv')
acc_time = read_csv('./data/decoding-on-time.csv')
display(acc_freq)
display(acc_time)

# %%
conf_fbcsp = read_csv('./data/decoding-fbcsp-confusion-matrix.csv')
conf_fbcnet = read_csv(
    '../MEG-EEG-MI-fbcnet/data/decoding-fbcnet-confusion-matrix.csv')
display(conf_fbcsp)
display(conf_fbcnet)


# %%
sns.set_theme(context='paper', style='ticks', font_scale=1)

TASK_TABLE = {
    '1': 'Hand',
    '2': 'Wrist',
    '3': 'Elbow',
    '4': 'Shoulder',
    '5': 'Rest'
}

# %%
df = pd.concat([conf_fbcsp, conf_fbcnet])

# Absolute
fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=200)
ticklabels = list(TASK_TABLE.values())
for i, mode in enumerate(['EEG', 'MEG', 'COMBINE']):
    for j, method in enumerate(['FBCSP', 'FBCNet']):
        mat = np.mean(df.query(f'mode=="{mode}" & method=="{method}"')[
                      'confusionMatrix'].to_numpy(), axis=0)
        if i == 0 and j == 0:
            base_mat = mat
        ax = axes[j][i]
        g = sns.heatmap(mat, vmin=0.1, vmax=0.6, annot=True,
                        cmap='Blues',
                        xticklabels=ticklabels if j == 1 else False,
                        yticklabels=ticklabels if i == 0 else False,
                        fmt='.2f',
                        annot_kws={'color': 'white'},
                        cbar=True,
                        cbar_kws={"shrink": 0.7},
                        ax=ax)
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        ax.set_title(f'{mode=} | {method=}')
plt.tight_layout()
plt.show()

# Relative
fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=200)
ticklabels = list(TASK_TABLE.values())
for i, mode in enumerate(['EEG', 'MEG', 'COMBINE']):
    for j, method in enumerate(['FBCSP', 'FBCNet']):
        mat = np.mean(df.query(f'mode=="{mode}" & method=="{method}"')[
                      'confusionMatrix'].to_numpy(), axis=0)
        mat -= base_mat
        ax = axes[j][i]
        g = sns.heatmap(mat, vmin=-0.2, vmax=0.2, annot=True,
                        cmap='RdBu',
                        xticklabels=ticklabels if j == 1 else False,
                        yticklabels=ticklabels if i == 0 else False,
                        fmt='.2f',
                        annot_kws={'color': 'gray' if i ==
                                   0 and j == 0 else 'white'},
                        cbar=True,
                        cbar_kws={"shrink": 0.7},
                        ax=ax)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-0.2, 0, 0.2])
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        ax.set_title(f'{mode=} | {method=}')
plt.tight_layout()
plt.show()

# %%
df = pd.concat([acc_fbcnet, acc_fbcsp])
sns.barplot(df, x='method', hue='mode', y='accuracy',
            hue_order=['EEG', 'MEG', 'COMBINE'])
plt.show()

# %%
sns.lineplot(acc_time, x='t', hue='mode', y='accuracy', style='method')
plt.axvline(x=0, linestyle=':', color='black')
plt.axhline(y=0.2, linestyle=':', color='black')
plt.show()

# %%
sns.lineplot(acc_freq, x='freq', hue='mode', y='accuracy')
plt.axhline(y=0.2, linestyle=':', color='black')
plt.show()

# %% ---- 2025-11-26 ------------------------
# Pending


# %% ---- 2025-11-26 ------------------------
# Pending
