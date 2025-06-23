"""
File: ersp-alpha-band-statistic.py
Author: Chuncheng Zhang
Date: 2025-06-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis the alpha band's ERSP across subjects.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-23 ------------------------
# Requirements and constants
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from rich import print
from pathlib import Path
from tqdm.auto import tqdm


# %% ---- 2025-06-23 ------------------------
# Function and class
def find_h5_files():
    directory = Path('./data/erd')
    pattern = 'ERSP Alpha Band*.h5'
    found = list(directory.rglob(pattern))
    return found


def concat_h5_files(found: list[Path]):
    dfs = []
    for f in tqdm(found, 'Read hdf'):
        df = pd.read_hdf(f)
        df['subject'] = f.parent.name
        dfs.append(df)
    df = pd.concat(dfs)
    df.index = pd.Index(range(len(df)))
    return pd.concat(dfs)


# %% ---- 2025-06-23 ------------------------
# Play ground
found = find_h5_files()
print(found)
table = concat_h5_files(found)
table['value'] *= 10
print(table)

averaged = table.groupby(['condition', 'time', 'channel', 'ch_type'], observed=True)[
    'value'].mean().reset_index()
averaged['subject'] = 'Averaged'
print(averaged)

table = pd.concat([table, averaged])
print(table)

# %% ---- 2025-06-23 ------------------------
# Pending
subjects = sorted(list(table['subject'].unique()))
conditions = sorted(list(table['condition'].unique()))
rows = len(subjects)
cols = len(conditions)
scatter_kwargs = dict(cmap='viridis', marker='s', vmin=-10, vmax=5)
sz_unit = 5  # inch

for ch_type in table['ch_type'].unique():
    print(ch_type)

    fig, axes = plt.subplots(
        rows, cols+1, figsize=(sz_unit*cols, sz_unit*rows),
        gridspec_kw={"width_ratios": [10] * cols + [1]}
    )

    for subject, condition in tqdm(itertools.product(subjects, conditions), 'Product subjects & conditions'):
        query = '&'.join([f'ch_type=="{ch_type}"',
                          f'condition=="{condition}"',
                          f'subject=="{subject}"'])
        df = table.query(query)

        i = subjects.index(subject)
        j = conditions.index(condition)
        ax = axes[i, j]

        ax.scatter(df['time'], df['channel'], c=df['value'], **scatter_kwargs)
        ax.set_title(f'ERSP @event: {condition} @sub: {subject}')
        ax.set_xlabel('Time (s)')
        if j == 0:
            ax.set_ylabel('Channel')
        else:
            ax.set_yticks([])

    for i in range(rows):
        fig.colorbar(axes[i, 0].collections[0], cax=axes[i, cols],
                     orientation='vertical').ax.set_yscale('linear')

    fig.tight_layout()
    plt.show()


# %% ---- 2025-06-23 ------------------------
# Pending

# %%


# %%
